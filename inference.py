#!/usr/bin/env python3
"""
Medical Triage Environment — Baseline Inference Script

Handles Options B+C+D mechanics:
  - Reads locked_investigations to avoid ordering blocked tests
  - Reads pending_results to know what is still in the lab
  - Adapts strategy based on arrived results
  - Orders physical_exam early to reveal hidden history (Option C)
"""
import asyncio, json, os, sys, textwrap
import yaml
from typing import Dict, List, Optional
from openai import OpenAI
from client import MedicalTriageEnv
from models import TriageAction
from dotenv import load_dotenv
load_dotenv()  # Load .env file

def _load_max_steps() -> Dict[str, int]:
    """Read max_steps from openenv.yaml — single source of truth."""
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    try:
        with open(yaml_path) as f:
            spec = yaml.safe_load(f)
        return {t["name"]: t["max_steps"] for t in spec.get("tasks", [])}
    except Exception as e:
        print(f"[WARN] Could not read openenv.yaml: {e}", file=sys.stderr)
        return {"easy": 3, "medium": 8, "hard": 12}

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
BASE_URL     = os.getenv("BASE_URL", "https://ishakhatana17-medical-triage-env.hf.space")
BENCHMARK    = "medical_triage_env"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set.", file=sys.stderr); sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASK_MAX_STEPS = _load_max_steps()   # sourced from openenv.yaml
SUCCESS_THRESHOLD = 0.3
TEMPERATURE       = 0.1


# ── Log helpers ────────────────────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


# ── LLM ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an experienced emergency physician. Respond with ONLY valid JSON — no markdown.

IMPORTANT MECHANICS (read carefully):
1. MAX 2 TESTS PER STEP — do not request more than 2 at once.
2. PREREQUISITES (Option B) — locked_investigations shows tests you cannot order yet.
   You must complete the prerequisite tests first.
3. DELAYED RESULTS (Option D) — pending_results shows tests ordered but not yet returned.
   Plan your next actions while waiting. Do not re-order pending tests.
4. HIDDEN INFORMATION (Option C) — order "physical_exam" early to reveal hidden clinical
   findings and full history. Many diagnoses require exam findings to unlock CT/LP.
5. Order tests logically — start with rapid tests (ecg, blood_glucose, physical_exam),
   then standard tests based on results, then confirmatory slow tests (CT, LP).

ACTION FORMATS:

{"task_type": "easy", "urgency_assignment": <1|2|3>}

{"task_type": "medium", "ordered_investigations": ["test1", "test2"]}   ← max 2
{"task_type": "medium", "ordered_investigations": []}                   ← done

{"task_type": "hard_investigate", "ordered_investigations": ["test1"]}  ← max 2
{"task_type": "hard_investigate", "ordered_investigations": []}         ← move to discharge

{"task_type": "hard_discharge",
 "diagnosis": "...", "disposition": "admit|discharge",
 "prescribed_medications": ["..."], "follow_up_days": <int>}

SAFETY: NEVER discharge SpO2<90%, BP<90/60, or urgency=1 patients.
""").strip()


def build_prompt(task_type: str, patient: dict, obs) -> str:
    vitals = (
        f"HR {patient.get('heart_rate')} | BP {patient.get('blood_pressure')} | "
        f"SpO2 {patient.get('spo2')}% | Temp {patient.get('temperature')}°C | "
        f"RR {patient.get('respiratory_rate')}"
    )
    history = ", ".join(patient.get("past_medical_history") or []) or "None disclosed yet"
    extra   = patient.get("additional_findings") or {}
    extra_str = "\n".join(f"  {k}: {v}" for k, v in extra.items()) if extra else "  None yet — order physical_exam"

    results = obs.investigation_results or {}
    results_str = "\n".join(f"  {k}: {v}" for k, v in results.items()) if results else "  None yet"

    pending = obs.pending_results or {}
    pending_str = ", ".join(f"{t}({s}step{'s' if s>1 else ''})" for t,s in pending.items()) or "none"

    available = obs.available_investigations or []
    locked    = obs.locked_investigations or {}
    locked_str = ", ".join(f"{t}(needs {','.join(p)})" for t,p in locked.items()) or "none"

    phase = patient.get("hard_phase", "")
    phase_hint = ""
    if task_type == "hard_investigate":
        phase_hint = "\nPhase: INVESTIGATION — order tests (max 2/step), send [] when ready."
    elif task_type == "hard_discharge":
        phase_hint = "\nPhase: DISCHARGE DECISION — all results available, make your final plan."

    ordered_so_far = patient.get("ordered_tests") or []
    do_not_order = ", ".join(ordered_so_far) if ordered_so_far else "none"

    return textwrap.dedent(f"""
    Task: {task_type}{phase_hint}
    Patient: {patient.get('age')}yo {patient.get('sex')} | ID: {patient.get('id')}
    Complaint: {patient.get('chief_complaint')}
    Vitals: {vitals}
    History: {history}
    Additional exam findings:
{extra_str}
    Allergies: {', '.join(patient.get('allergies') or []) or 'None'}

    ══ DO NOT RE-ORDER — already submitted to lab (pending or done) ══
    {do_not_order}
    ════════════════════════════════════════════════════════════════

    Results arrived:
{results_str}

    Pending (in lab, not yet returned): {pending_str}
    Available to order NOW: {', '.join(available[:15]) or 'none'}
    Locked (need prerequisite first): {locked_str}

    Respond with ONLY a JSON action. Max 2 tests. Never repeat already-ordered tests.
    """).strip()


def _call_llm_once(task_type: str, patient: dict, obs) -> dict:
    user_msg = build_prompt(task_type, patient, obs)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=TEMPERATURE, max_tokens=400,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())


def call_llm(task_type: str, patient: dict, obs) -> dict:
    """Call LLM with up to 2 retries. Validates urgency_assignment for easy task."""
    for attempt in range(3):
        try:
            result = _call_llm_once(task_type, patient, obs)
            # Validate easy task — urgency must be 1, 2, or 3
            if task_type == "easy":
                urgency = result.get("urgency_assignment")
                if urgency not in (1, 2, 3):
                    print(f"[DEBUG] easy: urgency={urgency} invalid, retry {attempt+1}", file=sys.stderr)
                    continue
            return result
        except Exception as e:
            print(f"[DEBUG] LLM error attempt {attempt+1}: {e}", file=sys.stderr)
    return _safe_default(task_type, patient, obs)


def _safe_default(task_type: str, patient: dict, obs) -> dict:
    available = obs.available_investigations or []
    pending   = obs.pending_results or {}
    ordered   = patient.get("ordered_tests") or []

    if task_type == "easy":
        spo2 = patient.get("spo2", 99)
        hr   = patient.get("heart_rate", 80)
        urgency = 1 if (spo2 < 90 or hr > 120) else (2 if spo2 < 95 else 3)
        return {"task_type": "easy", "urgency_assignment": urgency}

    elif task_type in ("medium", "hard_investigate"):
        if not ordered:
            # Always start with physical_exam if available (reveals hidden info)
            first = ["physical_exam"] if "physical_exam" in available else available[:2]
            return {"task_type": task_type, "ordered_investigations": first[:2]}
        if ordered:
            # If we have some results, signal done
            if len(ordered) >= 2:
                return {"task_type": task_type, "ordered_investigations": []}
            # Order next available tests
            not_ordered = [t for t in available if t not in ordered and t not in pending]
            return {"task_type": task_type, "ordered_investigations": not_ordered[:2] or []}

    else:  # hard_discharge
        spo2 = patient.get("spo2", 99)
        bp   = str(patient.get("blood_pressure", "120/80"))
        sbp  = int(bp.split("/")[0]) if "/" in bp else 120
        disp = "admit" if (spo2 < 95 or sbp < 100) else "discharge"
        return {
            "task_type": "hard_discharge",
            "diagnosis": "acute illness based on clinical findings",
            "disposition": disp,
            "prescribed_medications": ["supportive care"],
            "follow_up_days": 1,
        }


def make_action(data: dict) -> TriageAction:
    return TriageAction(**{k: v for k, v in data.items() if v is not None})

def action_label(data: dict) -> str:
    t = data.get("task_type", "?")
    if t == "easy":
        return f"triage(urgency={data.get('urgency_assignment')})"
    elif t == "medium":
        return f"order_tests({data.get('ordered_investigations', [])})"
    elif t == "hard_investigate":
        return f"hard_inv({data.get('ordered_investigations', [])})"
    elif t == "hard_discharge":
        return f"hard_discharge(disp={data.get('disposition')},dx={str(data.get('diagnosis',''))[:25]})"
    return f"unknown({t})"

def compute_score(task_name: str, rewards: List[float]) -> float:
    """
    Compute episode score clamped to [0.01, 0.99].

    easy:   single step → rewards[-1]
    medium: final F1 at [] step → rewards[-1]
    hard:   discharge reward (0.7 weight) + capped investigation contribution (0.3 weight).
            Investigation partial rewards are summed and capped at 0.3 so that
            wasted investigation steps cannot inflate the overall score.
            Score = min(sum(inv_rewards), 0.3) + discharge_reward
    """
    if not rewards:
        return 0.01
    if task_name == "hard" and len(rewards) >= 2:
        # Last reward is always the discharge step (done=True)
        discharge_reward = rewards[-1]
        inv_rewards = rewards[:-1]
        inv_contribution = min(sum(inv_rewards), 0.3)   # capped at 0.3
        raw = inv_contribution + discharge_reward
    else:
        raw = rewards[-1]
    return round(max(0.01, min(0.99, raw)), 4)


# ── Episode runners ────────────────────────────────────────────────────────

async def run_easy(env: MedicalTriageEnv) -> None:
    rewards, steps_taken, success, score = [], 0, False, 0.01
    log_start("easy", BENCHMARK, MODEL_NAME)
    try:
        result = await env.reset(task="easy")
        obs = result.observation
        patient = obs.current_patient or {}
        for step in range(1, TASK_MAX_STEPS["easy"] + 1):
            if result.done: break
            data = call_llm("easy", patient, obs)
            data["task_type"] = "easy"
            result = await env.step(make_action(data))
            obs = result.observation
            reward = result.reward if result.reward is not None else getattr(obs, "reward", 0.01) or 0.01
            rewards.append(reward); steps_taken = step
            log_step(step, action_label(data), reward, result.done,
                     (getattr(obs, "safety_flags", None) or [None])[0])
            if result.done: break
        score = compute_score("easy", rewards)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] easy: {e}", file=sys.stderr)
    finally:
        log_end(success, steps_taken, score, rewards)


async def run_medium(env: MedicalTriageEnv) -> None:
    rewards, steps_taken, success, score = [], 0, False, 0.01
    log_start("medium", BENCHMARK, MODEL_NAME)
    try:
        result = await env.reset(task="medium")
        obs = result.observation
        patient = obs.current_patient or {}
        max_steps = TASK_MAX_STEPS["medium"]

        for step in range(1, max_steps + 1):
            if result.done: break

            # Reserve last step for [] signal
            if step >= max_steps:
                data = {"task_type": "medium", "ordered_investigations": []}
            else:
                data = call_llm("medium", patient, obs)
                data["task_type"] = "medium"
                # Enforce max 2 tests
                if data.get("ordered_investigations"):
                    data["ordered_investigations"] = data["ordered_investigations"][:2]

            result = await env.step(make_action(data))
            obs = result.observation
            patient = obs.current_patient or patient
            reward = result.reward if result.reward is not None else getattr(obs, "reward", 0.01) or 0.01
            rewards.append(reward); steps_taken = step
            log_step(step, action_label(data), reward, result.done, None)
            if result.done: break

        score = compute_score("medium", rewards)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] medium: {e}", file=sys.stderr)
    finally:
        log_end(success, steps_taken, score, rewards)


async def run_hard(env: MedicalTriageEnv) -> None:
    rewards, steps_taken, success, score = [], 0, False, 0.01
    log_start("hard", BENCHMARK, MODEL_NAME)
    try:
        result = await env.reset(task="hard")
        obs = result.observation
        patient = obs.current_patient or {}
        phase = "investigate"
        max_steps = TASK_MAX_STEPS["hard"]

        for step in range(1, max_steps + 1):
            if result.done: break

            phase = patient.get("hard_phase", phase)

            if phase == "investigate":
                task_type = "hard_investigate"
                # Force transition on second-to-last step
                if step >= max_steps - 2:
                    data = {"task_type": task_type, "ordered_investigations": []}
                else:
                    data = call_llm(task_type, patient, obs)
                    data["task_type"] = task_type
                    if data.get("ordered_investigations"):
                        data["ordered_investigations"] = data["ordered_investigations"][:2]
            else:
                task_type = "hard_discharge"
                data = call_llm(task_type, patient, obs)
                data["task_type"] = task_type

            result = await env.step(make_action(data))
            obs = result.observation
            patient = obs.current_patient or patient
            reward = result.reward if result.reward is not None else getattr(obs, "reward", 0.01) or 0.01
            safety = getattr(obs, "safety_flags", []) or []
            rewards.append(reward); steps_taken = step
            log_step(step, action_label(data), reward, result.done, safety[0] if safety else None)
            if result.done: break

        score = compute_score("hard", rewards)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] hard: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
    finally:
        log_end(success, steps_taken, score, rewards)


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    async with MedicalTriageEnv(base_url=BASE_URL) as env:
        await run_easy(env)
    print("", flush=True)
    async with MedicalTriageEnv(base_url=BASE_URL) as env:
        await run_medium(env)
    print("", flush=True)
    async with MedicalTriageEnv(base_url=BASE_URL) as env:
        await run_hard(env)
    print("", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
