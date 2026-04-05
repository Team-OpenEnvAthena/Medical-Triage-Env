#!/usr/bin/env python3
"""
Medical Triage Environment — Baseline Inference Script
=======================================================
Mandatory stdout format:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Required env vars:
  API_BASE_URL, MODEL_NAME, HF_TOKEN, BASE_URL
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import MedicalTriageEnv
from models import TriageAction

# ── Config ─────────────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.getenv("HF_TOKEN", "")
BASE_URL: str     = os.getenv("BASE_URL", "https://garima-mahato-medical-triage-env.hf.space")
BENCHMARK: str    = "medical_triage_env"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASK_MAX_STEPS    = {"easy": 3, "medium": 6, "hard": 10}
SUCCESS_THRESHOLD = 0.3
TEMPERATURE       = 0.1


# ── Log helpers ────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)


# ── LLM prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an experienced emergency physician making clinical decisions.
Respond with ONLY a valid JSON object — no markdown, no explanation.

ACTION TYPES:

1. task_type="easy"  →  triage urgency:
   {"task_type": "easy", "urgency_assignment": <1|2|3>}
   1=Immediate, 2=Urgent, 3=Non-urgent

2. task_type="medium"  →  order tests (or [] when done):
   {"task_type": "medium", "ordered_investigations": ["ecg", "troponin"]}
   {"task_type": "medium", "ordered_investigations": []}   ← signals done

3. task_type="hard_investigate"  →  hard task phase 1, order tests:
   {"task_type": "hard_investigate", "ordered_investigations": ["ecg", "troponin"]}
   {"task_type": "hard_investigate", "ordered_investigations": []}  ← move to discharge

4. task_type="hard_discharge"  →  hard task phase 2, final decision:
   {"task_type": "hard_discharge",
    "diagnosis": "acute myocardial infarction",
    "disposition": "admit",
    "prescribed_medications": ["aspirin", "nitroglycerin"],
    "follow_up_days": 1}

Available tests: ecg, troponin, cbc, cxr, ct_head, ct_abdomen, ultrasound,
  urinalysis, blood_culture, lactate, bnp, inr, electrolytes, rapid_strep,
  xray_ankle, xray_leg, blood_glucose, bhcg, lumbar_puncture, endoscopy,
  compartment_pressure, urine_culture

SAFETY: NEVER discharge (disposition="discharge") a patient with SpO2<90% or SBP<90.
""").strip()


def build_prompt(task_type: str, patient: dict, ordered_so_far: List[str],
                 test_results: dict, step: int, phase: str = "") -> str:
    vitals = (
        f"HR {patient.get('heart_rate')} | BP {patient.get('blood_pressure')} | "
        f"SpO2 {patient.get('spo2')}% | Temp {patient.get('temperature')}°C | "
        f"RR {patient.get('respiratory_rate')}"
    )
    history_str = ", ".join(patient.get("past_medical_history") or []) or "None"
    allergies_str = ", ".join(patient.get("allergies") or []) or "None"

    results_str = ""
    if test_results:
        results_str = "\nTest results:\n" + "\n".join(
            f"  {k}: {v}" for k, v in test_results.items()
        )
    ordered_str = ", ".join(ordered_so_far) if ordered_so_far else "none yet"

    phase_hint = ""
    if task_type == "hard_investigate":
        phase_hint = "\nPhase: INVESTIGATION — order tests, then send [] when ready to decide."
    elif task_type == "hard_discharge":
        phase_hint = "\nPhase: DISCHARGE DECISION — all test results available, make your final plan."

    return textwrap.dedent(f"""
    Step {step} | Task: {task_type}{phase_hint}
    Patient: {patient.get('age')}yo {patient.get('sex')}
    Complaint: {patient.get('chief_complaint')}
    Vitals: {vitals}
    History: {history_str} | Allergies: {allergies_str}
    Tests ordered: {ordered_str}{results_str}
    Respond with ONLY a JSON action.
    """).strip()


def call_llm(task_type: str, patient: dict, ordered_so_far: List[str],
             test_results: dict, step: int) -> dict:
    phase = patient.get("hard_phase", "")
    user_msg = build_prompt(task_type, patient, ordered_so_far, test_results, step, phase)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=400,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM error step {step}: {exc}", file=sys.stderr)
        return _safe_default(task_type, patient, ordered_so_far)


def _safe_default(task_type: str, patient: dict, ordered_so_far: List[str]) -> dict:
    if task_type == "easy":
        spo2 = patient.get("spo2", 99)
        hr   = patient.get("heart_rate", 80)
        urgency = 1 if (spo2 < 90 or hr > 120) else (2 if spo2 < 95 else 3)
        return {"task_type": "easy", "urgency_assignment": urgency}

    elif task_type == "medium":
        if ordered_so_far:
            return {"task_type": "medium", "ordered_investigations": []}
        return {"task_type": "medium", "ordered_investigations": ["ecg", "cbc"]}

    elif task_type == "hard_investigate":
        if ordered_so_far:
            return {"task_type": "hard_investigate", "ordered_investigations": []}
        return {"task_type": "hard_investigate", "ordered_investigations": ["ecg", "cbc", "cxr"]}

    else:  # hard_discharge
        spo2 = patient.get("spo2", 99)
        bp   = str(patient.get("blood_pressure", "120/80"))
        sbp  = int(bp.split("/")[0]) if "/" in bp else 120
        disp = "admit" if (spo2 < 95 or sbp < 100) else "discharge"
        return {
            "task_type": "hard_discharge",
            "diagnosis": "acute illness — see test results",
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
        tests = data.get("ordered_investigations", [])
        return f"hard_investigate({'[]' if not tests else tests})"
    elif t == "hard_discharge":
        return f"hard_discharge(disp={data.get('disposition')},dx={str(data.get('diagnosis',''))[:25]})"
    return f"unknown({t})"


def compute_score(task_name: str, rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    if task_name == "medium":
        return max(0.0, min(1.0, rewards[-1]))
    elif task_name == "hard":
        # Sum all rewards: investigation partial + final discharge score
        return max(0.0, min(1.0, sum(rewards)))
    else:
        return max(0.0, min(1.0, rewards[-1]))


# ── Episode runners ────────────────────────────────────────────────────────

async def run_easy(env: MedicalTriageEnv) -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task="easy", env=BENCHMARK, model=MODEL_NAME)
    try:
        result = await env.reset(task="easy")
        obs = result.observation
        patient = obs.current_patient or {}

        for step in range(1, TASK_MAX_STEPS["easy"] + 1):
            if result.done:
                break
            action_data = call_llm("easy", patient, [], {}, step)
            action_data["task_type"] = "easy"
            action = make_action(action_data)
            result = await env.step(action)
            obs = result.observation
            reward = (result.reward if result.reward is not None
                      else getattr(obs, "reward", 0.0) or 0.0)
            done = result.done
            safety = getattr(obs, "safety_flags", []) or []
            rewards.append(reward)
            steps_taken = step
            log_step(step, action_label(action_data), reward, done, safety[0] if safety else None)
            if done:
                break

        score = compute_score("easy", rewards)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] easy error: {e}", file=sys.stderr)
    finally:
        log_end(success, steps_taken, score, rewards)


async def run_medium(env: MedicalTriageEnv) -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    ordered_so_far: List[str] = []

    log_start(task="medium", env=BENCHMARK, model=MODEL_NAME)
    try:
        result = await env.reset(task="medium")
        obs = result.observation
        patient = obs.current_patient or {}
        max_steps = TASK_MAX_STEPS["medium"]

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Force [] on the second-to-last step so F1 grader always runs.
            # Medium never auto-terminates — only [] ends the episode.
            # Reserve the last step for the [] signal so we don't exhaust steps
            # without triggering the final score.
            if step >= max_steps - 1:
                action_data = {"task_type": "medium", "ordered_investigations": []}
            else:
                action_data = call_llm("medium", patient, ordered_so_far, {}, step)
                action_data["task_type"] = "medium"
                new = action_data.get("ordered_investigations") or []
                ordered_so_far.extend(t for t in new if t not in ordered_so_far)
                # If LLM already sent [], don't override — let it terminate early by design
                if not new and action_data.get("ordered_investigations") == []:
                    pass  # LLM chose to finalise early — allow it

            action = make_action(action_data)
            result = await env.step(action)
            obs = result.observation
            reward = (result.reward if result.reward is not None
                      else getattr(obs, "reward", 0.0) or 0.0)
            done = result.done
            rewards.append(reward)
            steps_taken = step
            log_step(step, action_label(action_data), reward, done, None)
            if done:
                break

        score = compute_score("medium", rewards)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] medium error: {e}", file=sys.stderr)
    finally:
        log_end(success, steps_taken, score, rewards)


async def run_hard(env: MedicalTriageEnv) -> None:
    """
    Hard task: multi-step.
    Phase 1 (hard_investigate): order tests, send [] when ready.
    Phase 2 (hard_discharge): final discharge decision with test evidence.
    """
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    ordered_so_far: List[str] = []
    test_results: dict = {}
    phase = "investigate"
    max_steps = TASK_MAX_STEPS["hard"]

    log_start(task="hard", env=BENCHMARK, model=MODEL_NAME)
    try:
        result = await env.reset(task="hard")
        obs = result.observation
        patient = obs.current_patient or {}

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Determine current phase from observation
            phase = (patient.get("hard_phase") or phase)

            if phase == "investigate":
                # Force transition on second-to-last step
                if step >= max_steps - 2 and not result.done:
                    action_data = {"task_type": "hard_investigate",
                                   "ordered_investigations": []}
                else:
                    action_data = call_llm("hard_investigate", patient,
                                           ordered_so_far, test_results, step)
                    action_data["task_type"] = "hard_investigate"
                    new = action_data.get("ordered_investigations") or []
                    ordered_so_far.extend(t for t in new if t not in ordered_so_far)

            else:  # discharge phase
                action_data = call_llm("hard_discharge", patient,
                                       ordered_so_far, test_results, step)
                action_data["task_type"] = "hard_discharge"

            action = make_action(action_data)
            result = await env.step(action)
            obs = result.observation
            reward = (result.reward if result.reward is not None
                      else getattr(obs, "reward", 0.0) or 0.0)
            done = result.done
            safety = getattr(obs, "safety_flags", []) or []

            # Update local state from new observation
            patient = obs.current_patient or patient
            new_results = patient.get("test_results", {})
            test_results.update(new_results)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_label(action_data), reward, done,
                     safety[0] if safety else None)
            if done:
                break

        score = compute_score("hard", rewards)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] hard error: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
    finally:
        log_end(success, steps_taken, score, rewards)


# ── Main ───────────────────────────────────────────────────────────────────

async def main() -> None:
    # Each task gets its own connection
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
