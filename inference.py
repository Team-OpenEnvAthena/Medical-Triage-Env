#!/usr/bin/env python3
"""
Medical Triage Environment — Baseline Inference Script
=======================================================
Mandatory stdout format (exact):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Required environment variables:
  API_BASE_URL   LLM endpoint   (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model ID       (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
  BASE_URL       Space URL      (default: https://garima-mahato-medical-triage-env.hf.space)
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
    print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASK_MAX_STEPS   = {"easy": 3, "medium": 5, "hard": 8}
SUCCESS_THRESHOLD = 0.5
TEMPERATURE       = 0.1


# ── Mandatory log helpers ──────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)


# ── LLM ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an experienced emergency physician making rapid clinical decisions.
Respond with ONLY a valid JSON object — no markdown, no explanation.

For task_type="easy":
  {"task_type": "easy", "urgency_assignment": <1|2|3>}
  1=Immediate (life-threatening), 2=Urgent, 3=Non-urgent

For task_type="medium" (order one batch of tests per step):
  {"task_type": "medium", "ordered_investigations": ["test1", "test2"]}
  Pass empty list [] when done ordering.
  Available: ecg, troponin, cbc, cxr, ct_head, ct_abdomen, ultrasound,
             urinalysis, blood_culture, lactate, bnp, inr, electrolytes,
             rapid_strep, xray_ankle, xray_leg, blood_glucose, bhcg,
             lumbar_puncture, endoscopy, compartment_pressure, urine_culture

For task_type="hard":
  {"task_type": "hard", "diagnosis": "<string>", "disposition": "<admit|discharge>",
   "prescribed_medications": ["med1", "med2"], "follow_up_days": <int>}
  SAFETY: NEVER discharge a patient with SpO2 < 90% or SBP < 90.
""").strip()


def call_llm(task_type: str, patient: dict, ordered_so_far: List[str], step: int) -> dict:
    vitals = (
        f"HR {patient.get('heart_rate')} | BP {patient.get('blood_pressure')} | "
        f"SpO2 {patient.get('spo2')}% | Temp {patient.get('temperature')}°C | "
        f"RR {patient.get('respiratory_rate')}"
    )
    history_str   = ", ".join(patient.get("past_medical_history") or []) or "None"
    allergies_str = ", ".join(patient.get("allergies") or []) or "None"
    ordered_str   = ", ".join(ordered_so_far) if ordered_so_far else "None yet"

    user_msg = textwrap.dedent(f"""
    Step {step} | Task: {task_type}
    Patient: {patient.get('age')}yo {patient.get('sex')}
    Complaint: {patient.get('chief_complaint')}
    Vitals: {vitals}
    History: {history_str}
    Allergies: {allergies_str}
    Tests ordered so far: {ordered_str}
    Respond with ONLY a JSON action object.
    """).strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=300,
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
    else:
        spo2 = patient.get("spo2", 99)
        bp   = patient.get("blood_pressure", "120/80")
        sbp  = int(bp.split("/")[0]) if "/" in str(bp) else 120
        disp = "admit" if (spo2 < 95 or sbp < 100) else "discharge"
        return {
            "task_type": "hard",
            "diagnosis": "clinical assessment pending",
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
    else:
        return f"discharge(disp={data.get('disposition')},dx={str(data.get('diagnosis',''))[:30]})"


# ── Episode runner ─────────────────────────────────────────────────────────

async def run_episode(task_name: str) -> None:
    """Run one full multi-step episode for the given task."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    max_steps = TASK_MAX_STEPS[task_name]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with MedicalTriageEnv(base_url=BASE_URL) as env:
            result = await env.reset(task=task_name)
            obs    = result.observation
            patient = obs.current_patient or {}
            ordered_so_far: List[str] = []

            for step in range(1, max_steps + 1):
                if result.done:
                    break

                action_data = call_llm(task_name, patient, ordered_so_far, step)
                action_data["task_type"] = task_name  # enforce correct task

                if task_name == "medium":
                    new = action_data.get("ordered_investigations") or []
                    ordered_so_far.extend(t for t in new if t not in ordered_so_far)

                action = make_action(action_data)
                label  = action_label(action_data)

                result = await env.step(action)
                reward = result.reward or 0.0
                done   = result.done
                obs    = result.observation
                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=label, reward=reward, done=done, error=None)

                if done:
                    break

            score = max(0.0, min(1.0, max(rewards))) if rewards else 0.0
            success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ───────────────────────────────────────────────────────────────────

async def main() -> None:
    for task in ["easy", "medium", "hard"]:
        await run_episode(task)
        print("", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
