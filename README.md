---
title: Medical Triage Environment
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# 🏥 Medical Triage & Discharge Planning Environment

An OpenEnv reinforcement learning environment where an AI agent acts as an emergency department clinician — triaging patients, ordering diagnostic tests, and making discharge decisions.

Built for the **Meta PyTorch × HuggingFace OpenEnv Hackathon 2026**.

---

## 🎮 Try It Now (No Setup)

**Interactive web UI:** https://garima-mahato-medical-triage-env.hf.space/web

Click **Reset** to get a patient case, then fill in the action form and click **Step**.

**API docs (Swagger):** https://garima-mahato-medical-triage-env.hf.space/docs

---

## 📋 How to Use the Web UI

### Step 1 — Click Reset
Loads a random patient case. You'll see their:
- Age, sex, chief complaint
- Vitals (HR, BP, SpO2, temperature, RR)
- Medical history and allergies
- Task instruction

### Step 2 — Fill in the Action Form

The form has these fields. **Only fill in the fields for your chosen task.**

---

### Task 1 — Easy: Triage Prioritization

| Field | Value |
|---|---|
| `task_type` | `easy` |
| `urgency_assignment` | `1`, `2`, or `3` |

**Urgency guide:**
- `1` = **Immediate** — life-threatening (chest pain + low BP, unconscious, severe bleeding)
- `2` = **Urgent** — serious but stable (high fever, moderate pain, SpO2 92%)
- `3` = **Non-urgent** — minor complaint (sore throat, ankle sprain, rash)

**Example — chest pain patient with SpO2 94%, BP 90/60:**
```json
{
  "task_type": "easy",
  "urgency_assignment": 1
}
```

**Reward:** `1.0` exact match · `0.5` one tier off · `0.0` two tiers off

---

### Task 2 — Medium: Investigation Ordering (multi-step)

Order tests one step at a time. Send an **empty list `[]`** when done.

| Field | Value |
|---|---|
| `task_type` | `medium` |
| `ordered_investigations` | List of test codes, or `[]` to finish |

**Available tests:**
```
ecg, troponin, cbc, cxr, ct_head, ct_abdomen, ultrasound,
urinalysis, blood_culture, lactate, bnp, inr, electrolytes,
rapid_strep, xray_ankle, xray_leg, blood_glucose, bhcg,
lumbar_puncture, endoscopy, compartment_pressure, urine_culture
```

**Example — chest pain patient (order ECG + troponin first):**

Step 1:
```json
{
  "task_type": "medium",
  "ordered_investigations": ["ecg", "troponin"]
}
```
Step 2:
```json
{
  "task_type": "medium",
  "ordered_investigations": ["cbc", "cxr"]
}
```
Step 3 (done):
```json
{
  "task_type": "medium",
  "ordered_investigations": []
}
```

**Reward:** Partial credit per step based on required tests covered minus wasteful tests.

---

### Task 3 — Hard: Full Discharge Decision

Provide a complete clinical plan in one step.

| Field | What to put |
|---|---|
| `task_type` | `hard` |
| `diagnosis` | Plain text diagnosis string |
| `disposition` | `"admit"` or `"discharge"` |
| `prescribed_medications` | List of medication names |
| `follow_up_days` | Number of days until follow-up |

**⚠️ Safety rule:** Never set `disposition: "discharge"` if the patient has SpO2 < 90% or BP < 90/60.

**Example — chest pain patient:**
```json
{
  "task_type": "hard",
  "diagnosis": "acute myocardial infarction",
  "disposition": "admit",
  "prescribed_medications": ["aspirin", "nitroglycerin", "heparin"],
  "follow_up_days": 1
}
```

**Example — sore throat patient:**
```json
{
  "task_type": "hard",
  "diagnosis": "viral pharyngitis",
  "disposition": "discharge",
  "prescribed_medications": ["paracetamol", "lozenges"],
  "follow_up_days": 5
}
```

**Reward breakdown:**
- Diagnosis accuracy: **0.3** (fuzzy keyword match)
- Correct disposition: **0.3**
- Appropriate medications: **0.2** (partial credit)
- Appropriate follow-up: **0.2**
- Safety violation (discharge critical patient): **−0.5**

---

## 🔬 Three Tasks

| Task | Difficulty | Steps | Reward |
|---|---|---|---|
| Triage Prioritization | Easy | 1 | 0.0 / 0.5 / 1.0 |
| Investigation Ordering | Medium | Multi-step | 0.0–1.0 partial |
| Discharge Planning | Hard | 1 | 0.0–1.0 composite |

---

## ⚡ Quick Start (Python)

```bash
pip install "openenv-core[core]>=0.2.2"
pip install git+https://huggingface.co/spaces/garima-mahato/medical-triage-env
```

```python
import asyncio
from medical_triage_env import MedicalTriageEnv, TriageAction

BASE_URL = "https://garima-mahato-medical-triage-env.hf.space"

async def main():
    async with MedicalTriageEnv(base_url=BASE_URL) as env:

        # ── Task 1: Easy ──────────────────────────────────────────────────
        result = await env.reset(task="easy")
        print(result.observation.current_patient["chief_complaint"])

        result = await env.step(TriageAction(
            task_type="easy",
            urgency_assignment=1
        ))
        print(f"Easy reward: {result.reward}")  # 0.0, 0.5, or 1.0

        # ── Task 2: Medium (multi-step) ───────────────────────────────────
        result = await env.reset(task="medium")

        result = await env.step(TriageAction(
            task_type="medium",
            ordered_investigations=["ecg", "troponin", "cbc"]
        ))
        print(f"Medium step reward: {result.reward}")

        result = await env.step(TriageAction(
            task_type="medium",
            ordered_investigations=[]  # signal done
        ))
        print(f"Medium final reward: {result.reward}")

        # ── Task 3: Hard ──────────────────────────────────────────────────
        result = await env.reset(task="hard")

        result = await env.step(TriageAction(
            task_type="hard",
            diagnosis="acute myocardial infarction",
            disposition="admit",
            prescribed_medications=["aspirin", "nitroglycerin"],
            follow_up_days=1
        ))
        print(f"Hard reward: {result.reward}")  # 0.0–1.0

asyncio.run(main())
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit an action |
| `/state` | GET | Episode metadata |
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks and schemas |
| `/web` | GET | Interactive web UI |
| `/docs` | GET | Swagger API docs |

---

## 📦 Install & Run Locally

```bash
git clone https://huggingface.co/spaces/garima-mahato/medical-triage-env
cd medical-triage-env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 🏗️ Environment Architecture

```
medical_triage_env/
├── models.py          # TriageAction, TriageObservation, TriageState
├── client.py          # MedicalTriageEnv (WebSocket client)
├── openenv.yaml       # OpenEnv manifest
└── server/
    ├── environment.py # Core logic + graders
    └── app.py         # FastAPI server
```

---

## 📊 Baseline Scores

Run against `Qwen/Qwen2.5-72B-Instruct`:

| Task | Score |
|---|---|
| Easy (triage) | ~0.80 |
| Medium (investigations) | ~0.65 |
| Hard (discharge) | ~0.55 |

Run baseline yourself:
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```
