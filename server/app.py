# server/app.py - FastAPI Server for Medical Triage Environment
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Enable web UI at /web (required for HF Spaces App tab)
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

from openenv.core.env_server import create_app
from models import TriageAction, TriageObservation
from server.environment import MedicalTriageEnvironment

# Pass class (not instance) — create_app instantiates per session
app = create_app(
    MedicalTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="medical_triage_env",
    max_concurrent_envs=100,
)


@app.get("/tasks")
def list_tasks():
    """List available tasks and their action schemas."""
    return {
        "tasks": ["easy", "medium", "hard"],
        "task_descriptions": {
            "easy": (
                "Triage Prioritization — single step. "
                "Assign urgency 1 (Immediate), 2 (Urgent), or 3 (Non-urgent)."
            ),
            "medium": (
                "Investigation Ordering — multi-step. Order diagnostic tests "
                "(max 2/step) with lab delays and prerequisites. "
                "Send ordered_investigations=[] to finalise."
            ),
            "hard": (
                "Full Clinical Workup + Discharge — two-phase multi-step. "
                "Phase 1: use task_type='hard_investigate' to order tests. "
                "Phase 2: use task_type='hard_discharge' for final discharge plan."
            ),
        },
        "action_schema": {
            "easy": {
                "task_type": "easy",
                "urgency_assignment": "int: 1=Immediate | 2=Urgent | 3=Non-urgent",
            },
            "medium_ordering_step": {
                "task_type": "medium",
                "ordered_investigations": "list[str] — max 2 tests per step",
            },
            "medium_finalise": {
                "task_type": "medium",
                "ordered_investigations": "[] — triggers F1 grader and ends episode",
            },
            "hard_phase1_investigate": {
                "task_type": "hard_investigate",
                "ordered_investigations": "list[str] — max 2 tests per step",
            },
            "hard_phase1_finalise": {
                "task_type": "hard_investigate",
                "ordered_investigations": "[] — transitions to discharge phase",
            },
            "hard_phase2_discharge": {
                "task_type": "hard_discharge",
                "diagnosis": "str — primary diagnosis",
                "disposition": "str: admit | discharge",
                "prescribed_medications": "list[str] — medication names",
                "follow_up_days": "int — days until follow-up appointment",
            },
        },
        "mechanics": {
            "Option_B_prerequisites": (
                "Some tests require prerequisite tests first. "
                "locked_investigations in observation shows what is blocked and why."
            ),
            "Option_C_expanding_info": (
                "Order physical_exam to reveal hidden patient history and exam findings. "
                "Starts hidden, progressively revealed."
            ),
            "Option_D_lab_delays": (
                "Rapid tests (ecg, blood_glucose, physical_exam, xrays): 1 step delay. "
                "Standard tests (cbc, troponin, electrolytes, urinalysis...): 2 steps. "
                "Slow tests (ct_head, ct_abdomen, ultrasound, lp, blood_culture): 3 steps. "
                "pending_results in observation shows steps remaining per test."
            ),
        },
    }

# NOTE: /health is already provided by create_app internally — not duplicated here



@app.get("/")
def root():
    """Landing page — shows environment documentation."""
    from fastapi.responses import HTMLResponse
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🏥 Medical Triage Environment</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 860px; margin: 0 auto; padding: 2rem; color: #1a1a2e; background: #f8f9fa; }
  h1 { color: #c0392b; }
  h2 { color: #2c3e50; border-bottom: 2px solid #e74c3c; padding-bottom: 6px; }
  h3 { color: #e74c3c; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
  pre { background: #2c3e50; color: #ecf0f1; padding: 1rem; border-radius: 8px;
        overflow-x: auto; font-size: 0.85em; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th { background: #e74c3c; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 8px 12px; border-bottom: 1px solid #ddd; }
  tr:hover { background: #ffeaea; }
  .badge { display: inline-block; background: #27ae60; color: white;
           padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin: 2px; }
  .warn { background: #fff3cd; border-left: 4px solid #ffc107;
          padding: 0.75rem 1rem; border-radius: 4px; margin: 1rem 0; }
  .links { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
  .btn { background: #e74c3c; color: white; padding: 8px 18px; border-radius: 6px;
         text-decoration: none; font-weight: 600; }
  .btn:hover { background: #c0392b; }
  .btn-sec { background: #2c3e50; }
</style>
</head>
<body>
<h1>🏥 Medical Triage &amp; Discharge Planning Environment</h1>
<p>An <strong>OpenEnv</strong> RL environment where an AI agent acts as an emergency department clinician —
triaging patients, ordering diagnostic tests, and making discharge decisions.</p>
<p>
  <span class="badge">OpenEnv</span>
  <span class="badge">Healthcare</span>
  <span class="badge">RL Environment</span>
  <span class="badge">Meta Hackathon 2026</span>
</p>

<div class="links">
  <a class="btn" href="/web">🎮 Interactive Web UI</a>
  <a class="btn btn-sec" href="/docs">📖 API Docs</a>
  <a class="btn btn-sec" href="/tasks">📋 Task List</a>
  <a class="btn btn-sec" href="/health">✅ Health Check</a>
</div>

<h2>🔬 Three Tasks</h2>
<table>
<tr><th>Task</th><th>Difficulty</th><th>Steps</th><th>Reward</th></tr>
<tr><td>Triage Prioritization</td><td>Easy</td><td>1</td><td>0.0 / 0.5 / 1.0</td></tr>
<tr><td>Investigation Ordering</td><td>Medium</td><td>Multi-step</td><td>0.0–1.0 partial</td></tr>
<tr><td>Full Workup + Discharge</td><td>Hard</td><td>Multi-step (investigate → decide)</td><td>0.01–0.99 (inv 0.3 + discharge 0.7)</td></tr>
</table>

<h2>📋 How to Use the Web UI</h2>
<p>Go to <a href="/web">/web</a>, click <strong>Reset</strong>, then fill in the action form.</p>

<h3>Task 1 — Easy: Triage</h3>
<pre>{"task_type": "easy", "urgency_assignment": 1}
// urgency: 1=Immediate, 2=Urgent, 3=Non-urgent</pre>

<h3>Task 2 — Medium: Investigation Ordering</h3>
<pre>// Order tests:
{"task_type": "medium", "ordered_investigations": ["ecg", "troponin", "cbc"]}

// Signal done with empty list:
{"task_type": "medium", "ordered_investigations": []}</pre>
<p>Available tests: <code>ecg, troponin, cbc, cxr, ct_head, ct_abdomen, ultrasound,
urinalysis, blood_culture, lactate, bnp, inr, electrolytes, rapid_strep,
xray_ankle, xray_leg, blood_glucose, bhcg, lumbar_puncture, endoscopy</code></p>

<h3>Task 3 — Hard: Discharge Decision</h3>
<pre>// Phase 1 — order tests (repeat 2-3 times as results arrive):
{"task_type": "hard_investigate", "ordered_investigations": ["ecg", "physical_exam"]}

// Phase 1 — signal done investigating:
{"task_type": "hard_investigate", "ordered_investigations": []}

// Phase 2 — discharge decision:
{"task_type": "hard_discharge",
 "diagnosis": "acute myocardial infarction",
 "disposition": "admit",
 "prescribed_medications": ["aspirin", "nitroglycerin"],
 "follow_up_days": 1}</pre>
<div class="warn">⚠️ <strong>Safety rule:</strong> Never set <code>disposition: "discharge"</code>
if the patient has SpO2 &lt; 90% or BP &lt; 90/60. Penalty: −0.5 reward.</div>

<h2>⚡ Quick Start (Python)</h2>
<pre>pip install "openenv-core[core]>=0.2.2"
pip install git+https://huggingface.co/spaces/garima-mahato/medical-triage-env

import asyncio
from medical_triage_env import MedicalTriageEnv, TriageAction

async def main():
    async with MedicalTriageEnv(
        base_url="https://garima-mahato-medical-triage-env.hf.space"
    ) as env:
        result = await env.reset(task="easy")
        result = await env.step(TriageAction(task_type="easy", urgency_assignment=1))
        print(f"Reward: {result.reward}")

asyncio.run(main())</pre>

<h2>🔌 API Endpoints</h2>
<table>
<tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
<tr><td><a href="/reset">/reset</a></td><td>POST</td><td>Start a new episode</td></tr>
<tr><td><a href="/step">/step</a></td><td>POST</td><td>Submit an action</td></tr>
<tr><td><a href="/state">/state</a></td><td>GET</td><td>Episode metadata (task, step count)</td></tr>
<tr><td><a href="/health">/health</a></td><td>GET</td><td>Health check</td></tr>
<tr><td><a href="/tasks">/tasks</a></td><td>GET</td><td>Task list and action schemas</td></tr>
<tr><td><a href="/web">/web</a></td><td>GET</td><td>Interactive web UI</td></tr>
<tr><td><a href="/docs">/docs</a></td><td>GET</td><td>Swagger API documentation</td></tr>
</table>

<hr>
<p style="color:#888; font-size:0.85em;">
  Built for the Meta PyTorch × HuggingFace OpenEnv Hackathon 2026 •
  <a href="https://github.com/meta-pytorch/OpenEnv">OpenEnv</a>
</p>
</body>
</html>"""
    return HTMLResponse(content=html)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
