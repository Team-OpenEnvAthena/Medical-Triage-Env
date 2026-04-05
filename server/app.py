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
            "easy":   "Triage Prioritization — assign urgency 1/2/3.",
            "medium": "Investigation Ordering — select diagnostic tests (multi-step).",
            "hard":   "Full Discharge Decision — diagnosis, disposition, meds, follow-up.",
        },
        "action_schema": {
            "task_type":              "string: easy | medium | hard",
            "urgency_assignment":     "int 1-3 (easy only)",
            "ordered_investigations": "list[str] (medium only; pass [] to finish)",
            "diagnosis":              "str (hard only)",
            "disposition":            "str: admit | discharge (hard only)",
            "prescribed_medications": "list[str] (hard only)",
            "follow_up_days":         "int (hard only)",
        },
    }

# NOTE: /health is already provided by create_app internally — not duplicated here


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
