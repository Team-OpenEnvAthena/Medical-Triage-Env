# models.py - Medical Triage Environment Models
from typing import List, Optional, Dict, Any
from pydantic import Field, BaseModel
from openenv.core.env_server.types import Action, Observation, State


# ============================================================================
# PATIENT DATA MODELS (internal)
# ============================================================================

class Vitals(BaseModel):
    heart_rate: int = Field(..., description="Heart rate in bpm")
    blood_pressure: str = Field(..., description="BP as systolic/diastolic string")
    spo2: int = Field(..., description="Oxygen saturation %")
    temperature: float = Field(..., description="Temperature in Celsius")
    respiratory_rate: int = Field(..., description="Breaths per minute")


class PatientCase(BaseModel):
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    vitals: Vitals
    history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    true_diagnosis: str
    true_urgency: int
    required_investigations: List[str]
    correct_disposition: str
    safe_medications: List[str]


# ============================================================================
# ACTION
# Descriptions here appear as hints in the /web UI form.
# ============================================================================

class TriageAction(Action):
    """
    Action the agent sends each step.

    ── HOW TO USE ──────────────────────────────────────────────────────────

    TASK 1 — EASY (Triage):
      {"task_type": "easy", "urgency_assignment": 1}
      urgency_assignment: 1 = Immediate, 2 = Urgent, 3 = Non-urgent

    TASK 2 — MEDIUM (Investigation ordering, multi-step):
      Step 1+: {"task_type": "medium", "ordered_investigations": ["ecg", "troponin"]}
      Final:   {"task_type": "medium", "ordered_investigations": []}
      Available tests: ecg, troponin, cbc, cxr, ct_head, ct_abdomen,
        ultrasound, urinalysis, blood_culture, lactate, bnp, inr,
        electrolytes, rapid_strep, xray_ankle, xray_leg, blood_glucose,
        bhcg, lumbar_puncture, endoscopy, compartment_pressure, urine_culture

    TASK 3 — HARD (Full discharge decision):
      {
        "task_type": "hard",
        "diagnosis": "acute myocardial infarction",
        "disposition": "admit",
        "prescribed_medications": ["aspirin", "nitroglycerin"],
        "follow_up_days": 1
      }
      ⚠️ Never set disposition=discharge for a critically ill patient!
    """

    task_type: str = Field(
        ...,
        description=(
            "Which task / phase to perform:\n"
            "  'easy'             — Triage: set urgency_assignment (1/2/3).\n"
            "  'medium'           — Investigation ordering: set ordered_investigations.\n"
            "  'hard_investigate' — Hard phase 1: order tests, or send [] when done.\n"
            "  'hard_discharge'   — Hard phase 2: final discharge decision."
        ),
        examples=["easy", "medium", "hard_investigate", "hard_discharge"],
        json_schema_extra={"enum": ["easy", "medium", "hard_investigate", "hard_discharge"]},
    )

    # ── EASY task field ────────────────────────────────────────────────────
    urgency_assignment: Optional[int] = Field(
        None,
        description=(
            "[EASY task only] Urgency level for the patient. "
            "1 = Immediate (life-threatening, needs resuscitation NOW), "
            "2 = Urgent (serious but stable, seen within 30 min), "
            "3 = Non-urgent (minor complaint, can wait)."
        ),
        examples=[1, 2, 3],
        json_schema_extra={"minimum": 1, "maximum": 3},
    )

    # ── MEDIUM task field ──────────────────────────────────────────────────
    ordered_investigations: Optional[List[str]] = Field(
        None,
        description=(
            "[MEDIUM task only] List of diagnostic test codes to order this step. "
            "Pass an EMPTY LIST [] to signal you are done ordering. "
            "Available: ecg, troponin, cbc, cxr, ct_head, ct_abdomen, "
            "ultrasound, urinalysis, blood_culture, lactate, bnp, inr, "
            "electrolytes, rapid_strep, xray_ankle, xray_leg, blood_glucose, "
            "bhcg, lumbar_puncture, endoscopy, compartment_pressure, urine_culture."
        ),
        examples=[["ecg", "troponin", "cbc"], ["cxr", "blood_culture"], []],
    )

    # ── HARD task fields ───────────────────────────────────────────────────
    diagnosis: Optional[str] = Field(
        None,
        description=(
            "[HARD task only] Your primary diagnosis as a plain text string. "
            "Be specific — e.g. 'acute myocardial infarction', 'pneumonia', "
            "'ectopic pregnancy', 'anaphylaxis'."
        ),
        examples=["acute myocardial infarction", "pneumonia", "appendicitis"],
    )

    disposition: Optional[str] = Field(
        None,
        description=(
            "[HARD task only] Discharge decision. "
            "'admit' = keep in hospital, "
            "'discharge' = send home with instructions. "
            "⚠️ NEVER discharge a patient with SpO2 < 90% or BP < 90/60."
        ),
        examples=["admit", "discharge"],
        json_schema_extra={"enum": ["admit", "discharge"]},
    )

    prescribed_medications: Optional[List[str]] = Field(
        None,
        description=(
            "[HARD task only] List of medications to prescribe. "
            "Use generic names. Examples: aspirin, nitroglycerin, morphine, "
            "ceftriaxone, albuterol, furosemide, epinephrine, heparin."
        ),
        examples=[["aspirin", "nitroglycerin"], ["ceftriaxone", "oxygen"]],
    )

    follow_up_days: Optional[int] = Field(
        None,
        description=(
            "[HARD task only] Days until recommended follow-up appointment. "
            "Use 1-14 for discharged patients. "
            "For admitted patients, any value is accepted."
        ),
        examples=[1, 2, 7],
        json_schema_extra={"minimum": 0, "maximum": 30},
    )


# ============================================================================
# OBSERVATION
# ============================================================================

class TriageObservation(Observation):
    """What the agent sees after reset() or step()."""
    current_patient: Optional[Dict[str, Any]] = Field(
        None,
        description="Patient demographics, vitals, history, and complaints"
    )
    available_investigations: List[str] = Field(
        default_factory=list,
        description="Test codes the agent can order this episode"
    )
    investigation_results: Optional[Dict[str, str]] = Field(
        None,
        description="Results for tests ordered so far (medium/hard tasks)"
    )
    task_instruction: str = Field(
        "",
        description="Human-readable instruction for the current task"
    )
    partial_score: float = Field(
        0.0,
        description="Cumulative shaped reward accumulated so far this episode"
    )
    safety_flags: List[str] = Field(
        default_factory=list,
        description=(
            "Safety violations triggered this episode. "
            "Non-empty means the agent took a clinically unsafe action "
            "(e.g. discharging a critically ill patient). "
            "These cause a -0.5 reward penalty."
        )
    )


# ============================================================================
# STATE
# ============================================================================

class TriageState(State):
    """
    Episode metadata for the medical triage environment.
    Extends base State (episode_id + step_count) with current_task,
    used by TRL/GRPO to route task-specific reward functions.
    """
    current_task: str = Field(
        "easy",
        description=(
            "Active task: 'easy' | 'medium' | 'hard'. "
            "Used by training frameworks to route task-specific reward functions."
        )
    )
