# models.py - Medical Triage Environment Models
from typing import List, Optional, Dict, Any
from pydantic import Field, BaseModel
from openenv.core.env_server.types import Action, Observation, State


# ============================================================================
# PATIENT DATA MODELS
# ============================================================================

class Vitals(BaseModel):
    """Patient vital signs — some revealed at presentation, rest after assessment."""
    heart_rate: int = Field(..., description="Heart rate in bpm")
    blood_pressure: str = Field(..., description="BP as systolic/diastolic")
    spo2: int = Field(..., description="Oxygen saturation %")
    temperature: float = Field(..., description="Temperature in Celsius")
    respiratory_rate: int = Field(..., description="Breaths per minute")


class PatientCase(BaseModel):
    """Complete patient case with ground truth (hidden from agent)."""
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    vitals: Vitals
    history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)

    # Ground truth
    true_diagnosis: str
    true_urgency: int           # 1=immediate, 2=urgent, 3=non-urgent
    required_investigations: List[str]
    correct_disposition: str
    safe_medications: List[str]

    # Option C: What information is hidden at presentation
    # These fields are only revealed once specific tests are ordered
    hidden_history: List[str] = Field(
        default_factory=list,
        description="History items revealed only after patient assessment test"
    )
    hidden_vitals_detail: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional vital detail revealed after physical_exam"
    )

    # Option B: Test prerequisites — test can only be ordered after these are done
    test_prerequisites: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Maps test_name → [prerequisite tests that must be done first]"
    )

    # Option D: Test turnaround times in steps
    # 'rapid' tests return at next step (delay=1)
    # 'standard' tests return 2 steps later (delay=2)
    # 'slow' tests return 3 steps later (delay=3)
    test_turnaround: Dict[str, int] = Field(
        default_factory=dict,
        description="Maps test_name → steps until result available (1=rapid, 2=standard, 3=slow)"
    )


# ============================================================================
# ACTION
# ============================================================================

class TriageAction(Action):
    """
    Action the agent sends each step.

    ── HOW TO USE ──────────────────────────────────────────────────────────

    TASK: easy
      {"task_type": "easy", "urgency_assignment": 1}
      1=Immediate, 2=Urgent, 3=Non-urgent

    TASK: medium — Investigation ordering (multi-step with delays)
      Order tests (max 2 per step):
        {"task_type": "medium", "ordered_investigations": ["ecg", "troponin"]}
      Signal done when ready:
        {"task_type": "medium", "ordered_investigations": []}

    TASK: hard_investigate — Hard task phase 1 (order tests with delays)
      {"task_type": "hard_investigate", "ordered_investigations": ["ecg"]}
      {"task_type": "hard_investigate", "ordered_investigations": []}  ← move to discharge

    TASK: hard_discharge — Hard task phase 2 (final discharge decision)
      {"task_type": "hard_discharge",
       "diagnosis": "acute myocardial infarction",
       "disposition": "admit",
       "prescribed_medications": ["aspirin", "nitroglycerin"],
       "follow_up_days": 1}
    """

    task_type: str = Field(
        ...,
        description=(
            "'easy' | 'medium' | 'hard_investigate' | 'hard_discharge'"
        ),
        json_schema_extra={"enum": ["easy", "medium", "hard_investigate", "hard_discharge"]},
    )

    urgency_assignment: Optional[int] = Field(
        None,
        description="[easy] 1=Immediate, 2=Urgent, 3=Non-urgent",
        json_schema_extra={"minimum": 1, "maximum": 3},
    )

    ordered_investigations: Optional[List[str]] = Field(
        None,
        description=(
            "[medium/hard_investigate] Tests to order this step (max 2 per step). "
            "Pass [] to finalise ordering. "
            "Note: some tests have prerequisites and some results are delayed. "
            "Available: ecg, troponin, cbc, cxr, ct_head, ct_abdomen, ultrasound, "
            "urinalysis, blood_culture, lactate, bnp, inr, electrolytes, rapid_strep, "
            "xray_ankle, xray_leg, blood_glucose, compartment_pressure, "
            "urine_culture, bhcg, lumbar_puncture, endoscopy, physical_exam"
        ),
    )

    diagnosis: Optional[str] = Field(
        None,
        description="[hard_discharge] Primary diagnosis",
        examples=["acute myocardial infarction", "pneumonia", "appendicitis"],
    )
    disposition: Optional[str] = Field(
        None,
        description="[hard_discharge] 'admit' or 'discharge'",
        json_schema_extra={"enum": ["admit", "discharge"]},
    )
    prescribed_medications: Optional[List[str]] = Field(
        None,
        description="[hard_discharge] List of medications",
    )
    follow_up_days: Optional[int] = Field(
        None,
        description="[hard_discharge] Days until follow-up",
        json_schema_extra={"minimum": 0, "maximum": 30},
    )


# ============================================================================
# OBSERVATION
# ============================================================================

class TriageObservation(Observation):
    """
    What the agent sees after reset() or step().

    Option C: current_patient starts with LIMITED information.
    More details are revealed as tests are ordered.

    Option D: test_results_pending shows tests ordered but not yet returned.
    test_results shows results that have arrived.
    """
    current_patient: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Patient information visible so far. Starts limited — "
            "more detail revealed as tests are ordered and results arrive."
        )
    )
    available_investigations: List[str] = Field(
        default_factory=list,
        description="Tests currently orderable (prerequisites met)"
    )
    locked_investigations: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Tests not yet orderable. Maps test_name → [prerequisite tests still needed]. "
            "Option B: must complete prerequisites before ordering these."
        )
    )
    investigation_results: Optional[Dict[str, str]] = Field(
        None,
        description="Test results that have arrived and are ready to read."
    )
    pending_results: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Tests ordered but results not yet available. "
            "Maps test_name → steps_remaining until result arrives. "
            "Option D: plan your next actions while waiting."
        )
    )
    task_instruction: str = Field("", description="Current task instruction")
    partial_score: float = Field(0.0, description="Cumulative reward so far")
    safety_flags: List[str] = Field(default_factory=list)


# ============================================================================
# STATE
# ============================================================================

class TriageState(State):
    """
    Episode metadata for training framework reward routing.
    Extends base State (episode_id + step_count).
    """
    current_task: str = Field(
        "easy",
        description="Active task: 'easy' | 'medium' | 'hard_investigate' | 'hard_discharge'"
    )
