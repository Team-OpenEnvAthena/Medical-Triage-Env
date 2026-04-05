# server/environment.py - Medical Triage Environment Implementation
import random
from uuid import uuid4
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server.interfaces import Environment
from models import TriageAction, TriageObservation, TriageState, PatientCase
from patient_cases import PATIENT_CASES


class MedicalTriageEnvironment(Environment):
    """
    Medical Triage & Discharge Planning Environment.

    Three tasks of increasing difficulty:
      easy   — Triage prioritization: assign urgency 1/2/3.
      medium — Investigation ordering: select appropriate tests (multi-step).
      hard   — Full discharge decision: diagnosis, disposition, medications.

    Each instance is fully isolated per session.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    AVAILABLE_TESTS = [
        "ecg", "troponin", "cbc", "cxr", "ct_head", "ct_abdomen",
        "ultrasound", "urinalysis", "blood_culture", "lactate",
        "bnp", "inr", "electrolytes", "rapid_strep", "xray_ankle",
        "xray_leg", "blood_glucose", "compartment_pressure",
        "urine_culture", "bhcg", "lumbar_puncture", "endoscopy",
    ]

    def __init__(self):
        super().__init__()
        # TriageState tracks framework-level metadata (task name + episode lifecycle)
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task="easy",
        )
        self._current_patient: Optional[PatientCase] = None
        self._ordered_tests: list = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._safety_flags: list = []

    # ── OpenEnv required API ───────────────────────────────────────────────

    @property
    def state(self) -> TriageState:
        """
        Current episode state — @property per OpenEnv spec.
        Returns TriageState with episode_id, step_count, and current_task.
        Used by training frameworks to route task-specific reward functions.
        """
        return self._state

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> TriageObservation:
        """
        Reset environment for a new episode.

        Args:
            task: Task name — 'easy', 'medium', or 'hard'. Defaults to 'easy'.
            seed: Optional random seed for reproducibility.

        Returns:
            Initial TriageObservation with patient case and task instruction.
        """
        if seed is not None:
            random.seed(seed)

        chosen_task = task if task in ("easy", "medium", "hard") else "easy"

        self._current_patient = random.choice(PATIENT_CASES)
        self._ordered_tests = []
        self._cumulative_reward = 0.0
        self._done = False

        # Reset TriageState — new episode_id, step_count=0, task recorded
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task=chosen_task,
        )

        return self._make_observation(reward=None, done=False)

    def step(self, action: TriageAction) -> TriageObservation:
        """
        Execute one action and return the resulting observation.

        Updates step_count and current_task in state on every step.
        """
        self._state.step_count += 1
        # Update current_task in state so TRL/GRPO reward routing stays accurate
        self._state.current_task = action.task_type

        if action.task_type == "easy":
            reward, done = self._handle_easy_task(action)
        elif action.task_type == "medium":
            reward, done = self._handle_medium_task(action)
        elif action.task_type == "hard":
            reward, done = self._handle_hard_task(action)
        else:
            reward, done = 0.0, True

        self._cumulative_reward += reward
        self._done = done
        return self._make_observation(reward=reward, done=done)

    # ── Task handlers ──────────────────────────────────────────────────────

    def _handle_easy_task(self, action: TriageAction):
        """
        Task 1 (Easy): Triage Prioritization.
        Single step. Reward: 1.0 exact, 0.5 one tier off, 0.0 two tiers off.
        """
        if action.urgency_assignment is None:
            return 0.0, True

        correct  = self._current_patient.true_urgency
        assigned = action.urgency_assignment

        if assigned == correct:
            reward = 1.0
        elif abs(assigned - correct) == 1:
            reward = 0.5
        else:
            reward = 0.0

        return reward, True

    def _handle_medium_task(self, action: TriageAction):
        """
        Task 2 (Medium): Investigation Ordering.
        Multi-step. Order tests incrementally; pass [] to finalise.
        Partial reward per step = required coverage fraction minus waste penalty.
        """
        if action.ordered_investigations is None:
            return 0.0, True

        if len(action.ordered_investigations) == 0:
            return self._score_investigations(), True

        new_tests = [t for t in action.ordered_investigations if t not in self._ordered_tests]
        self._ordered_tests.extend(new_tests)

        required = set(self._current_patient.required_investigations)
        covered  = required & set(self._ordered_tests)
        partial  = len(covered) / max(len(required), 1)
        waste    = len([t for t in self._ordered_tests if t not in required])
        penalty  = waste * 0.05

        step_reward = max(0.0, round(partial - penalty, 4))
        done = required.issubset(set(self._ordered_tests))
        return step_reward, done

    def _score_investigations(self) -> float:
        """Final F1-style score when agent signals done with []."""
        required = set(self._current_patient.required_investigations)
        ordered  = set(self._ordered_tests)
        if not ordered:
            return 0.0
        tp        = len(required & ordered)
        precision = tp / len(ordered)
        recall    = tp / max(len(required), 1)
        if precision + recall == 0:
            return 0.0
        f1            = 2 * precision * recall / (precision + recall)
        waste_penalty = len(ordered - required) * 0.1
        return max(0.0, round(f1 - waste_penalty, 4))

    def _handle_hard_task(self, action: TriageAction):
        """
        Task 3 (Hard): Full Discharge Decision.
        Composite reward: diagnosis (0.3) + disposition (0.3)
        + medications (0.2) + follow-up (0.2).
        Safety penalty: -0.5 for discharging an immediate-urgency patient.
        """
        total = 0.0

        # 1. Diagnosis accuracy (0.3) — fuzzy keyword match
        if action.diagnosis:
            agent_dx = action.diagnosis.lower().replace("_", " ").replace("-", " ")
            true_dx  = self._current_patient.true_diagnosis.lower().replace("_", " ")
            keywords = [w for w in true_dx.split() if len(w) > 3]
            if keywords:
                hits   = sum(1 for kw in keywords if kw in agent_dx)
                total += 0.3 * (hits / len(keywords))
            elif agent_dx == true_dx:
                total += 0.3

        # 2. Disposition correctness (0.3)
        if action.disposition:
            if action.disposition.lower() == self._current_patient.correct_disposition.lower():
                total += 0.3

        # 3. Medication safety (0.2) — partial credit per matched safe med
        if action.prescribed_medications:
            safe = set(m.lower() for m in self._current_patient.safe_medications)
            prescribed = set(m.lower().replace("_", " ").replace("-", " ")
                             for m in action.prescribed_medications)
            prescribed_raw = set(m.lower() for m in action.prescribed_medications)
            all_prescribed = prescribed | prescribed_raw
            hits = sum(1 for s in safe if any(s in p or p in s for p in all_prescribed))
            if safe:
                total += 0.2 * min(1.0, hits / len(safe))

        # 4. Safety penalty: never discharge an immediate patient
        if action.disposition == "discharge" and self._current_patient.true_urgency == 1:
            total -= 0.5
            self._safety_flags.append(
                f"UNSAFE: discharged urgency-1 patient {self._current_patient.patient_id}")

        # 5. Follow-up appropriateness (0.2)
        if action.follow_up_days is not None:
            if action.disposition == "discharge" and 0 < action.follow_up_days <= 14:
                total += 0.2
            elif action.disposition == "admit":
                total += 0.2

        return max(0.0, min(1.0, round(total, 4))), True

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_observation(self, reward, done) -> TriageObservation:
        """
        Build TriageObservation from current episode state.
        partial_score lives here (agent-facing shaped reward signal),
        NOT in TriageState (which is framework-facing only).
        """
        patient_dict = None
        if self._current_patient:
            patient_dict = {
                "id":                   self._current_patient.patient_id,
                "age":                  self._current_patient.age,
                "sex":                  self._current_patient.sex,
                "chief_complaint":      self._current_patient.chief_complaint,
                "heart_rate":           self._current_patient.vitals.heart_rate,
                "blood_pressure":       self._current_patient.vitals.blood_pressure,
                "spo2":                 self._current_patient.vitals.spo2,
                "temperature":          self._current_patient.vitals.temperature,
                "respiratory_rate":     self._current_patient.vitals.respiratory_rate,
                "past_medical_history": self._current_patient.history,
                "allergies":            self._current_patient.allergies,
                "ordered_tests_so_far": list(self._ordered_tests),
            }

        return TriageObservation(
            done=done if done is not None else False,
            reward=reward,
            current_patient=patient_dict,
            available_investigations=self.AVAILABLE_TESTS,
            task_instruction=self._get_task_instruction(),
            partial_score=round(self._cumulative_reward, 4),  # shaped reward — observation only
            safety_flags=list(self._safety_flags),
        )

    def _get_task_instruction(self) -> str:
        task = self._state.current_task  # read from TriageState
        if task == "easy":
            return (
                "Task: Triage Prioritization.\n"
                "Set urgency_assignment: 1=Immediate (life-threatening), "
                "2=Urgent (needs prompt care), 3=Non-urgent (stable)."
            )
        elif task == "medium":
            return (
                "Task: Investigation Ordering.\n"
                "Order diagnostic tests using 'ordered_investigations'.\n"
                "Pass an empty list [] when you are done ordering.\n"
                "Avoid wasteful tests — they reduce your score."
            )
        elif task == "hard":
            return (
                "Task: Full Discharge Decision.\n"
                "Provide: diagnosis, disposition ('admit'/'discharge'), "
                "prescribed_medications, follow_up_days.\n"
                "WARNING: Never discharge a patient with urgency 1!"
            )
        return "Unknown task."
