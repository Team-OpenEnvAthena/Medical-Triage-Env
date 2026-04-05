# server/environment.py - Medical Triage Environment Implementation
import random
from uuid import uuid4
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import TriageAction, TriageObservation, PatientCase
from patient_cases import PATIENT_CASES, get_case_by_id


class MedicalTriageEnvironment(Environment):
    """
    Medical Triage & Discharge Planning Environment.

    Three tasks of increasing difficulty:
      easy   — Triage prioritization: assign urgency 1/2/3.
      medium — Investigation ordering: select appropriate tests (multi-step).
      hard   — Full discharge decision: diagnosis, disposition, medications.
    """

    AVAILABLE_TESTS = [
        "ecg", "troponin", "cbc", "cxr", "ct_head", "ct_abdomen",
        "ultrasound", "urinalysis", "blood_culture", "lactate",
        "bnp", "inr", "electrolytes", "rapid_strep", "xray_ankle",
        "xray_leg", "blood_glucose", "compartment_pressure",
        "urine_culture", "bhcg", "lumbar_puncture", "endoscopy",
    ]

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = "easy"
        self._current_patient: Optional[PatientCase] = None
        self._ordered_tests = []
        self._cumulative_reward = 0.0
        self._done = False

    # ------------------------------------------------------------------
    # OpenEnv API — state MUST be a @property per the official spec
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Required by OpenEnv spec as @property."""
        return self._state

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> TriageObservation:
        """Reset environment. Pass task='easy'|'medium'|'hard' to select task."""
        if seed is not None:
            random.seed(seed)

        if task in ("easy", "medium", "hard"):
            self._current_task = task
        else:
            self._current_task = "easy"

        self._current_patient = random.choice(PATIENT_CASES)
        self._ordered_tests = []
        self._cumulative_reward = 0.0
        self._done = False
        # Re-create State with a fresh episode_id
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._make_observation(reward=None, done=False)

    def step(self, action: TriageAction) -> TriageObservation:
        """Execute action and return observation with reward."""
        # Mutate step_count directly — avoids reconstructing State objects
        self._state.step_count += 1
        self._current_task = action.task_type

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

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------

    def _handle_easy_task(self, action: TriageAction):
        """Triage prioritization — single step, deterministic grader."""
        if action.urgency_assignment is None:
            return 0.0, True

        correct = self._current_patient.true_urgency
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
        Investigation ordering — multi-step.
        Uses self._current_patient set during reset() (not hardcoded P001).
        Empty list signals end of ordering phase.
        """
        if action.ordered_investigations is None:
            return 0.0, True

        if len(action.ordered_investigations) == 0:
            return self._score_investigations(), True

        new_tests = [
            t for t in action.ordered_investigations
            if t not in self._ordered_tests
        ]
        self._ordered_tests.extend(new_tests)

        required = set(self._current_patient.required_investigations)
        covered = required & set(self._ordered_tests)
        partial = len(covered) / max(len(required), 1)

        waste = len([t for t in self._ordered_tests if t not in required])
        penalty = waste * 0.05

        step_reward = max(0.0, round(partial - penalty, 4))
        done = required.issubset(set(self._ordered_tests))
        return step_reward, done

    def _score_investigations(self):
        """Final F1-style score for investigation task."""
        required = set(self._current_patient.required_investigations)
        ordered = set(self._ordered_tests)
        if not ordered:
            return 0.0
        tp = len(required & ordered)
        precision = tp / len(ordered)
        recall = tp / max(len(required), 1)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        waste_penalty = len(ordered - required) * 0.1
        return max(0.0, round(f1 - waste_penalty, 4))

    def _handle_hard_task(self, action: TriageAction):
        """Full discharge decision — fuzzy diagnosis matching, composite reward."""
        total = 0.0

        # 1. Diagnosis accuracy (0.3) — fuzzy keyword match
        if action.diagnosis:
            agent_dx = action.diagnosis.lower().replace("_", " ").replace("-", " ")
            true_dx  = self._current_patient.true_diagnosis.lower().replace("_", " ")
            keywords = [w for w in true_dx.split() if len(w) > 3]
            if keywords:
                hits = sum(1 for kw in keywords if kw in agent_dx)
                total += 0.3 * (hits / len(keywords))
            elif agent_dx == true_dx:
                total += 0.3

        # 2. Disposition correctness (0.3)
        if action.disposition:
            if action.disposition.lower() == self._current_patient.correct_disposition.lower():
                total += 0.3

        # 3. Medication safety (0.2) — partial credit per safe med matched
        if action.prescribed_medications:
            safe = set(m.lower() for m in self._current_patient.safe_medications)
            prescribed = set(m.lower().replace("_", " ").replace("-", " ")
                             for m in action.prescribed_medications)
            prescribed_raw = set(m.lower() for m in action.prescribed_medications)
            all_prescribed = prescribed | prescribed_raw
            hits = sum(1 for s in safe if any(s in p or p in s for p in all_prescribed))
            if safe:
                total += 0.2 * min(1.0, hits / len(safe))

        # 4. Safety violation: discharging an immediate patient (-0.5)
        if action.disposition == "discharge" and self._current_patient.true_urgency == 1:
            total -= 0.5

        # 5. Follow-up appropriateness (0.2)
        if action.follow_up_days is not None:
            if action.disposition == "discharge" and 0 < action.follow_up_days <= 14:
                total += 0.2
            elif action.disposition == "admit":
                total += 0.2

        return max(0.0, min(1.0, round(total, 4))), True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self, reward, done) -> TriageObservation:
        patient_dict = None
        if self._current_patient:
            patient_dict = {
                "id": self._current_patient.patient_id,
                "age": self._current_patient.age,
                "sex": self._current_patient.sex,
                "chief_complaint": self._current_patient.chief_complaint,
                "heart_rate": self._current_patient.vitals.heart_rate,
                "blood_pressure": self._current_patient.vitals.blood_pressure,
                "spo2": self._current_patient.vitals.spo2,
                "temperature": self._current_patient.vitals.temperature,
                "respiratory_rate": self._current_patient.vitals.respiratory_rate,
                "past_medical_history": self._current_patient.history,
                "allergies": self._current_patient.allergies,
                "ordered_tests_so_far": list(self._ordered_tests),
            }

        return TriageObservation(
            done=done if done is not None else False,
            reward=reward,
            current_patient=patient_dict,
            available_investigations=self.AVAILABLE_TESTS,
            task_instruction=self._get_task_instruction(),
            partial_score=round(self._cumulative_reward, 4),
        )

    def _get_task_instruction(self) -> str:
        if self._current_task == "easy":
            return (
                "Task: Triage Prioritization.\n"
                "Set urgency_assignment: 1=Immediate (life-threatening), "
                "2=Urgent (needs prompt care), 3=Non-urgent (stable)."
            )
        elif self._current_task == "medium":
            return (
                "Task: Investigation Ordering.\n"
                "Order diagnostic tests using 'ordered_investigations'.\n"
                "Pass an empty list [] when you are done ordering.\n"
                "Avoid wasteful tests — they reduce your score."
            )
        elif self._current_task == "hard":
            return (
                "Task: Full Discharge Decision.\n"
                "Provide: diagnosis, disposition ('admit'/'discharge'), "
                "prescribed_medications, follow_up_days.\n"
                "WARNING: Never discharge a patient with urgency 1!"
            )
        return "Unknown task."
