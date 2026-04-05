# server/environment.py - Medical Triage Environment Implementation
import random
from uuid import uuid4
from typing import Dict, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server.interfaces import Environment
from models import TriageAction, TriageObservation, TriageState, PatientCase
from patient_cases import PATIENT_CASES


# Simulated test results keyed by (patient_id, test_name)
# Returns realistic results based on the patient's true diagnosis
def simulate_test_result(test_name: str, patient: PatientCase) -> str:
    dx = patient.true_diagnosis.lower()

    RESULTS: Dict[str, Dict[str, str]] = {
        "ecg": {
            "acute_myocardial_infarction": "ST elevation leads II,III,aVF. Rate 108bpm. Consistent with inferior STEMI.",
            "ischemic_stroke": "Sinus rhythm. No ischaemic changes. Rate 88bpm.",
            "afib_with_rvr": "Atrial fibrillation. Rapid ventricular response ~145bpm.",
            "bradycardia": "Sinus bradycardia. Rate 48bpm. Prolonged PR interval.",
            "default": "Normal sinus rhythm. Rate 78bpm. No acute changes.",
        },
        "troponin": {
            "acute_myocardial_infarction": "Troponin I: 4.8 ng/mL (HIGH — ref <0.04). Rising pattern.",
            "aortic_dissection": "Troponin I: 0.08 ng/mL (mildly elevated — possible demand ischaemia).",
            "default": "Troponin I: 0.01 ng/mL (normal).",
        },
        "cbc": {
            "pneumonia": "WBC 18.4 (HIGH). Neutrophilia 85%. Hgb 13.1. Plt 310.",
            "upper_gi_bleed": "WBC 11.2. Hgb 7.8 (LOW). Plt 188. Haemorrhagic picture.",
            "appendicitis": "WBC 15.6 (HIGH). Neutrophilia 82%. Hgb 14.0.",
            "default": "WBC 8.9. Hgb 13.8. Plt 224. Normal differential.",
        },
        "cxr": {
            "pneumonia": "Right lower lobe consolidation. Air bronchograms present.",
            "acute_pulmonary_edema": "Bilateral pleural effusions. Cardiomegaly. Pulmonary oedema.",
            "default": "No acute cardiopulmonary abnormality.",
        },
        "ct_head": {
            "subarachnoid_hemorrhage": "Hyperdense blood in basal cisterns. No midline shift.",
            "ischemic_stroke": "Loss of grey-white differentiation left MCA territory. No haemorrhage.",
            "mild_traumatic_brain_injury": "No intracranial haemorrhage. No mass effect.",
            "default": "No acute intracranial pathology.",
        },
        "ct_abdomen": {
            "perforated_appendicitis": "Perforated appendix with periappendiceal abscess. Free air noted.",
            "appendicitis": "Dilated appendix 11mm. Periappendiceal fat stranding. No perforation.",
            "default": "No acute intra-abdominal pathology.",
        },
        "ultrasound": {
            "ectopic_pregnancy": "No intrauterine pregnancy. 2.8cm adnexal mass with ring-of-fire sign. Free fluid in POD.",
            "default": "No abnormality detected.",
        },
        "urinalysis": {
            "pyelonephritis": "Nitrites +ve. Leucocytes 3+. Bacteria +ve. WBCs >10/hpf.",
            "uncomplicated_uti": "Nitrites +ve. Leucocytes 2+. No casts.",
            "default": "Normal. No significant abnormality.",
        },
        "blood_culture": {
            "default": "Pending. Results in 48-72h.",
        },
        "lactate": {
            "perforated_appendicitis": "Lactate: 4.2 mmol/L (HIGH — ref <2.0). Tissue hypoperfusion.",
            "upper_gi_bleed": "Lactate: 3.8 mmol/L (HIGH).",
            "default": "Lactate: 1.1 mmol/L (normal).",
        },
        "bnp": {
            "acute_pulmonary_edema": "BNP: 1840 pg/mL (HIGH — ref <100). Severe heart failure.",
            "default": "BNP: 48 pg/mL (normal).",
        },
        "inr": {
            "upper_gi_bleed": "INR 2.8 (HIGH — patient on warfarin). PT 34s.",
            "default": "INR 1.1. PT 13s. Normal coagulation.",
        },
        "electrolytes": {
            "severe_hypoglycemia": "Na 138, K 3.8, Glucose 1.8 (CRITICALLY LOW), Cr 0.9.",
            "default": "Na 139, K 4.1, Cl 102, CO2 24, Cr 0.9, Glucose 5.4. All normal.",
        },
        "blood_glucose": {
            "severe_hypoglycemia": "Glucose: 1.6 mmol/L (CRITICALLY LOW — ref 3.9-7.8).",
            "default": "Glucose: 5.8 mmol/L (normal).",
        },
        "bhcg": {
            "ectopic_pregnancy": "Beta-hCG: 4200 mIU/mL (positive). Rapidly rising.",
            "default": "Beta-hCG: negative.",
        },
        "lumbar_puncture": {
            "subarachnoid_hemorrhage": "Xanthochromia present. RBC 32000 (non-clearing). Elevated opening pressure.",
            "default": "Clear CSF. Normal pressure, cells, protein, glucose.",
        },
        "rapid_strep": {
            "pharyngitis": "Rapid strep: POSITIVE. Group A Streptococcus detected.",
            "default": "Rapid strep: negative.",
        },
        "xray_ankle": {
            "default": "No fracture or dislocation. Soft tissue swelling noted laterally.",
        },
        "xray_leg": {
            "compartment_syndrome": "Fracture mid-shaft tibia. Significant soft tissue swelling.",
            "default": "No fracture identified.",
        },
        "compartment_pressure": {
            "compartment_syndrome": "Compartment pressure: 42 mmHg (HIGH — ref <30). Surgical emergency.",
            "default": "Compartment pressure within normal limits.",
        },
        "urine_culture": {
            "pyelonephritis": "E. coli >100,000 CFU/mL. Pan-sensitive. Sensitivity results pending.",
            "default": "Pending.",
        },
        "endoscopy": {
            "upper_gi_bleed": "Oesophageal varices grade III with active bleeding. Banding applied.",
            "default": "No acute pathology identified.",
        },
    }

    test_results = RESULTS.get(test_name, {})
    # Match against true diagnosis
    for key, val in test_results.items():
        if key != "default" and key in dx:
            return val
    return test_results.get("default", f"{test_name}: result normal.")


class MedicalTriageEnvironment(Environment):
    """
    Medical Triage & Discharge Planning Environment.

    Three tasks of increasing difficulty:

    easy   — Triage Prioritization (1 step):
               Assign urgency tier 1/2/3 to the patient.

    medium — Investigation Ordering (multi-step):
               Order diagnostic tests one batch per step.
               Send [] to finalise. Rewarded on F1 vs required tests.

    hard   — Full Clinical Workup + Discharge (multi-step):
               Phase 1 (steps 1-N): order diagnostic tests to gather evidence.
               Phase 2 (final step): issue a discharge action with diagnosis,
               disposition, medications, and follow-up.
               Reward = investigation quality (0.3) + discharge quality (0.7).
               The agent must investigate BEFORE deciding — test results are
               revealed progressively and inform the final decision.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    AVAILABLE_TESTS = [
        "ecg", "troponin", "cbc", "cxr", "ct_head", "ct_abdomen",
        "ultrasound", "urinalysis", "blood_culture", "lactate",
        "bnp", "inr", "electrolytes", "rapid_strep", "xray_ankle",
        "xray_leg", "blood_glucose", "compartment_pressure",
        "urine_culture", "bhcg", "lumbar_puncture", "endoscopy",
    ]

    # Max steps per task
    MAX_STEPS = {"easy": 3, "medium": 6, "hard": 10}

    def __init__(self):
        super().__init__()
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task="easy",
        )
        self._current_patient: Optional[PatientCase] = None
        self._ordered_tests: list = []
        self._test_results: dict = {}       # test_name → result string
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._safety_flags: list = []
        self._hard_phase: str = "investigate"  # "investigate" or "discharge"

    # ── OpenEnv API ────────────────────────────────────────────────────────

    @property
    def state(self) -> TriageState:
        return self._state

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> TriageObservation:
        if seed is not None:
            random.seed(seed)

        chosen_task = task if task in ("easy", "medium", "hard") else "easy"

        self._current_patient = random.choice(PATIENT_CASES)
        self._ordered_tests = []
        self._test_results = {}
        self._cumulative_reward = 0.0
        self._done = False
        self._safety_flags = []
        self._hard_phase = "investigate"

        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task=chosen_task,
        )

        return self._make_observation(reward=None, done=False)

    def step(self, action: TriageAction) -> TriageObservation:
        self._state.step_count += 1
        self._state.current_task = action.task_type

        if action.task_type == "easy":
            reward, done = self._handle_easy_task(action)

        elif action.task_type == "medium":
            reward, done = self._handle_medium_task(action)

        elif action.task_type == "hard_investigate":
            # Hard task phase 1: order tests
            reward, done = self._handle_hard_investigate(action)

        elif action.task_type == "hard_discharge":
            # Hard task phase 2: final discharge decision
            reward, done = self._handle_hard_discharge(action)

        else:
            reward, done = 0.0, True

        self._cumulative_reward += reward
        self._done = done
        return self._make_observation(reward=reward, done=done)

    # ── Easy ───────────────────────────────────────────────────────────────

    def _handle_easy_task(self, action: TriageAction):
        """
        Single step. Assign urgency tier 1/2/3.
        Reward: 1.0 exact, 0.5 one tier off, 0.0 two tiers off.
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

    # ── Medium ─────────────────────────────────────────────────────────────

    def _handle_medium_task(self, action: TriageAction):
        """
        Multi-step investigation ordering.
        Pass [] to finalise and receive F1 score.
        Never terminates immediately — always requires at least one ordering step.
        """
        if action.ordered_investigations is None:
            return 0.0, True

        if len(action.ordered_investigations) == 0:
            # Agent signals done — return final F1 score and terminate
            final = self._score_investigations()
            return final, True

        new_tests = [t for t in action.ordered_investigations if t not in self._ordered_tests]
        self._ordered_tests.extend(new_tests)
        # Reveal results for newly ordered tests
        for t in new_tests:
            self._test_results[t] = simulate_test_result(t, self._current_patient)

        required = set(self._current_patient.required_investigations)

        # Guard: empty required_investigations shouldn't happen, but handle gracefully
        if not required:
            waste   = len(self._ordered_tests)
            penalty = waste * 0.1
            return max(0.0, round(0.5 - penalty, 4)), False

        covered  = required & set(self._ordered_tests)
        partial  = len(covered) / len(required)
        waste    = len([t for t in self._ordered_tests if t not in required])
        penalty  = waste * 0.05
        step_reward = max(0.0, round(partial - penalty, 4))

        # NEVER auto-terminate on required coverage.
        # The ONLY way to end the medium episode is sending ordered_investigations=[].
        # This enforces multi-step behaviour — the agent must consciously decide
        # when it has enough evidence, preventing trivial 1-step completion.
        return step_reward, False

    def _score_investigations(self) -> float:
        required = set(self._current_patient.required_investigations)
        ordered  = set(self._ordered_tests)
        if not ordered:
            return 0.0
        tp        = len(required & ordered)
        precision = tp / len(ordered)
        recall    = tp / max(len(required), 1)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        waste_penalty = len(ordered - required) * 0.1
        return max(0.0, round(f1 - waste_penalty, 4))

    # ── Hard: Phase 1 — investigate ────────────────────────────────────────

    def _handle_hard_investigate(self, action: TriageAction):
        """
        Hard task phase 1: order tests to gather evidence.

        Each step reveals test results. Agent should order relevant tests
        before committing to a discharge decision.

        Pass ordered_investigations=[] to signal ready to discharge
        (transitions to phase 2 — next action must be hard_discharge).

        Partial reward per step based on required test coverage.
        """
        if action.ordered_investigations is None:
            # Treat as done ordering — transition to discharge phase
            self._hard_phase = "discharge"
            return 0.0, False

        if len(action.ordered_investigations) == 0:
            # Agent signals done ordering — move to discharge phase
            self._hard_phase = "discharge"
            inv_score = self._score_investigations()
            # Small partial reward for investigation quality (0.3 weight of total)
            return round(inv_score * 0.3, 4), False   # not done yet — still need discharge

        new_tests = [t for t in action.ordered_investigations if t not in self._ordered_tests]
        self._ordered_tests.extend(new_tests)
        # Reveal results for new tests immediately
        for t in new_tests:
            self._test_results[t] = simulate_test_result(t, self._current_patient)

        required = set(self._current_patient.required_investigations)
        covered  = required & set(self._ordered_tests)
        partial  = len(covered) / max(len(required), 1)
        waste    = len([t for t in self._ordered_tests if t not in required])
        penalty  = waste * 0.05
        step_reward = max(0.0, round(partial * 0.3 - penalty, 4))  # 0.3 weight

        # NEVER auto-transition. The only way to move to discharge phase
        # is sending ordered_investigations=[] explicitly.
        # This forces the agent to consciously decide when it has enough evidence.
        return step_reward, False   # never done during investigation phase

    # ── Hard: Phase 2 — discharge ──────────────────────────────────────────

    def _handle_hard_discharge(self, action: TriageAction):
        """
        Hard task phase 2: final discharge decision.

        Graded on:
          - Diagnosis accuracy  (0.3 of 0.7 discharge weight = 0.21 total)
          - Disposition correct  (0.3 of 0.7 = 0.21 total)
          - Medications         (0.2 of 0.7 = 0.14 total)
          - Follow-up           (0.2 of 0.7 = 0.14 total)

        Safety penalty: -0.5 for discharging an immediate-urgency patient.

        Total max reward across full episode = 0.3 (investigation) + 0.7 (discharge) = 1.0
        """
        discharge_score = 0.0

        # 1. Diagnosis accuracy (weight 0.3 of discharge portion)
        if action.diagnosis:
            agent_dx = action.diagnosis.lower().replace("_", " ").replace("-", " ")
            true_dx  = self._current_patient.true_diagnosis.lower().replace("_", " ")
            keywords = [w for w in true_dx.split() if len(w) > 3]
            if keywords:
                hits = sum(1 for kw in keywords if kw in agent_dx)
                discharge_score += 0.3 * (hits / len(keywords))
            elif agent_dx == true_dx:
                discharge_score += 0.3

        # 2. Disposition correctness (weight 0.3)
        if action.disposition:
            if action.disposition.lower() == self._current_patient.correct_disposition.lower():
                discharge_score += 0.3

        # 3. Medications (weight 0.2)
        if action.prescribed_medications:
            safe = set(m.lower() for m in self._current_patient.safe_medications)
            prescribed = set(m.lower().replace("_", " ").replace("-", " ")
                             for m in action.prescribed_medications)
            prescribed_raw = set(m.lower() for m in action.prescribed_medications)
            all_prescribed = prescribed | prescribed_raw
            hits = sum(1 for s in safe if any(s in p or p in s for p in all_prescribed))
            if safe:
                discharge_score += 0.2 * min(1.0, hits / len(safe))

        # 4. Safety penalty
        if action.disposition == "discharge" and self._current_patient.true_urgency == 1:
            discharge_score -= 0.5
            self._safety_flags.append(
                f"UNSAFE: discharged urgency-1 patient {self._current_patient.patient_id}"
            )

        # 5. Follow-up (weight 0.2)
        if action.follow_up_days is not None:
            if action.disposition == "discharge" and 0 < action.follow_up_days <= 14:
                discharge_score += 0.2
            elif action.disposition == "admit":
                discharge_score += 0.2

        # Scale discharge score by 0.7 weight
        final_discharge = max(0.0, min(0.7, round(discharge_score * 0.7, 4)))

        return final_discharge, True   # always terminal

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_observation(self, reward, done) -> TriageObservation:
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
                "test_results":         dict(self._test_results),
                "hard_phase":           self._hard_phase,
            }

        return TriageObservation(
            done=done if done is not None else False,
            reward=reward,
            current_patient=patient_dict,
            available_investigations=self.AVAILABLE_TESTS,
            investigation_results=dict(self._test_results) if self._test_results else None,
            task_instruction=self._get_task_instruction(),
            partial_score=round(self._cumulative_reward, 4),
            safety_flags=list(self._safety_flags),
        )

    def _get_task_instruction(self) -> str:
        task = self._state.current_task

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
        elif task in ("hard_investigate", "hard"):
            if self._hard_phase == "investigate":
                ordered_str = ", ".join(self._ordered_tests) if self._ordered_tests else "none yet"
                results_str = (
                    "\n".join(f"  {k}: {v}" for k, v in self._test_results.items())
                    if self._test_results else "  none yet"
                )
                return (
                    f"Task: Hard — Phase 1 (Investigation).\n"
                    f"Order diagnostic tests to gather evidence before deciding.\n"
                    f"Tests ordered: {ordered_str}\n"
                    f"Results so far:\n{results_str}\n"
                    f"When ready to decide, send task_type='hard_investigate' "
                    f"with ordered_investigations=[] to move to discharge phase."
                )
            else:
                results_str = (
                    "\n".join(f"  {k}: {v}" for k, v in self._test_results.items())
                    if self._test_results else "  No tests ordered."
                )
                return (
                    f"Task: Hard — Phase 2 (Discharge Decision).\n"
                    f"ALL test results available:\n{results_str}\n"
                    f"Now send task_type='hard_discharge' with:\n"
                    f"  diagnosis, disposition ('admit'/'discharge'),\n"
                    f"  prescribed_medications, follow_up_days.\n"
                    f"WARNING: Never discharge a critically ill patient (urgency 1)!"
                )
        elif task == "hard_discharge":
            return "Task: Hard — Phase 2 (Discharge Decision). Provide your final clinical plan."

        return "Unknown task."
