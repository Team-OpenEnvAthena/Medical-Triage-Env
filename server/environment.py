# server/environment.py - Medical Triage Environment
#
# Implements Options B + C + D:
#
# Option B — Sequential dependency:
#   Tests have prerequisites. You cannot order CT head until physical_exam is done.
#   Cannot order lumbar_puncture until ct_head is clear.
#   The observation shows locked_investigations so the agent knows what is blocked.
#
# Option C — Expanding information:
#   At reset(), the agent sees only chief complaint and basic vitals.
#   hidden_history is revealed when physical_exam is ordered.
#   hidden_vitals_detail is revealed when physical_exam results arrive.
#   This forces iterative hypothesis refinement as evidence accumulates.
#
# Option D — Lab turnaround delay:
#   Tests ordered at step N return at different speeds:
#     Rapid   (1 step): ECG, blood glucose, rapid strep, X-ray, physical exam
#     Standard (2 steps): CBC, troponin, CXR, electrolytes, INR, BNP, urinalysis...
#     Slow    (3 steps): CT, ultrasound, lumbar puncture, blood culture, endoscopy
#   pending_results in observation shows what is still in the lab.
#   The agent must plan actions while awaiting results — realistic emergency medicine.

import random
from uuid import uuid4
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server.interfaces import Environment
from models import TriageAction, TriageObservation, TriageState, PatientCase
from patient_cases import PATIENT_CASES


def _clamp(value: float) -> float:
    """Clamp reward to (0.01, 0.99) — evaluator rejects exact 0.0 and 1.0."""
    return round(max(0.01, min(0.99, float(value))), 4)


# Simulated test results keyed by true_diagnosis keyword
TEST_RESULT_DB: Dict[str, Dict[str, str]] = {
    "physical_exam": {
        "acute_myocardial_infarction": "Diaphoretic. Pale. JVP elevated. S3 gallop heard. Peripheral oedema.",
        "acute_pulmonary_edema": "Widespread bilateral crackles. Peripheral oedema 3+. JVP raised.",
        "perforated_appendicitis": "Rigid abdomen. Absent bowel sounds. Guarding RIF and generalised.",
        "subarachnoid_hemorrhage": "Photophobic. Neck stiffness +++. Kernig positive. GCS 13.",
        "anaphylaxis": "Widespread urticaria. Audible stridor. Lips swollen. BP unrecordable.",
        "ischemic_stroke": "Left facial droop. Right arm/leg weakness. Slurred speech. GCS 13.",
        "ectopic_pregnancy": "RIF tenderness. Cervical excitation. Adnexal mass right side.",
        "upper_gi_bleed": "Spider naevi. Ascites. Hepatic flap. Rectal exam: malaena.",
        "compartment_syndrome": "Leg woody hard. No dorsalis pedis pulse. Pain on passive stretch.",
        "appendicitis": "RIF tenderness. Rebound. Rovsing positive. No peritonism.",
        "pneumonia": "Dullness right base. Bronchial breathing. Crackles right lower lobe.",
        "pyelonephritis": "Left CVA tenderness +++. Suprapubic mild tenderness.",
        "asthma_exacerbation": "Widespread wheeze. Pulsus paradoxus 18mmHg. Accessory muscles.",
        "migraine": "Photophobic. Neck mildly stiff. No focal neurology. GCS 15.",
        "mild_traumatic_brain_injury": "GCS 14. Pupils equal reactive. No focal deficit.",
        "bradycardia": "Irregular pulse 48bpm. Orthostatic drop BP 35mmHg. No oedema.",
        "cellulitis": "Erythema tracking up leg. Lymphangitis. Fluctuance absent.",
        "pharyngitis": "Tonsillar exudate. Anterior cervical lymphadenopathy. No stridor.",
        "ankle_sprain_grade1": "Bony tenderness medial malleolus. Ottawa rules positive.",
        "uncomplicated_uti": "Mild suprapubic tenderness. CVA absent. No fever.",
        "allergic_rhinitis": "Pale boggy mucosa. Conjunctival injection. No rash.",
        "bacterial_conjunctivitis": "Mucopurulent discharge bilateral. Corneal clear. VA 6/6.",
        "mechanical_low_back_pain": "SLR negative. Saddle anaesthesia absent. Normal neurology.",
        "contact_dermatitis": "Vesicular rash bilateral forearms. Glove distribution.",
        "constipation": "Mild distension. Hypoactive bowels. Faeces palpable LIF.",
        "default": "Examination findings pending review.",
    },
    "ecg": {
        "acute_myocardial_infarction": "ST elevation leads II, III, aVF. Q waves developing. Rate 110bpm. INFERIOR STEMI.",
        "acute_pulmonary_edema": "Sinus tachycardia 125bpm. Right heart strain pattern. No ischaemia.",
        "ischemic_stroke": "Sinus rhythm 88bpm. No ischaemic changes. PR interval normal.",
        "afib_with_rvr": "ATRIAL FIBRILLATION. Ventricular rate 145bpm. No ST changes.",
        "bradycardia": "Sinus bradycardia 48bpm. PR interval prolonged at 240ms. No block.",
        "anaphylaxis": "Sinus tachycardia 130bpm. No ischaemia. QTc normal.",
        "default": "Normal sinus rhythm. Rate 78bpm. No acute ischaemia.",
    },
    "troponin": {
        "acute_myocardial_infarction": "Troponin I: 4.8 ng/mL (HIGH — ref <0.04). Rising pattern confirms MI.",
        "default": "Troponin I: 0.01 ng/mL (normal — cardiac injury excluded).",
    },
    "cbc": {
        "perforated_appendicitis": "WBC 22.4 (HIGH). Neutrophilia 89%. Hgb 13.8. Plt 290. Bandaemia present.",
        "pneumonia": "WBC 18.4 (HIGH). Neutrophilia 85%. Hgb 13.1. Plt 310.",
        "pyelonephritis": "WBC 16.2 (HIGH). Neutrophilia 82%. Hgb 11.9. Plt 180.",
        "upper_gi_bleed": "WBC 11.2. Hgb 7.8 (LOW). MCV 72 (LOW). Plt 88 (LOW). Haemorrhagic.",
        "appendicitis": "WBC 15.6 (HIGH). Neutrophilia 82%. Hgb 14.0. Plt 280.",
        "cellulitis": "WBC 14.8 (HIGH). Neutrophilia 79%. Hgb 14.2. Plt 320.",
        "severe_hypoglycemia": "WBC 9.2. Hgb 13.4. Plt 220. Normal — glucose is the issue.",
        "default": "WBC 8.9. Hgb 13.8. Plt 224. Normal differential.",
    },
    "cxr": {
        "acute_pulmonary_edema": "Bilateral pleural effusions. Cardiomegaly CTR 0.65. Pulmonary oedema. Kerley B lines.",
        "pneumonia": "RIGHT LOWER LOBE CONSOLIDATION. Air bronchograms. Silhouette sign positive.",
        "asthma_exacerbation": "Hyperinflation. Flattened diaphragms. No consolidation. No pneumothorax.",
        "default": "No acute cardiopulmonary abnormality. Heart size normal.",
    },
    "ct_head": {
        "subarachnoid_hemorrhage": "HYPERDENSE BLOOD in basal cisterns and sylvian fissure. No hydrocephalus.",
        "ischemic_stroke": "Loss of grey-white differentiation LEFT MCA territory. No haemorrhage.",
        "mild_traumatic_brain_injury": "No intracranial haemorrhage. No midline shift. No mass.",
        "migraine": "No intracranial pathology. Normal for age.",
        "default": "No acute intracranial pathology.",
    },
    "ct_abdomen": {
        "perforated_appendicitis": "PERFORATED APPENDIX. Periappendiceal abscess 4cm. Free air. Faecal peritonitis.",
        "appendicitis": "Appendix 11mm diameter. Periappendiceal fat stranding. No perforation.",
        "default": "No acute intra-abdominal pathology.",
    },
    "lumbar_puncture": {
        "subarachnoid_hemorrhage": "XANTHOCHROMIA PRESENT. RBC 32,000 (non-clearing). OP 280mmH2O. Protein elevated.",
        "default": "Clear colourless CSF. OP 140mmH2O. WBC 3. Protein 0.3. Glucose 3.8. Normal.",
    },
    "ultrasound": {
        "ectopic_pregnancy": "NO INTRAUTERINE PREGNANCY. 2.8cm adnexal mass right. Ring-of-fire sign doppler. Free fluid POD.",
        "default": "No abnormality detected.",
    },
    "bhcg": {
        "ectopic_pregnancy": "Beta-hCG: 4,200 mIU/mL (POSITIVE). Doubling time suggests ectopic.",
        "default": "Beta-hCG: NEGATIVE.",
    },
    "lactate": {
        "perforated_appendicitis": "Lactate: 4.2 mmol/L (HIGH — ref <2.0). Tissue hypoperfusion.",
        "upper_gi_bleed": "Lactate: 3.8 mmol/L (HIGH). Haemorrhagic shock.",
        "compartment_syndrome": "Lactate: 5.1 mmol/L (HIGH). Severe ischaemia.",
        "default": "Lactate: 1.1 mmol/L (normal — adequate perfusion).",
    },
    "blood_glucose": {
        "severe_hypoglycemia": "GLUCOSE: 1.6 mmol/L (CRITICALLY LOW — ref 3.9-7.8). Immediate glucose required.",
        "default": "Glucose: 5.8 mmol/L (normal).",
    },
    "electrolytes": {
        "severe_hypoglycemia": "Na 138, K 3.6, Cl 101, Cr 0.9, Glucose 1.6 (CRITICALLY LOW).",
        "bradycardia": "Na 139, K 5.8 (HIGH), Cr 1.4. Hyperkalaemia contributing.",
        "default": "Na 139, K 4.1, Cl 102, CO2 24, Cr 0.9, Glucose 5.4. All normal.",
    },
    "bnp": {
        "acute_pulmonary_edema": "BNP: 1,840 pg/mL (HIGH — ref <100). Severe heart failure.",
        "default": "BNP: 48 pg/mL (normal — heart failure unlikely).",
    },
    "inr": {
        "upper_gi_bleed": "INR 2.8 (HIGH). PT 34s. Patient on warfarin — supratherapeutic.",
        "afib_with_rvr": "INR 1.1 (subtherapeutic — warfarin compliance issue).",
        "default": "INR 1.1. PT 13s. Normal coagulation.",
    },
    "urinalysis": {
        "pyelonephritis": "Nitrites POSITIVE. WBC 3+. Bacteria 3+. RBC 2+. Protein trace.",
        "uncomplicated_uti": "Nitrites POSITIVE. WBC 2+. Bacteria 2+. No casts.",
        "default": "Normal. No significant abnormality.",
    },
    "urine_culture": {
        "pyelonephritis": "E. coli >100,000 CFU/mL. Sensitive to ciprofloxacin. Result confirmed at 48h.",
        "uncomplicated_uti": "E. coli 50,000 CFU/mL. Pan-sensitive. Result confirmed at 48h.",
        "default": "No significant growth. Result at 48h.",
    },
    "rapid_strep": {
        "pharyngitis": "RAPID STREP: POSITIVE. Group A Streptococcus detected.",
        "default": "Rapid strep: NEGATIVE. Viral cause likely.",
    },
    "xray_ankle": {
        "ankle_sprain_grade1": "No fracture. Soft tissue swelling lateral malleolus. Ottawa positive but no fracture.",
        "default": "No fracture or dislocation identified.",
    },
    "xray_leg": {
        "compartment_syndrome": "MIDSHAFT TIBIA FRACTURE. Significant soft tissue swelling. No vascular calcification.",
        "default": "No fracture identified. Soft tissue normal.",
    },
    "compartment_pressure": {
        "compartment_syndrome": "Compartment pressure: 48 mmHg (CRITICAL — ref <30). SURGICAL EMERGENCY. Fasciotomy required.",
        "default": "Compartment pressure within normal limits (<20 mmHg).",
    },
    "blood_culture": {
        "perforated_appendicitis": "Gram negative rods on initial smear. Formal ID and sensitivities pending 48-72h.",
        "pneumonia": "Streptococcus pneumoniae isolated. Sensitive to penicillin. 48h result.",
        "default": "No growth at 24h. Final result at 48-72h.",
    },
    "endoscopy": {
        "upper_gi_bleed": "OESOPHAGEAL VARICES GRADE III with active spurting. Band ligation performed. Haemostasis achieved.",
        "default": "No acute pathology. Normal mucosa throughout.",
    },
}


def get_test_result(test_name: str, patient: PatientCase) -> str:
    dx = patient.true_diagnosis.lower()
    results = TEST_RESULT_DB.get(test_name, {})
    for key, val in results.items():
        if key != "default" and key in dx:
            return val
    # Also check hidden_history for secondary diagnoses
    for h in patient.hidden_history:
        for key, val in results.items():
            if key != "default" and key in h.lower():
                return val
    return results.get("default", f"{test_name}: result within normal limits.")


class MedicalTriageEnvironment(Environment):
    """
    Medical Triage & Discharge Planning — Full Clinical Simulation

    Three tasks:
      easy             — single step triage (urgency assignment)
      medium           — multi-step investigation (Options B+C+D applied)
      hard             — multi-step investigation then discharge (Options B+C+D applied)

    Options B, C, D are active on medium and hard tasks.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = {"easy": 3, "medium": 8, "hard": 12}
    MAX_TESTS_PER_STEP = 2   # Option A: batch size limit

    ALWAYS_AVAILABLE = [
        "ecg", "blood_glucose", "rapid_strep", "physical_exam",
        "xray_ankle", "xray_leg",
    ]

    def __init__(self):
        super().__init__()
        self._state = TriageState(episode_id=str(uuid4()), step_count=0, current_task="easy")
        self._patient: Optional[PatientCase] = None
        self._ordered_tests: List[str] = []
        self._arrived_results: Dict[str, str] = {}        # test → result string (arrived)
        self._pending: Dict[str, int] = {}               # test → steps_until_arrival
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._safety_flags: List[str] = []
        self._hard_phase: str = "investigate"
        self._revealed_history: List[str] = []           # Option C: accumulated history
        self._revealed_vitals_detail: Dict[str, str] = {}

    # ── OpenEnv API ────────────────────────────────────────────────────────

    @property
    def state(self) -> TriageState:
        return self._state

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None, **kwargs) -> TriageObservation:
        if seed is not None:
            random.seed(seed)

        chosen = task if task in ("easy", "medium", "hard") else "easy"

        self._patient = random.choice(PATIENT_CASES)
        self._ordered_tests = []
        self._arrived_results = {}
        self._pending = {}
        self._cumulative_reward = 0.0
        self._done = False
        self._safety_flags = []
        self._hard_phase = "investigate"
        self._revealed_history = list(self._patient.history)   # base history shown
        self._revealed_vitals_detail = {}

        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task=chosen,
        )
        return self._make_observation(reward=None, done=False)

    def step(self, action: TriageAction) -> TriageObservation:
        self._state.step_count += 1
        self._state.current_task = action.task_type

        # Option D: age all pending results by 1 step, reveal those that are ready
        self._tick_pending()

        if action.task_type == "easy":
            reward, done = self._handle_easy(action)
        elif action.task_type == "medium":
            reward, done = self._handle_medium(action)
        elif action.task_type == "hard_investigate":
            reward, done = self._handle_hard_investigate(action)
        elif action.task_type == "hard_discharge":
            reward, done = self._handle_hard_discharge(action)
        else:
            reward, done = _clamp(0.01), True

        self._cumulative_reward += reward
        self._done = done
        return self._make_observation(reward=reward, done=done)

    # ── Option D: Pending result ticker ───────────────────────────────────

    def _tick_pending(self):
        """Reduce all pending delays by 1. Arrive results that reach 0."""
        arrived = []
        for test, steps_left in self._pending.items():
            new_delay = steps_left - 1
            if new_delay <= 0:
                arrived.append(test)
            else:
                self._pending[test] = new_delay
        for test in arrived:
            del self._pending[test]
            self._arrived_results[test] = get_test_result(test, self._patient)
            # Option C: if physical_exam arrives, reveal hidden history + vitals
            if test == "physical_exam":
                self._revealed_history = (
                    list(self._patient.history) + list(self._patient.hidden_history)
                )
                self._revealed_vitals_detail.update(self._patient.hidden_vitals_detail)

    # ── Option B: Prerequisite checking ───────────────────────────────────

    def _available_tests(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Returns (available_list, locked_dict).
        A test is available if all its prerequisites have arrived results.
        A test already ordered is excluded.
        """
        prereqs = self._patient.test_prerequisites
        already_done = set(self._ordered_tests)

        available = []
        locked = {}

        all_tests = list(self._patient.test_turnaround.keys())
        # Add always-available tests
        for t in self.ALWAYS_AVAILABLE:
            if t not in all_tests:
                all_tests.append(t)

        for test in all_tests:
            if test in already_done:
                continue
            required_prereqs = prereqs.get(test, [])
            missing = [p for p in required_prereqs if p not in self._arrived_results]
            if missing:
                locked[test] = missing
            else:
                available.append(test)

        return available, locked

    # ── Option D: Queuing a test ──────────────────────────────────────────

    def _queue_test(self, test: str):
        """Add test to pending queue with its turnaround delay."""
        if test not in self._ordered_tests:
            self._ordered_tests.append(test)
            delay = self._patient.test_turnaround.get(test, 2)
            self._pending[test] = delay

    # ── Easy ───────────────────────────────────────────────────────────────

    def _handle_easy(self, action: TriageAction) -> Tuple[float, bool]:
        if action.urgency_assignment is None:
            return _clamp(0.01), True
        correct  = self._patient.true_urgency
        assigned = action.urgency_assignment
        if assigned == correct:
            reward = 0.99
        elif abs(assigned - correct) == 1:
            reward = 0.5
        else:
            reward = 0.01
        return _clamp(reward), True

    # ── Medium ─────────────────────────────────────────────────────────────

    def _handle_medium(self, action: TriageAction) -> Tuple[float, bool]:
        if action.ordered_investigations is None:
            return _clamp(0.01), True

        # [] = agent signals done → compute final F1
        if len(action.ordered_investigations) == 0:
            return _clamp(self._score_investigations()), True

        available, _ = self._available_tests()

        # Option A: max 2 tests per step
        requested = action.ordered_investigations[:self.MAX_TESTS_PER_STEP]

        queued_count = 0
        blocked_count = 0
        for test in requested:
            if test in available:
                self._queue_test(test)
                queued_count += 1
            else:
                blocked_count += 1  # prerequisite not met or already ordered

        # Partial reward: proportion of required tests now ordered (pending or arrived)
        required = set(self._patient.required_investigations)
        ordered_set = set(self._ordered_tests)
        covered = required & ordered_set
        partial = len(covered) / max(len(required), 1)

        waste = len([t for t in ordered_set if t not in required])
        penalty = waste * 0.05

        # Small penalty for trying to order blocked tests
        if blocked_count > 0:
            penalty += blocked_count * 0.02

        step_reward = _clamp(partial - penalty)
        return step_reward, False   # only [] ends episode

    def _score_investigations(self) -> float:
        required = set(self._patient.required_investigations)
        ordered  = set(self._ordered_tests)
        if not ordered:
            return 0.01
        tp = len(required & ordered)
        precision = tp / len(ordered)
        recall    = tp / max(len(required), 1)
        if precision + recall == 0:
            return 0.01
        f1 = 2 * precision * recall / (precision + recall)
        waste_penalty = len(ordered - required) * 0.1
        return _clamp(f1 - waste_penalty)

    # ── Hard: Phase 1 — investigate ────────────────────────────────────────

    def _handle_hard_investigate(self, action: TriageAction) -> Tuple[float, bool]:
        if action.ordered_investigations is None:
            self._hard_phase = "discharge"
            return _clamp(0.01), False

        # [] = transition to discharge phase
        if len(action.ordered_investigations) == 0:
            self._hard_phase = "discharge"
            inv_score = self._score_investigations()
            return _clamp(inv_score * 0.3), False   # 0.3 weight; not done — need discharge

        available, _ = self._available_tests()
        requested = action.ordered_investigations[:self.MAX_TESTS_PER_STEP]

        blocked_count = 0
        for test in requested:
            if test in available:
                self._queue_test(test)
            else:
                blocked_count += 1

        required = set(self._patient.required_investigations)
        ordered_set = set(self._ordered_tests)
        covered = required & ordered_set
        partial = len(covered) / max(len(required), 1)
        waste = len([t for t in ordered_set if t not in required])
        penalty = waste * 0.05 + blocked_count * 0.02

        return _clamp(partial * 0.3 - penalty), False

    # ── Hard: Phase 2 — discharge ──────────────────────────────────────────

    def _handle_hard_discharge(self, action: TriageAction) -> Tuple[float, bool]:
        discharge_score = 0.0

        # 1. Diagnosis (0.3 of discharge weight)
        if action.diagnosis:
            agent_dx = action.diagnosis.lower().replace("_", " ").replace("-", " ")
            true_dx  = self._patient.true_diagnosis.lower().replace("_", " ")
            keywords = [w for w in true_dx.split() if len(w) > 3]
            if keywords:
                hits = sum(1 for kw in keywords if kw in agent_dx)
                discharge_score += 0.3 * (hits / len(keywords))
            elif agent_dx == true_dx:
                discharge_score += 0.3

        # 2. Disposition (0.3)
        if action.disposition:
            if action.disposition.lower() == self._patient.correct_disposition.lower():
                discharge_score += 0.3

        # 3. Medications (0.2) — partial credit
        if action.prescribed_medications:
            safe = set(m.lower() for m in self._patient.safe_medications)
            prescribed = set(m.lower().replace("_", " ").replace("-", " ")
                             for m in action.prescribed_medications)
            prescribed_raw = set(m.lower() for m in action.prescribed_medications)
            all_p = prescribed | prescribed_raw
            hits = sum(1 for s in safe if any(s in p or p in s for p in all_p))
            if safe:
                discharge_score += 0.2 * min(1.0, hits / len(safe))

        # 4. Safety penalty
        if action.disposition == "discharge" and self._patient.true_urgency == 1:
            discharge_score -= 0.5
            self._safety_flags.append(
                f"UNSAFE: discharged urgency-1 patient {self._patient.patient_id}"
            )

        # 5. Follow-up (0.2)
        if action.follow_up_days is not None:
            if action.disposition == "discharge" and 0 < action.follow_up_days <= 14:
                discharge_score += 0.2
            elif action.disposition == "admit":
                discharge_score += 0.2

        return _clamp(discharge_score * 0.7), True

    # ── Observation builder ────────────────────────────────────────────────

    def _make_observation(self, reward, done) -> TriageObservation:
        available, locked = self._available_tests()

        patient_dict = None
        if self._patient:
            patient_dict = {
                # Always visible
                "id":               self._patient.patient_id,
                "age":              self._patient.age,
                "sex":              self._patient.sex,
                "chief_complaint":  self._patient.chief_complaint,
                # Basic vitals always visible
                "heart_rate":       self._patient.vitals.heart_rate,
                "blood_pressure":   self._patient.vitals.blood_pressure,
                "spo2":             self._patient.vitals.spo2,
                "temperature":      self._patient.vitals.temperature,
                "respiratory_rate": self._patient.vitals.respiratory_rate,
                # Option C: history expands as physical_exam arrives
                "past_medical_history": self._revealed_history,
                "additional_findings":  self._revealed_vitals_detail,
                "allergies":        self._patient.allergies,
                # Progress tracking
                "ordered_tests":    list(self._ordered_tests),
                "hard_phase":       self._hard_phase,
            }

        return TriageObservation(
            done=done if done is not None else False,
            reward=reward,
            current_patient=patient_dict,
            available_investigations=available,          # Option B: only unlocked tests
            locked_investigations=locked,               # Option B: shows what is blocked
            investigation_results=dict(self._arrived_results) if self._arrived_results else None,
            pending_results=dict(self._pending),        # Option D: steps until each arrives
            task_instruction=self._get_instruction(),
            partial_score=_clamp(self._cumulative_reward) if self._cumulative_reward else 0.01,
            safety_flags=list(self._safety_flags),
        )

    def _get_instruction(self) -> str:
        task = self._state.current_task

        if task == "easy":
            return (
                "TASK: Triage Prioritization.\n"
                "Set urgency_assignment:\n"
                "  1 = Immediate (life-threatening, resuscitation needed now)\n"
                "  2 = Urgent (serious, must be seen within 30 min)\n"
                "  3 = Non-urgent (stable, can wait)"
            )

        elif task == "medium":
            pending_str = ", ".join(
                f"{t}({s}step{'s' if s>1 else ''} remaining)"
                for t, s in self._pending.items()
            ) or "none"
            arrived_str = ", ".join(self._arrived_results.keys()) or "none"
            locked_str  = ", ".join(
                f"{t}(needs: {', '.join(prereqs)})"
                for t, prereqs in self._get_locked_display().items()
            ) or "none"

            return (
                "TASK: Investigation Ordering (multi-step, B+C+D mechanics active).\n\n"
                f"Results arrived: {arrived_str}\n"
                f"Results pending: {pending_str}\n"
                f"Locked tests (prerequisites needed): {locked_str}\n\n"
                "Rules:\n"
                "  - Max 2 tests per step\n"
                "  - Some tests require prerequisites (Option B)\n"
                "  - Results arrive with realistic delay: rapid=1 step, standard=2, slow=3 (Option D)\n"
                "  - Order physical_exam to reveal hidden history and examination findings (Option C)\n"
                "  - Send ordered_investigations=[] when you have enough evidence to decide"
            )

        elif task in ("hard_investigate", "hard"):
            pending_str = ", ".join(
                f"{t}({s}step{'s' if s>1 else ''} remaining)"
                for t, s in self._pending.items()
            ) or "none"
            arrived_str = (
                "\n".join(f"  {k}: {v}" for k, v in self._arrived_results.items())
                or "  none yet"
            )
            locked_str = ", ".join(
                f"{t}(needs: {', '.join(prereqs)})"
                for t, prereqs in self._get_locked_display().items()
            ) or "none"

            if self._hard_phase == "investigate":
                return (
                    "TASK: Hard — Phase 1: Investigation\n\n"
                    f"Test results so far:\n{arrived_str}\n\n"
                    f"Pending (steps remaining): {pending_str}\n"
                    f"Locked (prerequisites missing): {locked_str}\n\n"
                    "Rules:\n"
                    "  - Max 2 tests per step (Option A)\n"
                    "  - Prerequisites must be met before ordering certain tests (Option B)\n"
                    "  - Order physical_exam to reveal hidden clinical findings (Option C)\n"
                    "  - Results arrive with realistic delay (Option D)\n"
                    "  - Plan your next actions while waiting for pending results\n\n"
                    "When you have enough evidence: send task_type='hard_investigate' with ordered_investigations=[]"
                )
            else:
                return (
                    "TASK: Hard — Phase 2: Discharge Decision\n\n"
                    f"ALL AVAILABLE RESULTS:\n{arrived_str}\n\n"
                    f"Still pending (won't arrive): {pending_str}\n\n"
                    "Now send task_type='hard_discharge' with:\n"
                    "  diagnosis, disposition ('admit'/'discharge'),\n"
                    "  prescribed_medications, follow_up_days\n\n"
                    "⚠️  SAFETY: Never discharge SpO2<90% or BP<90/60 or urgency=1 patient."
                )

        elif task == "hard_discharge":
            return "TASK: Hard — Phase 2: Discharge Decision. Provide your final clinical plan."

        return "Unknown task."

    def _get_locked_display(self) -> Dict[str, List[str]]:
        """Return locked tests with their missing prerequisites."""
        _, locked = self._available_tests()
        return locked
