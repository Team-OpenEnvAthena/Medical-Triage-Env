# patient_cases.py - Synthetic Patient Cases with Ground Truth
#
# Each case now includes:
#   Option B: test_prerequisites — some tests require others first
#   Option C: hidden_history / hidden_vitals_detail — revealed progressively
#   Option D: test_turnaround — realistic lab delay in steps
#             1=rapid (point-of-care, ECG reader)
#             2=standard (most labs, 20-40 min)
#             3=slow (cultures, complex imaging)
#
# Turnaround categories:
#   Rapid  (1 step): ecg, blood_glucose, rapid_strep, physical_exam, xray_*
#   Standard (2 steps): troponin, cbc, cxr, electrolytes, inr, bnp, urinalysis,
#                        bhcg, lactate, urine_culture, compartment_pressure
#   Slow   (3 steps): ct_head, ct_abdomen, ultrasound, lumbar_puncture,
#                      blood_culture, endoscopy

from models import PatientCase, Vitals

# Default turnaround map reused across cases
RAPID    = 1
STANDARD = 2
SLOW     = 3

TURNAROUND = {
    "ecg": RAPID, "blood_glucose": RAPID, "rapid_strep": RAPID,
    "physical_exam": RAPID, "xray_ankle": RAPID, "xray_leg": RAPID,
    "troponin": STANDARD, "cbc": STANDARD, "cxr": STANDARD,
    "electrolytes": STANDARD, "inr": STANDARD, "bnp": STANDARD,
    "urinalysis": STANDARD, "bhcg": STANDARD, "lactate": STANDARD,
    "urine_culture": STANDARD, "compartment_pressure": STANDARD,
    "ct_head": SLOW, "ct_abdomen": SLOW, "ultrasound": SLOW,
    "lumbar_puncture": SLOW, "blood_culture": SLOW, "endoscopy": SLOW,
}


def turnaround(tests):
    return {t: TURNAROUND.get(t, STANDARD) for t in tests}


PATIENT_CASES = [

    # ── IMMEDIATE (Urgency 1) ─────────────────────────────────────────────

    PatientCase(
        patient_id="P001",
        age=54, sex="M",
        chief_complaint="Chest pain radiating to left arm, sweating",
        vitals=Vitals(heart_rate=110, blood_pressure="90/60",
                      spo2=94, temperature=37.8, respiratory_rate=22),
        history=["hypertension"],
        current_medications=["lisinopril"],
        allergies=["penicillin"],
        # Option C: diabetes history hidden until physical_exam done
        hidden_history=["type_2_diabetes"],
        hidden_vitals_detail={"glucose": "18.2 mmol/L (HIGH)"},
        true_diagnosis="acute_myocardial_infarction",
        true_urgency=1,
        # Option B: troponin can only be ordered after ECG (need ECG to confirm cardiac)
        #           cbc can only be ordered after troponin (confirm cardiac workup first)
        test_prerequisites={
            "troponin": ["ecg"],
            "cbc": ["ecg"],
            "cxr": ["ecg"],
        },
        required_investigations=["ecg", "troponin", "cbc"],
        test_turnaround=turnaround(["ecg", "troponin", "cbc", "cxr", "electrolytes"]),
        correct_disposition="admit",
        safe_medications=["aspirin", "nitroglycerin", "morphine"],
    ),

    PatientCase(
        patient_id="P002",
        age=67, sex="F",
        chief_complaint="Severe shortness of breath, cannot complete sentences",
        vitals=Vitals(heart_rate=125, blood_pressure="180/110",
                      spo2=88, temperature=36.9, respiratory_rate=32),
        history=["heart_failure"],
        current_medications=["furosemide"],
        allergies=[],
        hidden_history=["copd", "atrial_fibrillation"],
        hidden_vitals_detail={"peak_flow": "180 L/min (severely reduced)"},
        true_diagnosis="acute_pulmonary_edema",
        true_urgency=1,
        test_prerequisites={
            "bnp": ["cxr"],       # confirm pulmonary picture before BNP
            "ecg": [],
            "cxr": [],
        },
        required_investigations=["cxr", "ecg", "bnp"],
        test_turnaround=turnaround(["cxr", "ecg", "bnp", "electrolytes", "cbc"]),
        correct_disposition="admit",
        safe_medications=["furosemide", "oxygen", "nitroglycerin"],
    ),

    PatientCase(
        patient_id="P003",
        age=32, sex="M",
        chief_complaint="Stabbing abdominal pain, rigid abdomen, fever",
        vitals=Vitals(heart_rate=115, blood_pressure="85/55",
                      spo2=96, temperature=38.9, respiratory_rate=24),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["previous_appendicitis_scar"],
        hidden_vitals_detail={"bowel_sounds": "absent"},
        true_diagnosis="perforated_appendicitis",
        true_urgency=1,
        test_prerequisites={
            "ct_abdomen": ["cbc", "lactate"],  # need labs before CT
            "blood_culture": ["cbc"],
        },
        required_investigations=["cbc", "lactate", "ct_abdomen"],
        test_turnaround=turnaround(["cbc", "lactate", "ct_abdomen", "blood_culture"]),
        correct_disposition="admit",
        safe_medications=["morphine", "ceftriaxone", "metronidazole"],
    ),

    PatientCase(
        patient_id="P004",
        age=45, sex="F",
        chief_complaint="Sudden severe headache, worst of life, neck stiffness",
        vitals=Vitals(heart_rate=95, blood_pressure="165/95",
                      spo2=98, temperature=37.2, respiratory_rate=18),
        history=["migraine"],
        current_medications=["sumatriptan"],
        allergies=[],
        hidden_history=["family_history_of_aneurysm"],
        hidden_vitals_detail={"photophobia": "severe", "kernig_sign": "positive"},
        true_diagnosis="subarachnoid_hemorrhage",
        true_urgency=1,
        test_prerequisites={
            "lumbar_puncture": ["ct_head"],  # must rule out raised ICP before LP
        },
        required_investigations=["ct_head", "lumbar_puncture"],
        test_turnaround=turnaround(["ct_head", "lumbar_puncture", "cbc", "electrolytes"]),
        correct_disposition="admit",
        safe_medications=["nimodipine", "pain_control"],
    ),

    PatientCase(
        patient_id="P005",
        age=28, sex="M",
        chief_complaint="Difficulty breathing after bee sting, facial swelling",
        vitals=Vitals(heart_rate=130, blood_pressure="80/50",
                      spo2=90, temperature=37.0, respiratory_rate=28),
        history=[],
        current_medications=[],
        allergies=["bee_venom"],
        hidden_history=["previous_anaphylaxis_episode"],
        hidden_vitals_detail={"urticaria": "widespread", "stridor": "audible"},
        true_diagnosis="anaphylaxis",
        true_urgency=1,
        test_prerequisites={},
        required_investigations=["ecg", "cbc", "electrolytes"],
        test_turnaround=turnaround(["ecg", "cbc", "electrolytes"]),
        correct_disposition="admit",
        safe_medications=["epinephrine", "diphenhydramine", "methylprednisolone"],
    ),

    PatientCase(
        patient_id="P006",
        age=71, sex="M",
        chief_complaint="Sudden right-sided weakness and slurred speech",
        vitals=Vitals(heart_rate=88, blood_pressure="155/90",
                      spo2=97, temperature=36.8, respiratory_rate=16),
        history=["atrial_fibrillation"],
        current_medications=["metoprolol"],
        allergies=[],
        hidden_history=["hypertension", "previous_TIA"],
        hidden_vitals_detail={"gcs": "13", "facial_droop": "left", "onset_time": "90 minutes ago"},
        true_diagnosis="ischemic_stroke",
        true_urgency=1,
        test_prerequisites={
            "inr": ["cbc"],       # coag after CBC
        },
        required_investigations=["ct_head", "cbc", "inr"],
        test_turnaround=turnaround(["ct_head", "cbc", "inr", "electrolytes"]),
        correct_disposition="admit",
        safe_medications=["aspirin", "tpa_if_eligible"],
    ),

    PatientCase(
        patient_id="P007",
        age=19, sex="F",
        chief_complaint="Vaginal bleeding, 8 weeks pregnant, severe abdominal pain",
        vitals=Vitals(heart_rate=118, blood_pressure="88/60",
                      spo2=98, temperature=37.3, respiratory_rate=20),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["previous_PID", "IUD_in_situ"],
        hidden_vitals_detail={"cervical_os": "closed", "adnexal_tenderness": "right"},
        true_diagnosis="ectopic_pregnancy",
        true_urgency=1,
        test_prerequisites={
            "ultrasound": ["bhcg"],   # confirm pregnancy first then image
        },
        required_investigations=["bhcg", "cbc", "ultrasound"],
        test_turnaround=turnaround(["bhcg", "cbc", "ultrasound"]),
        correct_disposition="admit",
        safe_medications=["pain_control", "possible_surgery"],
    ),

    PatientCase(
        patient_id="P008",
        age=55, sex="M",
        chief_complaint="Vomiting blood, black tarry stools since yesterday",
        vitals=Vitals(heart_rate=120, blood_pressure="85/50",
                      spo2=96, temperature=36.5, respiratory_rate=22),
        history=["cirrhosis"],
        current_medications=[],
        allergies=[],
        hidden_history=["alcohol_use_disorder", "oesophageal_varices_known"],
        hidden_vitals_detail={"spider_naevi": "present", "ascites": "moderate"},
        true_diagnosis="upper_gi_bleed",
        true_urgency=1,
        test_prerequisites={
            "endoscopy": ["cbc", "inr"],   # stabilise and get labs before scoping
        },
        required_investigations=["cbc", "inr", "endoscopy"],
        test_turnaround=turnaround(["cbc", "inr", "endoscopy", "lactate"]),
        correct_disposition="admit",
        safe_medications=["proton_pump_inhibitor", "octreotide"],
    ),

    PatientCase(
        patient_id="P009",
        age=62, sex="F",
        chief_complaint="Found unconscious by family, diabetic",
        vitals=Vitals(heart_rate=75, blood_pressure="110/70",
                      spo2=99, temperature=36.2, respiratory_rate=14),
        history=["type_1_diabetes"],
        current_medications=["insulin"],
        allergies=[],
        hidden_history=["recent_insulin_dose_doubled"],
        hidden_vitals_detail={"gcs": "8", "diaphoresis": "profuse"},
        true_diagnosis="severe_hypoglycemia",
        true_urgency=1,
        test_prerequisites={
            "cbc": ["blood_glucose"],       # confirm glucose first
            "electrolytes": ["blood_glucose"],
        },
        required_investigations=["blood_glucose", "cbc", "electrolytes"],
        test_turnaround=turnaround(["blood_glucose", "cbc", "electrolytes"]),
        correct_disposition="admit",
        safe_medications=["dextrose", "glucagon"],
    ),

    PatientCase(
        patient_id="P010",
        age=38, sex="M",
        chief_complaint="Crush injury to leg, pulseless foot after industrial accident",
        vitals=Vitals(heart_rate=110, blood_pressure="95/60",
                      spo2=97, temperature=37.0, respiratory_rate=24),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["3_hour_entrapment"],
        hidden_vitals_detail={"compartment_tension": "woody hard", "capillary_refill": ">3 seconds"},
        true_diagnosis="compartment_syndrome",
        true_urgency=1,
        test_prerequisites={
            "compartment_pressure": ["xray_leg"],   # image first then measure pressure
        },
        required_investigations=["xray_leg", "compartment_pressure", "cbc"],
        test_turnaround=turnaround(["xray_leg", "compartment_pressure", "cbc", "lactate"]),
        correct_disposition="admit",
        safe_medications=["morphine", "immediate_surgery"],
    ),

    # ── URGENT (Urgency 2) ────────────────────────────────────────────────

    PatientCase(
        patient_id="P011",
        age=42, sex="F",
        chief_complaint="Right lower quadrant pain for 12 hours, fever",
        vitals=Vitals(heart_rate=98, blood_pressure="125/80",
                      spo2=98, temperature=38.5, respiratory_rate=18),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["previous_ovarian_cyst"],
        hidden_vitals_detail={"rebound_tenderness": "positive RLQ", "Rovsing_sign": "positive"},
        true_diagnosis="appendicitis",
        true_urgency=2,
        test_prerequisites={
            "ct_abdomen": ["cbc", "urinalysis"],
        },
        required_investigations=["cbc", "urinalysis", "ct_abdomen"],
        test_turnaround=turnaround(["cbc", "urinalysis", "ct_abdomen"]),
        correct_disposition="admit",
        safe_medications=["morphine", "ceftriaxone"],
    ),

    PatientCase(
        patient_id="P012",
        age=35, sex="M",
        chief_complaint="Fever 39.5°C, productive cough, chills for 3 days",
        vitals=Vitals(heart_rate=105, blood_pressure="118/75",
                      spo2=92, temperature=39.5, respiratory_rate=22),
        history=["asthma"],
        current_medications=["albuterol"],
        allergies=[],
        hidden_history=["recent_overseas_travel"],
        hidden_vitals_detail={"dullness_to_percussion": "right base", "crackles": "right lower lobe"},
        true_diagnosis="pneumonia",
        true_urgency=2,
        test_prerequisites={
            "blood_culture": ["cbc"],
        },
        required_investigations=["cxr", "cbc", "blood_culture"],
        test_turnaround=turnaround(["cxr", "cbc", "blood_culture", "electrolytes"]),
        correct_disposition="admit",
        safe_medications=["ceftriaxone", "azithromycin", "oxygen"],
    ),

    PatientCase(
        patient_id="P013",
        age=58, sex="F",
        chief_complaint="Urinary frequency, burning, flank pain and rigors",
        vitals=Vitals(heart_rate=92, blood_pressure="130/85",
                      spo2=98, temperature=38.8, respiratory_rate=16),
        history=["recurrent_uti"],
        current_medications=[],
        allergies=["sulfa"],
        hidden_history=["immunosuppressed_post_transplant"],
        hidden_vitals_detail={"CVA_tenderness": "left positive"},
        true_diagnosis="pyelonephritis",
        true_urgency=2,
        test_prerequisites={
            "urine_culture": ["urinalysis"],
            "blood_culture": ["cbc"],
        },
        required_investigations=["urinalysis", "urine_culture", "cbc"],
        test_turnaround=turnaround(["urinalysis", "urine_culture", "cbc", "blood_culture"]),
        correct_disposition="admit",
        safe_medications=["ciprofloxacin", "pain_control"],
    ),

    PatientCase(
        patient_id="P014",
        age=27, sex="M",
        chief_complaint="Deep laceration on forearm from glass, actively bleeding",
        vitals=Vitals(heart_rate=85, blood_pressure="120/78",
                      spo2=99, temperature=37.0, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["blood_thinner_use_not_disclosed"],
        hidden_vitals_detail={"tendon_integrity": "intact", "sensation": "intact"},
        true_diagnosis="laceration_requiring_sutures",
        true_urgency=2,
        test_prerequisites={
            "inr": ["cbc"],
        },
        required_investigations=["cbc", "inr"],
        test_turnaround=turnaround(["cbc", "inr"]),
        correct_disposition="discharge",
        safe_medications=["local_anesthetic", "tetanus", "antibiotics_if_dirty"],
    ),

    PatientCase(
        patient_id="P015",
        age=65, sex="F",
        chief_complaint="Palpitations, rapid heart rate, feeling faint",
        vitals=Vitals(heart_rate=145, blood_pressure="135/85",
                      spo2=96, temperature=37.1, respiratory_rate=18),
        history=["atrial_fibrillation"],
        current_medications=["warfarin"],
        allergies=[],
        hidden_history=["recent_dose_increase_warfarin"],
        hidden_vitals_detail={"irregular_rhythm": "confirmed", "JVP": "raised"},
        true_diagnosis="afib_with_rvr",
        true_urgency=2,
        test_prerequisites={
            "inr": ["ecg"],
            "electrolytes": ["ecg"],
        },
        required_investigations=["ecg", "electrolytes", "inr"],
        test_turnaround=turnaround(["ecg", "electrolytes", "inr"]),
        correct_disposition="admit",
        safe_medications=["diltiazem", "metoprolol"],
    ),

    PatientCase(
        patient_id="P016",
        age=50, sex="M",
        chief_complaint="Fall from ladder, hit head, brief loss of consciousness",
        vitals=Vitals(heart_rate=78, blood_pressure="140/88",
                      spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["on_aspirin_daily"],
        hidden_vitals_detail={"gcs": "14", "pupils": "equal reactive", "amnesia": "2 minutes"},
        true_diagnosis="mild_traumatic_brain_injury",
        true_urgency=2,
        test_prerequisites={
            "ct_head": ["physical_exam"],   # exam first to decide if CT needed
        },
        required_investigations=["physical_exam", "ct_head", "cbc"],
        test_turnaround=turnaround(["physical_exam", "ct_head", "cbc"]),
        correct_disposition="discharge",
        safe_medications=["acetaminophen", "concussion_precautions"],
    ),

    PatientCase(
        patient_id="P017",
        age=29, sex="F",
        chief_complaint="Asthma exacerbation, wheezing, not responding to inhaler",
        vitals=Vitals(heart_rate=102, blood_pressure="118/72",
                      spo2=91, temperature=37.2, respiratory_rate=26),
        history=["asthma"],
        current_medications=["albuterol", "fluticasone"],
        allergies=[],
        hidden_history=["3_previous_ICU_admissions_for_asthma"],
        hidden_vitals_detail={"pulsus_paradoxus": "18mmHg", "accessory_muscles": "in_use"},
        true_diagnosis="asthma_exacerbation",
        true_urgency=2,
        test_prerequisites={
            "cxr": ["physical_exam"],
            "electrolytes": ["cbc"],
        },
        required_investigations=["physical_exam", "cxr", "cbc", "electrolytes"],
        test_turnaround=turnaround(["physical_exam", "cxr", "cbc", "electrolytes"]),
        correct_disposition="admit",
        safe_medications=["albuterol_nebs", "ipratropium", "methylprednisolone", "oxygen"],
    ),

    PatientCase(
        patient_id="P018",
        age=44, sex="M",
        chief_complaint="Severe headache, photophobia, vomiting, known migraineur",
        vitals=Vitals(heart_rate=88, blood_pressure="145/90",
                      spo2=99, temperature=37.0, respiratory_rate=14),
        history=["migraine"],
        current_medications=["sumatriptan"],
        allergies=[],
        hidden_history=["this_headache_different_from_usual"],
        hidden_vitals_detail={"neck_stiffness": "mild", "kernig_sign": "negative"},
        true_diagnosis="migraine",
        true_urgency=2,
        test_prerequisites={
            "ct_head": ["physical_exam"],
            "electrolytes": ["cbc"],
        },
        required_investigations=["physical_exam", "ct_head", "cbc", "electrolytes"],
        test_turnaround=turnaround(["physical_exam", "ct_head", "cbc", "electrolytes"]),
        correct_disposition="discharge",
        safe_medications=["sumatriptan", "metoclopramide", "ketorolac"],
    ),

    PatientCase(
        patient_id="P019",
        age=72, sex="F",
        chief_complaint="Dizziness and near-syncope on standing, on beta-blockers",
        vitals=Vitals(heart_rate=48, blood_pressure="100/65",
                      spo2=98, temperature=36.8, respiratory_rate=14),
        history=["hypertension", "atrial_fibrillation"],
        current_medications=["metoprolol", "warfarin"],
        allergies=[],
        hidden_history=["recent_metoprolol_dose_doubled_by_GP"],
        hidden_vitals_detail={"orthostatic_drop": "BP 85/55 on standing"},
        true_diagnosis="bradycardia",
        true_urgency=2,
        test_prerequisites={
            "electrolytes": ["ecg"],
            "inr": ["ecg"],
        },
        required_investigations=["ecg", "electrolytes", "inr"],
        test_turnaround=turnaround(["ecg", "electrolytes", "inr"]),
        correct_disposition="admit",
        safe_medications=["hold_beta_blocker", "possible_pacing"],
    ),

    PatientCase(
        patient_id="P020",
        age=33, sex="M",
        chief_complaint="Redness, swelling and warmth spreading up left leg since yesterday",
        vitals=Vitals(heart_rate=92, blood_pressure="128/82",
                      spo2=99, temperature=38.2, respiratory_rate=16),
        history=["diabetes"],
        current_medications=["metformin"],
        allergies=[],
        hidden_history=["poor_glycaemic_control_HbA1c_10"],
        hidden_vitals_detail={"fluctuance": "absent", "lymphangitis": "present"},
        true_diagnosis="cellulitis",
        true_urgency=2,
        test_prerequisites={
            "blood_culture": ["cbc"],
        },
        required_investigations=["cbc", "blood_culture"],
        test_turnaround=turnaround(["cbc", "blood_culture", "electrolytes"]),
        correct_disposition="admit",
        safe_medications=["cephalexin", "elevation", "follow_up_48h"],
    ),

    # ── NON-URGENT (Urgency 3) ────────────────────────────────────────────

    PatientCase(
        patient_id="P021",
        age=25, sex="F",
        chief_complaint="Sore throat, mild fever, difficulty swallowing",
        vitals=Vitals(heart_rate=78, blood_pressure="118/75",
                      spo2=99, temperature=37.8, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["contact_with_strep_case_at_work"],
        hidden_vitals_detail={"tonsillar_exudate": "present", "anterior_cervical_lymphadenopathy": "present"},
        true_diagnosis="pharyngitis",
        true_urgency=3,
        test_prerequisites={
            "cbc": ["rapid_strep"],
        },
        required_investigations=["rapid_strep", "cbc"],
        test_turnaround=turnaround(["rapid_strep", "cbc"]),
        correct_disposition="discharge",
        safe_medications=["acetaminophen", "throat_lozenges", "antibiotics_if_strep"],
    ),

    PatientCase(
        patient_id="P022",
        age=40, sex="M",
        chief_complaint="Ankle twisted playing football, painful to walk",
        vitals=Vitals(heart_rate=72, blood_pressure="125/80",
                      spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["previous_ankle_fracture_same_side"],
        hidden_vitals_detail={"Ottawa_rules": "positive — bony tenderness at medial malleolus"},
        true_diagnosis="ankle_sprain_grade1",
        true_urgency=3,
        test_prerequisites={
            "xray_ankle": ["physical_exam"],
        },
        required_investigations=["physical_exam", "xray_ankle", "cbc"],
        test_turnaround=turnaround(["physical_exam", "xray_ankle", "cbc"]),
        correct_disposition="discharge",
        safe_medications=["ibuprofen", "rice_protocol"],
    ),

    PatientCase(
        patient_id="P023",
        age=31, sex="F",
        chief_complaint="Burning urination for 2 days, no fever",
        vitals=Vitals(heart_rate=75, blood_pressure="120/78",
                      spo2=99, temperature=37.1, respiratory_rate=14),
        history=["recurrent_uti"],
        current_medications=[],
        allergies=[],
        hidden_history=["sexually_active_new_partner"],
        hidden_vitals_detail={"suprapubic_tenderness": "mild", "CVA_tenderness": "absent"},
        true_diagnosis="uncomplicated_uti",
        true_urgency=3,
        test_prerequisites={
            "urine_culture": ["urinalysis"],
        },
        required_investigations=["urinalysis", "urine_culture"],
        test_turnaround=turnaround(["urinalysis", "urine_culture"]),
        correct_disposition="discharge",
        safe_medications=["nitrofurantoin", "phenazopyridine"],
    ),

    PatientCase(
        patient_id="P024",
        age=22, sex="M",
        chief_complaint="Runny nose, sneezing, itchy eyes all week",
        vitals=Vitals(heart_rate=70, blood_pressure="118/72",
                      spo2=99, temperature=36.8, respiratory_rate=14),
        history=["seasonal_allergies"],
        current_medications=[],
        allergies=[],
        hidden_history=["new_cat_at_home"],
        hidden_vitals_detail={"conjunctival_injection": "bilateral", "nasal_mucosa": "pale boggy"},
        true_diagnosis="allergic_rhinitis",
        true_urgency=3,
        test_prerequisites={
            "cbc": ["physical_exam"],
        },
        required_investigations=["physical_exam", "cbc", "urinalysis"],
        test_turnaround=turnaround(["physical_exam", "cbc", "urinalysis"]),
        correct_disposition="discharge",
        safe_medications=["cetirizine", "nasal_steroid"],
    ),

    PatientCase(
        patient_id="P025",
        age=48, sex="F",
        chief_complaint="Acute low back pain after lifting boxes, cannot straighten",
        vitals=Vitals(heart_rate=74, blood_pressure="122/80",
                      spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["osteoporosis_risk_factors"],
        hidden_vitals_detail={"straight_leg_raise": "negative bilateral", "saddle_anaesthesia": "absent"},
        true_diagnosis="mechanical_low_back_pain",
        true_urgency=3,
        test_prerequisites={
            "cbc": ["physical_exam"],
            "electrolytes": ["physical_exam"],
        },
        required_investigations=["physical_exam", "cbc", "urinalysis", "electrolytes"],
        test_turnaround=turnaround(["physical_exam", "cbc", "urinalysis", "electrolytes"]),
        correct_disposition="discharge",
        safe_medications=["ibuprofen", "muscle_relaxant", "physical_therapy"],
    ),

    PatientCase(
        patient_id="P026",
        age=35, sex="M",
        chief_complaint="Red eye, discharge, woke up with eyes stuck together",
        vitals=Vitals(heart_rate=72, blood_pressure="120/78",
                      spo2=99, temperature=37.0, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["contact_lens_wearer"],
        hidden_vitals_detail={"visual_acuity": "6/6 bilateral", "corneal_staining": "negative"},
        true_diagnosis="bacterial_conjunctivitis",
        true_urgency=3,
        test_prerequisites={
            "cbc": ["physical_exam"],
        },
        required_investigations=["physical_exam", "cbc", "urinalysis"],
        test_turnaround=turnaround(["physical_exam", "cbc", "urinalysis"]),
        correct_disposition="discharge",
        safe_medications=["erythromycin_ointment"],
    ),

    PatientCase(
        patient_id="P027",
        age=28, sex="F",
        chief_complaint="Vaginal discharge, itching, cottage-cheese appearance",
        vitals=Vitals(heart_rate=70, blood_pressure="115/72",
                      spo2=99, temperature=36.8, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["recent_antibiotic_course_for_UTI"],
        hidden_vitals_detail={"pH": "4.0 (normal)", "KOH_prep": "hyphae_present"},
        true_diagnosis="vulvovaginal_candidiasis",
        true_urgency=3,
        test_prerequisites={
            "cbc": ["urinalysis"],
        },
        required_investigations=["urinalysis", "cbc"],
        test_turnaround=turnaround(["urinalysis", "cbc"]),
        correct_disposition="discharge",
        safe_medications=["fluconazole", "topical_antifungal"],
    ),

    PatientCase(
        patient_id="P028",
        age=55, sex="M",
        chief_complaint="Itchy red rash on both forearms after gardening",
        vitals=Vitals(heart_rate=76, blood_pressure="128/82",
                      spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["new_gardening_gloves_latex"],
        hidden_vitals_detail={"distribution": "bilateral_forearms_glove_pattern", "vesicles": "present"},
        true_diagnosis="contact_dermatitis",
        true_urgency=3,
        test_prerequisites={
            "cbc": ["physical_exam"],
        },
        required_investigations=["physical_exam", "cbc", "urinalysis", "electrolytes"],
        test_turnaround=turnaround(["physical_exam", "cbc", "urinalysis", "electrolytes"]),
        correct_disposition="discharge",
        safe_medications=["hydrocortisone_cream", "antihistamine"],
    ),

    PatientCase(
        patient_id="P029",
        age=38, sex="F",
        chief_complaint="No bowel movement for 4 days, bloating and cramps",
        vitals=Vitals(heart_rate=72, blood_pressure="120/78",
                      spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["recent_opioid_prescription_post_dental"],
        hidden_vitals_detail={"abdominal_distension": "mild", "bowel_sounds": "hypoactive"},
        true_diagnosis="constipation",
        true_urgency=3,
        test_prerequisites={
            "cbc": ["physical_exam"],
        },
        required_investigations=["physical_exam", "electrolytes", "cbc", "urinalysis"],
        test_turnaround=turnaround(["physical_exam", "electrolytes", "cbc", "urinalysis"]),
        correct_disposition="discharge",
        safe_medications=["polyethylene_glycol", "docusate"],
    ),

    PatientCase(
        patient_id="P030",
        age=26, sex="M",
        chief_complaint="Runny nose, sore throat, mild cough, feeling run down",
        vitals=Vitals(heart_rate=74, blood_pressure="118/75",
                      spo2=99, temperature=37.2, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        hidden_history=["office_outbreak_of_influenza"],
        hidden_vitals_detail={"pharynx": "mildly_red_no_exudate", "lymph_nodes": "not_enlarged"},
        true_diagnosis="viral_upper_respiratory_infection",
        true_urgency=3,
        test_prerequisites={
            "cbc": ["physical_exam"],
        },
        required_investigations=["physical_exam", "cbc", "urinalysis"],
        test_turnaround=turnaround(["physical_exam", "cbc", "urinalysis"]),
        correct_disposition="discharge",
        safe_medications=["symptomatic_treatment", "rest", "fluids"],
    ),
]


def get_cases_by_urgency(urgency: int):
    return [c for c in PATIENT_CASES if c.true_urgency == urgency]

def get_case_by_id(patient_id: str):
    for c in PATIENT_CASES:
        if c.patient_id == patient_id:
            return c
    return None
