# patient_cases.py - Synthetic Patient Cases with Ground Truth
from models import PatientCase, Vitals

# ============================================================================
# SYNTHETIC PATIENT CASES (30 cases across urgency levels)
# ============================================================================

PATIENT_CASES = [
    # IMMEDIATE (Urgency 1) - 10 cases
    PatientCase(
        patient_id="P001",
        age=54,
        sex="M",
        chief_complaint="Chest pain radiating to left arm, sweating",
        vitals=Vitals(heart_rate=110, blood_pressure="90/60", spo2=94, temperature=37.8, respiratory_rate=22),
        history=["hypertension", "type_2_diabetes"],
        current_medications=["metformin", "lisinopril"],
        allergies=["penicillin"],
        true_diagnosis="acute_myocardial_infarction",
        true_urgency=1,
        required_investigations=["troponin", "ecg", "cbc"],
        correct_disposition="admit",
        safe_medications=["aspirin", "nitroglycerin", "morphine"]
    ),
    
    PatientCase(
        patient_id="P002",
        age=67,
        sex="F",
        chief_complaint="Severe shortness of breath, unable to speak full sentences",
        vitals=Vitals(heart_rate=125, blood_pressure="180/110", spo2=88, temperature=36.9, respiratory_rate=32),
        history=["copd", "heart_failure"],
        current_medications=["furosemide", "albuterol"],
        allergies=[],
        true_diagnosis="acute_pulmonary_edema",
        true_urgency=1,
        required_investigations=["cxr", "ecg", "bnp"],
        correct_disposition="admit",
        safe_medications=["furosemide", "oxygen", "nitroglycerin"]
    ),
    
    PatientCase(
        patient_id="P003",
        age=32,
        sex="M",
        chief_complaint="Stabbing abdominal pain, rigid abdomen",
        vitals=Vitals(heart_rate=115, blood_pressure="85/55", spo2=96, temperature=38.9, respiratory_rate=24),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="perforated_appendicitis",
        true_urgency=1,
        required_investigations=["ct_abdomen", "cbc", "lactate"],
        correct_disposition="admit",
        safe_medications=["morphine", "ceftriaxone", "metronidazole"]
    ),
    
    PatientCase(
        patient_id="P004",
        age=45,
        sex="F",
        chief_complaint="Sudden severe headache, worst of life",
        vitals=Vitals(heart_rate=95, blood_pressure="165/95", spo2=98, temperature=37.2, respiratory_rate=18),
        history=["migraine"],
        current_medications=["sumatriptan"],
        allergies=[],
        true_diagnosis="subarachnoid_hemorrhage",
        true_urgency=1,
        required_investigations=["ct_head", "lumbar_puncture"],
        correct_disposition="admit",
        safe_medications=["nimodipine", "pain_control"]
    ),
    
    PatientCase(
        patient_id="P005",
        age=28,
        sex="M",
        chief_complaint="Difficulty breathing after bee sting, facial swelling",
        vitals=Vitals(heart_rate=130, blood_pressure="80/50", spo2=90, temperature=37.0, respiratory_rate=28),
        history=[],
        current_medications=[],
        allergies=["bee_venom"],
        true_diagnosis="anaphylaxis",
        true_urgency=1,
        required_investigations=["ecg", "cbc", "electrolytes"],
        correct_disposition="admit",
        safe_medications=["epinephrine", "diphenhydramine", "methylprednisolone"]
    ),
    
    PatientCase(
        patient_id="P006",
        age=71,
        sex="M",
        chief_complaint="Sudden right-sided weakness, slurred speech",
        vitals=Vitals(heart_rate=88, blood_pressure="155/90", spo2=97, temperature=36.8, respiratory_rate=16),
        history=["atrial_fibrillation", "hypertension"],
        current_medications=["warfarin", "metoprolol"],
        allergies=[],
        true_diagnosis="ischemic_stroke",
        true_urgency=1,
        required_investigations=["ct_head", "cbc", "inr"],
        correct_disposition="admit",
        safe_medications=["aspirin", "tpa_if_eligible"]
    ),
    
    PatientCase(
        patient_id="P007",
        age=19,
        sex="F",
        chief_complaint="Vaginal bleeding, 8 weeks pregnant, severe abdominal pain",
        vitals=Vitals(heart_rate=118, blood_pressure="88/60", spo2=98, temperature=37.3, respiratory_rate=20),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="ectopic_pregnancy",
        true_urgency=1,
        required_investigations=["ultrasound", "bhcg", "cbc"],
        correct_disposition="admit",
        safe_medications=["pain_control", "possible_surgery"]
    ),
    
    PatientCase(
        patient_id="P008",
        age=55,
        sex="M",
        chief_complaint="Vomiting blood, black tarry stools",
        vitals=Vitals(heart_rate=120, blood_pressure="85/50", spo2=96, temperature=36.5, respiratory_rate=22),
        history=["cirrhosis", "alcohol_abuse"],
        current_medications=[],
        allergies=[],
        true_diagnosis="upper_gi_bleed",
        true_urgency=1,
        required_investigations=["cbc", "inr", "endoscopy"],
        correct_disposition="admit",
        safe_medications=["proton_pump_inhibitor", "octreotide"]
    ),
    
    PatientCase(
        patient_id="P009",
        age=62,
        sex="F",
        chief_complaint="Unconscious, blood glucose 28 mg/dL",
        vitals=Vitals(heart_rate=75, blood_pressure="110/70", spo2=99, temperature=36.2, respiratory_rate=14),
        history=["type_1_diabetes"],
        current_medications=["insulin"],
        allergies=[],
        true_diagnosis="severe_hypoglycemia",
        true_urgency=1,
        required_investigations=["blood_glucose", "cbc", "electrolytes"],
        correct_disposition="admit",
        safe_medications=["dextrose", "glucagon"]
    ),
    
    PatientCase(
        patient_id="P010",
        age=38,
        sex="M",
        chief_complaint="Crush injury to leg, pulseless foot",
        vitals=Vitals(heart_rate=110, blood_pressure="95/60", spo2=97, temperature=37.0, respiratory_rate=24),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="compartment_syndrome",
        true_urgency=1,
        required_investigations=["xray_leg", "compartment_pressure"],
        correct_disposition="admit",
        safe_medications=["morphine", "immediate_surgery"]
    ),
    
    # URGENT (Urgency 2) - 10 cases
    PatientCase(
        patient_id="P011",
        age=42,
        sex="F",
        chief_complaint="Right lower quadrant pain, fever",
        vitals=Vitals(heart_rate=98, blood_pressure="125/80", spo2=98, temperature=38.5, respiratory_rate=18),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="appendicitis",
        true_urgency=2,
        required_investigations=["ct_abdomen", "cbc", "urinalysis"],
        correct_disposition="admit",
        safe_medications=["morphine", "ceftriaxone"]
    ),
    
    PatientCase(
        patient_id="P012",
        age=35,
        sex="M",
        chief_complaint="Fever 39.5°C, productive cough, chills",
        vitals=Vitals(heart_rate=105, blood_pressure="118/75", spo2=92, temperature=39.5, respiratory_rate=22),
        history=["asthma"],
        current_medications=["albuterol"],
        allergies=[],
        true_diagnosis="pneumonia",
        true_urgency=2,
        required_investigations=["cxr", "cbc", "blood_culture"],
        correct_disposition="admit",
        safe_medications=["ceftriaxone", "azithromycin", "oxygen"]
    ),
    
    PatientCase(
        patient_id="P013",
        age=58,
        sex="F",
        chief_complaint="Urinary frequency, burning, flank pain",
        vitals=Vitals(heart_rate=92, blood_pressure="130/85", spo2=98, temperature=38.8, respiratory_rate=16),
        history=["recurrent_uti"],
        current_medications=[],
        allergies=["sulfa"],
        true_diagnosis="pyelonephritis",
        true_urgency=2,
        required_investigations=["urinalysis", "urine_culture", "cbc"],
        correct_disposition="admit",
        safe_medications=["ciprofloxacin", "pain_control"]
    ),
    
    PatientCase(
        patient_id="P014",
        age=27,
        sex="M",
        chief_complaint="Laceration on forearm, bleeding controlled",
        vitals=Vitals(heart_rate=85, blood_pressure="120/78", spo2=99, temperature=37.0, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="laceration_requiring_sutures",
        true_urgency=2,
        required_investigations=["cbc", "inr"],
        correct_disposition="discharge",
        safe_medications=["local_anesthetic", "tetanus", "antibiotics_if_dirty"]
    ),
    
    PatientCase(
        patient_id="P015",
        age=65,
        sex="F",
        chief_complaint="Atrial fibrillation with rapid rate, palpitations",
        vitals=Vitals(heart_rate=145, blood_pressure="135/85", spo2=96, temperature=37.1, respiratory_rate=18),
        history=["atrial_fibrillation"],
        current_medications=["warfarin"],
        allergies=[],
        true_diagnosis="afib_with_rvr",
        true_urgency=2,
        required_investigations=["ecg", "electrolytes", "inr"],
        correct_disposition="admit",
        safe_medications=["diltiazem", "metoprolol"]
    ),
    
    PatientCase(
        patient_id="P016",
        age=50,
        sex="M",
        chief_complaint="Fall from ladder, head strike, brief LOC",
        vitals=Vitals(heart_rate=78, blood_pressure="140/88", spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="mild_traumatic_brain_injury",
        true_urgency=2,
        required_investigations=["ct_head", "cbc"],
        correct_disposition="discharge",
        safe_medications=["acetaminophen", "concussion_precautions"]
    ),
    
    PatientCase(
        patient_id="P017",
        age=29,
        sex="F",
        chief_complaint="Asthma exacerbation, wheezing",
        vitals=Vitals(heart_rate=102, blood_pressure="118/72", spo2=91, temperature=37.2, respiratory_rate=26),
        history=["asthma"],
        current_medications=["albuterol", "fluticasone"],
        allergies=[],
        true_diagnosis="asthma_exacerbation",
        true_urgency=2,
        required_investigations=["cxr", "cbc", "electrolytes"],
        correct_disposition="discharge",
        safe_medications=["albuterol_nebs", "prednisone", "home_inhaler"]
    ),
    
    PatientCase(
        patient_id="P018",
        age=44,
        sex="M",
        chief_complaint="Severe migraine, photophobia, nausea",
        vitals=Vitals(heart_rate=88, blood_pressure="145/90", spo2=99, temperature=37.0, respiratory_rate=14),
        history=["migraine"],
        current_medications=["sumatriptan"],
        allergies=[],
        true_diagnosis="migraine",
        true_urgency=2,
        required_investigations=["ct_head", "cbc", "electrolytes"],
        correct_disposition="discharge",
        safe_medications=["sumatriptan", "metoclopramide", "ketorolac"]
    ),
    
    PatientCase(
        patient_id="P019",
        age=72,
        sex="F",
        chief_complaint="Dizziness, near syncope",
        vitals=Vitals(heart_rate=58, blood_pressure="100/65", spo2=98, temperature=36.8, respiratory_rate=14),
        history=["hypertension", "atrial_fibrillation"],
        current_medications=["metoprolol", "warfarin"],
        allergies=[],
        true_diagnosis="bradycardia",
        true_urgency=2,
        required_investigations=["ecg", "electrolytes"],
        correct_disposition="admit",
        safe_medications=["hold_beta_blocker", "possible_pacing"]
    ),
    
    PatientCase(
        patient_id="P020",
        age=33,
        sex="M",
        chief_complaint="Cellulitis of leg, red, swollen, painful",
        vitals=Vitals(heart_rate=92, blood_pressure="128/82", spo2=99, temperature=38.2, respiratory_rate=16),
        history=["diabetes"],
        current_medications=["metformin"],
        allergies=[],
        true_diagnosis="cellulitis",
        true_urgency=2,
        required_investigations=["blood_culture", "cbc"],
        correct_disposition="discharge",
        safe_medications=["cephalexin", "elevation", "follow_up_48h"]
    ),
    
    # NON-URGENT (Urgency 3) - 10 cases
    PatientCase(
        patient_id="P021",
        age=25,
        sex="F",
        chief_complaint="Sore throat, mild fever",
        vitals=Vitals(heart_rate=78, blood_pressure="118/75", spo2=99, temperature=37.8, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="pharyngitis",
        true_urgency=3,
        required_investigations=["rapid_strep", "cbc"],
        correct_disposition="discharge",
        safe_medications=["acetaminophen", "throat_lozenges", "antibiotics_if_strep"]
    ),
    
    PatientCase(
        patient_id="P022",
        age=40,
        sex="M",
        chief_complaint="Ankle sprain, able to bear weight",
        vitals=Vitals(heart_rate=72, blood_pressure="125/80", spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="ankle_sprain_grade1",
        true_urgency=3,
        required_investigations=["xray_ankle", "cbc"],
        correct_disposition="discharge",
        safe_medications=["ibuprofen", "rice_protocol"]
    ),
    
    PatientCase(
        patient_id="P023",
        age=31,
        sex="F",
        chief_complaint="Urinary tract infection symptoms, no fever",
        vitals=Vitals(heart_rate=75, blood_pressure="120/78", spo2=99, temperature=37.1, respiratory_rate=14),
        history=["recurrent_uti"],
        current_medications=[],
        allergies=[],
        true_diagnosis="uncomplicated_uti",
        true_urgency=3,
        required_investigations=["urinalysis", "urine_culture"],
        correct_disposition="discharge",
        safe_medications=["nitrofurantoin", "phenazopyridine"]
    ),
    
    PatientCase(
        patient_id="P024",
        age=22,
        sex="M",
        chief_complaint="Allergic rhinitis, sneezing, itchy eyes",
        vitals=Vitals(heart_rate=70, blood_pressure="118/72", spo2=99, temperature=36.8, respiratory_rate=14),
        history=["seasonal_allergies"],
        current_medications=[],
        allergies=[],
        true_diagnosis="allergic_rhinitis",
        true_urgency=3,
        required_investigations=["cbc", "urinalysis"],
        correct_disposition="discharge",
        safe_medications=["cetirizine", "nasal_steroid"]
    ),
    
    PatientCase(
        patient_id="P025",
        age=48,
        sex="F",
        chief_complaint="Acute low back pain after lifting",
        vitals=Vitals(heart_rate=74, blood_pressure="122/80", spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="mechanical_low_back_pain",
        true_urgency=3,
        required_investigations=["cbc", "urinalysis", "electrolytes"],
        correct_disposition="discharge",
        safe_medications=["ibuprofen", "muscle_relaxant", "physical_therapy"]
    ),
    
    PatientCase(
        patient_id="P026",
        age=35,
        sex="M",
        chief_complaint="Conjunctivitis, red eye, discharge",
        vitals=Vitals(heart_rate=72, blood_pressure="120/78", spo2=99, temperature=37.0, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="bacterial_conjunctivitis",
        true_urgency=3,
        required_investigations=["cbc", "urinalysis"],
        correct_disposition="discharge",
        safe_medications=["erythromycin_ointment"]
    ),
    
    PatientCase(
        patient_id="P027",
        age=28,
        sex="F",
        chief_complaint="Yeast infection symptoms",
        vitals=Vitals(heart_rate=70, blood_pressure="115/72", spo2=99, temperature=36.8, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="vulvovaginal_candidiasis",
        true_urgency=3,
        required_investigations=["urinalysis", "cbc"],
        correct_disposition="discharge",
        safe_medications=["fluconazole", "topical_antifungal"]
    ),
    
    PatientCase(
        patient_id="P028",
        age=55,
        sex="M",
        chief_complaint="Minor skin rash, itchy",
        vitals=Vitals(heart_rate=76, blood_pressure="128/82", spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="contact_dermatitis",
        true_urgency=3,
        required_investigations=["cbc", "urinalysis", "electrolytes"],
        correct_disposition="discharge",
        safe_medications=["hydrocortisone_cream", "antihistamine"]
    ),
    
    PatientCase(
        patient_id="P029",
        age=38,
        sex="F",
        chief_complaint="Constipation for 3 days",
        vitals=Vitals(heart_rate=72, blood_pressure="120/78", spo2=99, temperature=36.9, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="constipation",
        true_urgency=3,
        required_investigations=["electrolytes", "cbc", "urinalysis"],
        correct_disposition="discharge",
        safe_medications=["polyethylene_glycol", "docusate"]
    ),
    
    PatientCase(
        patient_id="P030",
        age=26,
        sex="M",
        chief_complaint="Common cold symptoms, runny nose",
        vitals=Vitals(heart_rate=74, blood_pressure="118/75", spo2=99, temperature=37.2, respiratory_rate=14),
        history=[],
        current_medications=[],
        allergies=[],
        true_diagnosis="viral_upper_respiratory_infection",
        true_urgency=3,
        required_investigations=["cbc", "urinalysis"],
        correct_disposition="discharge",
        safe_medications=["symptomatic_treatment", "rest", "fluids"]
    ),
]

# Helper function to get cases by urgency
def get_cases_by_urgency(urgency: int):
    return [case for case in PATIENT_CASES if case.true_urgency == urgency]

# Helper function to get case by ID
def get_case_by_id(patient_id: str):
    for case in PATIENT_CASES:
        if case.patient_id == patient_id:
            return case
    return None