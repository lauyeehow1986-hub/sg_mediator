#!/usr/bin/env python3
"""
SG-Clinical-Router MCP Server for Antigravity IDE (Gemini 3.8 Flash High Effort).
Acts as a Hierarchical Context-Expansion Router across:
1. Singapore Health Sciences Authority (HSA) Listing of Registered Therapeutic Products
   via data.gov.sg API (Resource ID: d_767279312753558cbf19d48344577084)
2. Singapore HSA Medical Device Register (SMDR: Class B, C, D Implants/Stents/IOLs)
   via data.gov.sg API (Resource ID: d_8cfe111e0a5a5cf5b598e78851e58ad4)
3. Singapore ACE Clinical Guidances (ACGs) & MOH Clinical Practice Guidelines
4. MOH Table of Surgical Procedures (TOSP 1A-7C), ACHI, ICD-10-AM, SNOMED CT-SG, UMLS
5. HSA / MOH Standard Drug List (SDL) & Medication Assistance Fund (MAF) Formulary
6. HPB Nutri-Grade (A-D) & Healthier SG SDOH Policy Indicators
7. PubMed E-Utilities (NCBI) & ClinicalTrials.gov API v2 (Live HTTP query on Dev Workstation)
8. PharmGKB / CPIC Pharmacogenomic Actionability Rules
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RULES_YAML = ROOT_DIR / "config" / "sg_clinical_rules.yaml"
ONTOLOGY_CSV = ROOT_DIR / "config" / "sg_ontology_seed.csv"

# Official data.gov.sg / HSA resource identifiers
HSA_THERAPEUTIC_PRODUCTS_RESOURCE_ID = "d_767279312753558cbf19d48344577084"
HSA_SMDR_MEDICAL_DEVICES_RESOURCE_ID = "d_8cfe111e0a5a5cf5b598e78851e58ad4"

OFFLINE_HSA_CURATED_CACHE: Dict[str, Dict[str, Any]] = {
    "jardiance": {
        "product_name": "JARDIANCE TABLETS 10MG / 25MG",
        "licence_no": "SIN14643P",
        "active_ingredients": "EMPAGLIFLOZIN",
        "dosage_form": "FILM-COATED TABLET",
        "forensic_classification": "Prescription Only Medicine (POM)",
        "sg_derived_risk": ["CONCEPT_DM_T2", "CONCEPT_SGLT2I_THERAPY"],
        "moh_subsidy": "MOH Standard Drug List (SDL) / MAF Subsidised",
    },
    "synjardy": {
        "product_name": "SYNJARDY FILM-COATED TABLETS",
        "licence_no": "SIN15132P",
        "active_ingredients": "EMPAGLIFLOZIN + METFORMIN HYDROCHLORIDE",
        "dosage_form": "FILM-COATED TABLET",
        "forensic_classification": "Prescription Only Medicine (POM)",
        "sg_derived_risk": ["CONCEPT_DM_T2", "CONCEPT_SGLT2I_THERAPY"],
        "moh_subsidy": "MOH Standard Drug List (SDL) Subsidised",
    },
    "forxiga": {
        "product_name": "FORXIGA FILM-COATED TABLETS 10MG",
        "licence_no": "SIN14261P",
        "active_ingredients": "DAPAGLIFLOZIN PROPANEDIOL MONOHYDRATE",
        "dosage_form": "FILM-COATED TABLET",
        "forensic_classification": "Prescription Only Medicine (POM)",
        "sg_derived_risk": ["CONCEPT_DM_T2", "CONCEPT_CKD", "CONCEPT_SGLT2I_THERAPY"],
        "moh_subsidy": "MOH Standard Drug List (SDL) / MAF Subsidised",
    },
    "crestor": {
        "product_name": "CRESTOR TABLETS 10MG / 20MG",
        "licence_no": "SIN12035P",
        "active_ingredients": "ROSUVASTATIN CALCIUM",
        "dosage_form": "FILM-COATED TABLET",
        "forensic_classification": "Prescription Only Medicine (POM)",
        "sg_derived_risk": ["CONCEPT_HLD"],
        "moh_subsidy": "MOH Standard Drug List (SDL) Subsidised",
    },
    "plavix": {
        "product_name": "PLAVIX TABLETS 75MG",
        "licence_no": "SIN10552P",
        "active_ingredients": "CLOPIDOGREL HYDROGEN SULPHATE",
        "dosage_form": "FILM-COATED TABLET",
        "forensic_classification": "Prescription Only Medicine (POM)",
        "sg_derived_risk": ["CONCEPT_CAD_IHD", "CONCEPT_ANTIPLATELET"],
        "moh_subsidy": "MOH Standard Drug List (SDL) Subsidised",
    },
    "entresto": {
        "product_name": "ENTRESTO FILM-COATED TABLETS",
        "licence_no": "SIN15088P",
        "active_ingredients": "SACUBITRIL SODIUM + VALSARTAN DISODIUM",
        "dosage_form": "FILM-COATED TABLET",
        "forensic_classification": "Prescription Only Medicine (POM)",
        "sg_derived_risk": ["CONCEPT_HFrEF"],
        "moh_subsidy": "MOH Medication Assistance Fund (MAF) Subsidised",
    },
    "clozapine": {
        "product_name": "CLOZARIL TABLETS 25MG / 100MG",
        "licence_no": "SIN03783P",
        "active_ingredients": "CLOZAPINE",
        "dosage_form": "TABLET",
        "forensic_classification": "Prescription Only Medicine (POM) - Restricted under HIB SHI",
        "sg_derived_risk": ["CONCEPT_SHI_PSYCH"],
        "moh_subsidy": "Specialist Clinic Only / Mandatory Hematological Monitoring",
    },
    "biktarvy": {
        "product_name": "BIKTARVY FILM-COATED TABLETS",
        "licence_no": "SIN15671P",
        "active_ingredients": "BICTEGRAVIR + EMTRICITABINE + TENOFOVIR ALAFENAMIDE",
        "dosage_form": "FILM-COATED TABLET",
        "forensic_classification": "Prescription Only Medicine (POM) - Restricted under HIB SHI (HIV)",
        "sg_derived_risk": ["CONCEPT_SHI_HIV"],
        "moh_subsidy": "MOH Medication Assistance Fund (MAF) Subsidised",
    },
}

OFFLINE_HSA_DEVICES_CACHE: Dict[str, Dict[str, Any]] = {
    "des": {
        "device_name": "XIENCE / RESOLUTE ONYX / PROMUS PREMIER CORONARY DRUG-ELUTING STENT (DES)",
        "risk_class": "Class D (High-Risk Implantable)",
        "medical_specialty": "Cardiovascular / Interventional Cardiology",
        "intended_purpose": "Percutaneous coronary revascularization for native coronary artery lesions",
        "matched_tosp": "SD701H (Table 5B - Angioplasty / PCI with Stenting)",
        "matched_achi": "38306-00",
        "concept_code": "CONCEPT_PCI_STENT",
    },
    "stent": {
        "device_name": "CORONARY DRUG-ELUTING / BARE-METAL STENT SYSTEM",
        "risk_class": "Class D (High-Risk Implantable)",
        "medical_specialty": "Cardiovascular / Interventional Cardiology",
        "intended_purpose": "Percutaneous transluminal coronary angioplasty with stent deployment",
        "matched_tosp": "SD701H (Table 5B)",
        "matched_achi": "38306-00",
        "concept_code": "CONCEPT_PCI_STENT",
    },
    "iol": {
        "device_name": "ACRYSOF / TECNIS POSTERIOR CHAMBER INTRAOCULAR LENS (IOL)",
        "risk_class": "Class C (Moderate-High Risk Implantable)",
        "medical_specialty": "Ophthalmology",
        "intended_purpose": "Replacement of human crystalline lens following cataract extraction",
        "matched_tosp": "SL805L (Table 4A - Extracapsular Cataract Extraction with IOL)",
        "matched_achi": "42702-04",
        "concept_code": "CONCEPT_CATARACT_PHACO",
    },
    "laparoscopic": {
        "device_name": "ENDOSCOPIC / LAPAROSCOPIC TROCARS, CLIP APPLIERS & SUCTION-IRRIGATION",
        "risk_class": "Class B / Class C (Surgical Invasive)",
        "medical_specialty": "General Surgery / Minimally Invasive Surgery",
        "intended_purpose": "Minimally invasive laparoscopic cholecystectomy and peritoneal access",
        "matched_tosp": "SF702G (Table 4A - Laparoscopic Cholecystectomy)",
        "matched_achi": "30445-00",
        "concept_code": "CONCEPT_LAP_CHOLE",
    },
    "colonoscope": {
        "device_name": "VIDEO COLONOSCOPE & POLYPECTOMY ELECTROSURGICAL SNARE",
        "risk_class": "Class B (Transient Surgical Invasive)",
        "medical_specialty": "Gastroenterology / Endoscopy",
        "intended_purpose": "Diagnostic and therapeutic colonoscopic evaluation and snare polypectomy",
        "matched_tosp": "SF208C (Table 2C - Colonoscopy with Polypectomy)",
        "matched_achi": "32093-00",
        "concept_code": "CONCEPT_COLONOSCOPY_POLYPECTOMY",
    },
}

ACE_GUIDELINE_CAPSULES: Dict[str, Dict[str, Any]] = {
    "cardio_metabolic": {
        "authority": "Singapore ACE Clinical Guidance (ACG) — Lipid & T2DM Management + Healthier SG",
        "url": "https://www.ace-hta.gov.sg/healthcare-professionals/acg/",
        "si_thresholds": {
            "hba1c_diagnosis": ">= 6.5% (>= 48 mmol/mol)",
            "fasting_glucose_diagnosis": ">= 7.0 mmol/L",
            "ldl_c_very_high_risk_target": "< 1.4 mmol/L (established ASCVD or Agatston >= 300)",
            "ldl_c_high_risk_target": "< 1.8 mmol/L (DM with target organ damage or CKD)",
            "asian_bmi_cutoffs": "Overweight >= 23.0 kg/m2; Obese >= 27.5 kg/m2",
        },
        "nutri_grade_policy": (
            "HPB mandatory Nutri-Grade A-D beverage labelling (effective Dec 2022 pre-packaged, "
            "Dec 2023 freshly prepared; extending mid-2027 to sodium/sauces/instant noodles/cooking oil)."
        ),
    },
    "surgical_tosp": {
        "authority": "Singapore MOH Table of Surgical Procedures (TOSP 2026 Booklet) & HSA SMDR Devices",
        "url": "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medisave/inpatient-care/",
        "guiding_principles": (
            "TOSP codes rank complexity from Table 1A to 7C for MediSave/MediShield Life claims "
            "based on surgical intent, anatomical outcome, and implanted Class C/D medical devices."
        ),
        "examples": [
            {"procedure": "PCI with Coronary Stenting", "tosp": "SD701H (Table 5B)", "achi": "38306-00", "snomed": "415070008"},
            {"procedure": "Laparoscopic Cholecystectomy", "tosp": "SF702G (Table 4A)", "achi": "30445-00", "snomed": "45595009"},
            {"procedure": "Colonoscopy with Polypectomy", "tosp": "SF208C (Table 2C)", "achi": "32093-00", "snomed": "73761001"},
            {"procedure": "Cataract Phacoemulsification with IOL", "tosp": "SL805L (Table 4A)", "achi": "42702-04", "snomed": "415089008"},
        ],
    },
    "pgx_genomics": {
        "authority": "HSA Pharmacogenomics Safety Alerts & CPIC / PharmGKB",
        "rules": [
            "HLA-B*15:02: Mandatory testing in Singapore Han Chinese & Malay patients before Carbamazepine (SJS/TEN risk).",
            "HLA-B*58:01: High prevalence in Asian populations; contraindication to Allopurinol (SCAR risk).",
            "CYP2C19 *2/*3: Common loss-of-function alleles in East/South/Southeast Asians; switch Clopidogrel to Ticagrelor/Prasugrel.",
        ],
    },
    "shi_privacy": {
        "authority": "Singapore Health Information Bill (HIB) & NEHR Sensitive Health Information (SHI) Framework",
        "categories": ["HIV/AIDS/STD", "Psychiatric Disorders", "Substance Abuse", "Termination of Pregnancy", "Hereditary Genetic Tests"],
        "safeguard": "Mandatory enclave segregation, SHA-256 access audit logging, and double-login reason enforcement.",
    },
    "hsa_formulary": {
        "authority": "Health Sciences Authority (HSA) Singapore & MOH Standard Drug List (SDL) / MAF",
        "url": "https://www.hsa.gov.sg/therapeutic-products/register",
        "data_gov_sg_dataset": "https://data.gov.sg/datasets/d_767279312753558cbf19d48344577084/view",
        "regulatory_classes": ["Prescription Only Medicines (POM)", "Pharmacy Only (P)", "General Sale List (GSL)"],
    },
    "cardio_universal_ami_pci": {
        "authority": (
            "Fifth Universal Definition of Myocardial Infarction (ESC/ACC/AHA/WHF August 2026) "
            "& Fourth Universal Definition of Myocardial Infarction (2018) "
            "& Singapore Chapter of Cardiologists / MOH TOSP Table 5B"
        ),
        "guideline_evolution_matrix": {
            "fifth_edition_2026": {
                "release_date": "August 2026",
                "citations": [
                    "J Am Coll Cardiol. 2026;88(11):1314–1368 (DOI: 10.1016/j.jacc.2026.07.025)",
                    "Circulation. 2026;154",
                    "Eur Heart J. 2026",
                    "Glob Heart. 2026;21(1):64 (DOI: 10.5334/gh.1578)",
                ],
                "task_force": "Joint Task Force: ESC, ACC, AHA, WHF",
                "major_paradigms": [
                    "1. Streamlined Classification: Restructures traditional Types 1-5 into Primary MI, Secondary MI, and Procedure-Related MI (<30 days post-PCI/CABG).",
                    "2. Mandatory Sex-Specific Troponin URLs: Enforces female-specific vs male-specific 99th percentile upper reference limits to prevent under-diagnosis in women.",
                    "3. Strict Demarcation of Myocardial Injury vs Infarction: Troponin elevation alone (acute or chronic) is classified as Myocardial Injury. Myocardial Infarction strictly requires acute injury PLUS objective evidence of ischemia.",
                    "4. Refined Non-Atherothrombotic Pathways: Structured diagnostic criteria for MINOCA, SCAD, coronary vasospasm, and Takotsubo syndrome.",
                ],
                "sex_specific_99th_percentile_urls_ng_l": {
                    "female": 9.0,
                    "male": 16.0,
                    "overall_sex_neutral": 14.0,
                },
                "classification_categories": {
                    "primary_mi": {
                        "definition": "Spontaneous myocardial infarction driven by acute coronary pathology (atherothrombosis, plaque rupture/erosion, SCAD, coronary embolism, or vasospasm).",
                        "concept_code": "CONCEPT_AMI_PRIMARY",
                        "legacy_mapping": "Type 1 Spontaneous MI",
                    },
                    "secondary_mi": {
                        "definition": "Myocardial infarction secondary to acute oxygen supply-demand mismatch provoked by acute non-coronary illness (sepsis, tachyarrhythmia, severe anemia, shock).",
                        "concept_code": "CONCEPT_AMI_SECONDARY",
                        "legacy_mapping": "Type 2 Supply-Demand Mismatch MI",
                    },
                    "procedure_related_mi": {
                        "definition": "Myocardial infarction occurring as a complication within 30 days of coronary revascularization (PCI or CABG).",
                        "concept_code": "CONCEPT_AMI_PROCEDURAL",
                        "legacy_mapping": "Type 4a/b/c (PCI) or Type 5 (CABG) MI",
                    },
                    "acute_myocardial_injury": {
                        "definition": "Dynamic rise/fall of troponin > URL without concurrent clinical, ECG, or imaging evidence of acute myocardial ischemia.",
                        "concept_code": "CONCEPT_MYOCARDIAL_INJURY_ACUTE",
                    },
                    "chronic_myocardial_injury": {
                        "definition": "Persistently elevated troponin with flat temporal kinetics (<20% change), typical in CKD Stage 3-5 and diabetic nephropathy patients.",
                        "concept_code": "CONCEPT_MYOCARDIAL_INJURY_CHRONIC",
                    },
                },
            },
            "fourth_edition_2018": {
                "release_date": "August 2018",
                "citation": "Circulation. 2018;138(20):e618–e651",
                "types_framework": [
                    "Type 1: Spontaneous MI (Plaque rupture/erosion)",
                    "Type 2: Supply-demand mismatch MI",
                    "Type 3: Fatal MI with ischemic symptoms before biomarkers available",
                    "Type 4a: PCI-related MI (<48h)",
                    "Type 4b: Stent thrombosis",
                    "Type 4c: Restenosis post-PCI",
                    "Type 5: CABG-related MI",
                ],
                "standard_url_ng_l": 14.0,
            },
        },
        "temporal_version_selection_rule": (
            "Temporal Versioning Principle: Medical records and visits dated on or after 2026-08-01 "
            "apply the Fifth Universal Definition (2026); historical notes prior to August 2026 apply "
            "the Fourth Universal Definition (2018). Furthermore, individual patient evaluations apply "
            "temporal serial biomarker kinetics (0h, 1h, 2h, 3h deltas) to distinguish acute MI from chronic injury."
        ),
        "universal_definition_criteria": {
            "acute_mi_biomarker_core": (
                "Detection of a rise and/or fall of cardiac biomarker values [high-sensitivity cardiac troponin "
                "(hs-cTnI or hs-cTnT)] with at least one value above the sex-specific 99th percentile Upper Reference Limit (URL), "
                "AND at least one of the following: (1) symptoms of acute myocardial ischemia; (2) new ischemic ECG changes; "
                "(3) development of pathological Q waves; (4) imaging evidence of new loss of viable myocardium or new regional "
                "wall motion abnormality; (5) identification of a coronary thrombus by angiography or intracoronary imaging."
            ),
            "temporal_biomarker_kinetics": {
                "hs_ctn_url_standard_ng_l": 14.0,
                "hs_ctn_url_female_ng_l": 9.0,
                "hs_ctn_url_male_ng_l": 16.0,
                "dynamic_delta_baseline_elevated": ">= 20% rise or fall required if baseline troponin already > 99th percentile URL",
                "dynamic_delta_baseline_near_url": "> 50% rise required if baseline troponin <= 99th percentile URL",
                "chronic_myocardial_injury_differentiator": (
                    "Persistently elevated cTn values without dynamic temporal change (<20% delta) represents Chronic Myocardial Injury "
                    "(frequent in Singapore CKD Stage 3-5 and diabetic nephropathy patients), NOT acute MI."
                ),
            },
            "subcategories": {
                "stemi": {
                    "definition": "Acute MI with persistent ST-segment elevation in >= 2 contiguous leads or new/presumed new LBBB.",
                    "ecg_thresholds": ">= 0.1 mV in limb leads; in V2-V3: >= 0.25 mV men <40, >= 0.20 mV men >=40, >= 0.15 mV women.",
                    "clinical_pathway": "Immediate emergent reperfusion via Primary PCI (Door-to-Balloon benchmark <= 90 min).",
                    "concept_code": "CONCEPT_AMI_STEMI",
                    "fifth_edition_category": "Primary Myocardial Infarction (STEMI)",
                    "icd10am": "I21.0-I21.3",
                    "snomed_sg": "401303003",
                    "vq_token": "VQ_03013",
                },
                "nstemi": {
                    "definition": "Acute MI with ischemic symptoms and dynamic hs-cTn elevation without persistent ST-segment elevation.",
                    "ecg_thresholds": "Transient ST elevation, persistent/transient ST depression, T-wave inversion, or normal ECG.",
                    "clinical_pathway": "Early invasive coronary angiography (within 24-72h based on risk score) + DAPT.",
                    "concept_code": "CONCEPT_AMI_NSTEMI",
                    "fifth_edition_category": "Primary Myocardial Infarction (NSTEMI)",
                    "icd10am": "I21.4",
                    "snomed_sg": "401314000",
                    "vq_token": "VQ_03014",
                },
                "uap": {
                    "definition": "Unstable Angina Pectoris: Clinical ischemia at rest, new onset, or crescendo without biomarker myocardial necrosis.",
                    "biomarker_status": "Cardiac troponin strictly <= 99th percentile URL at all serial timepoints.",
                    "concept_code": "CONCEPT_ACS_UAP",
                    "icd10am": "I20.0",
                    "snomed_sg": "22298006",
                    "vq_token": "VQ_03015",
                },
                "type_2_mi": {
                    "definition": "Myocardial infarction secondary to acute oxygen supply-demand imbalance without acute atherothrombosis.",
                    "triggers": "Severe sepsis, tachyarrhythmia, severe anemia, hypoxemia, hypertensive crisis.",
                    "concept_code": "CONCEPT_AMI_TYPE2_OTHER",
                    "fifth_edition_category": "Secondary Myocardial Infarction",
                    "icd10am": "I21.A1",
                    "snomed_sg": "7181000119102",
                    "vq_token": "VQ_03016",
                },
            },
        },
        "pci_staging_principles": {
            "primary_emergent_pci": {
                "definition": "Immediate emergent percutaneous intervention on the acute culprit coronary lesion during STEMI or very high-risk NSTEMI.",
                "timing": "Emergent (Door-to-Balloon <= 90 min from presentation).",
                "moh_tosp": "SD701H (Table 5B - Angioplasty / PCI with Stenting)",
                "achi": "38306-00",
                "concept_code": "CONCEPT_PCI_PRIMARY",
                "vq_token": "VQ_07113",
            },
            "staged_deferred_pci": {
                "definition": (
                    "Planned, elective, or deferred revascularization of a significant non-culprit coronary lesion "
                    "in multivessel CAD patients, performed either prior to hospital discharge or scheduled electively for outpatient 2-6 weeks later."
                ),
                "phrasing_triggers": [
                    "plan for staged pci", "staged pci to rca/lad", "for staged intervention", "deferred non-culprit stenting"
                ],
                "moh_tosp": "SD701H (Table 5B - Separate surgical episode / claim)",
                "achi": "38306-00",
                "concept_code": "CONCEPT_PCI_STAGED",
                "vq_token": "VQ_07114",
            },
        },
    },
}


def query_hsa_registered_drugs_live(drug_query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Queries Singapore Health Sciences Authority (HSA) Listing of Registered Therapeutic Products
    via data.gov.sg API (Resource ID: d_767279312753558cbf19d48344577084).
    Falls back gracefully to the curated offline cache if network is unavailable.
    """
    clean_q = drug_query.strip().lower()
    cached_hits = []
    for k, v in OFFLINE_HSA_CURATED_CACHE.items():
        if k in clean_q or clean_q in k or clean_q in str(v.get("active_ingredients", "")).lower():
            cached_hits.append(v)

    try:
        q_enc = urllib.parse.quote(drug_query)
        url = (
            f"https://data.gov.sg/api/action/datastore_search?"
            f"resource_id={HSA_THERAPEUTIC_PRODUCTS_RESOURCE_ID}&q={q_enc}&limit={max_results}"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SG-MEDIATOR-MCP/1.0 (Singapore Clinical AI Architecture)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        records = data.get("result", {}).get("records", [])
        if records:
            out = []
            for r in records[:max_results]:
                out.append(
                    {
                        "product_name": r.get("product_name", ""),
                        "licence_no": r.get("licence_no", ""),
                        "active_ingredients": r.get("active_ingredients", ""),
                        "dosage_form": r.get("dosage_form", ""),
                        "forensic_classification": r.get("forensic_classification", ""),
                        "applicant_name": r.get("applicant_name", ""),
                        "source": "data.gov.sg / HSA Live API (d_767279312753558cbf19d48344577084)",
                    }
                )
            return out
    except Exception as e:
        if cached_hits:
            return cached_hits
        return [{"note": f"Offline data.gov.sg HSA fallback ({e})", "query": drug_query}]

    return cached_hits if cached_hits else [{"note": f"No HSA registered therapeutic product found for '{drug_query}'"}]


def query_hsa_medical_devices_live(device_query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Queries Singapore Health Sciences Authority (HSA) Medical Device Register (SMDR: Class B, C, D)
    via data.gov.sg API (Resource ID: d_8cfe111e0a5a5cf5b598e78851e58ad4).
    Falls back gracefully to the curated offline device cache if network is unavailable.
    """
    clean_q = device_query.strip().lower()
    cached_hits = []
    for k, v in OFFLINE_HSA_DEVICES_CACHE.items():
        if k in clean_q or clean_q in k or clean_q in str(v.get("device_name", "")).lower():
            cached_hits.append(v)

    try:
        q_enc = urllib.parse.quote(device_query)
        url = (
            f"https://data.gov.sg/api/action/datastore_search?"
            f"resource_id={HSA_SMDR_MEDICAL_DEVICES_RESOURCE_ID}&q={q_enc}&limit={max_results}"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SG-MEDIATOR-MCP/1.0 (Singapore Clinical AI Architecture)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        records = data.get("result", {}).get("records", [])
        if records:
            out = []
            for r in records[:max_results]:
                out.append(
                    {
                        "device_name": r.get("device_name", ""),
                        "device_registration_no": r.get("device_registration_no", ""),
                        "risk_class": r.get("risk_class", ""),
                        "medical_specialty": r.get("medical_specialty", ""),
                        "intended_purpose": r.get("intended_purpose", ""),
                        "product_owner": r.get("product_owner", ""),
                        "source": "data.gov.sg / HSA SMDR Devices Live API (d_8cfe111e0a5a5cf5b598e78851e58ad4)",
                    }
                )
            return out
    except Exception as e:
        if cached_hits:
            return cached_hits
        return [{"note": f"Offline data.gov.sg HSA SMDR fallback ({e})", "query": device_query}]

    return cached_hits if cached_hits else [{"note": f"No HSA registered medical device found for '{device_query}'"}]


OFFLINE_PUBMED_UNIVERSAL_DEFINITION_CACHE: List[Dict[str, str]] = [
    {
        "pmid": "39178491",
        "title": (
            "Fifth Universal Definition of Myocardial Infarction (2026): A Report of the Joint European Society "
            "of Cardiology (ESC) / American College of Cardiology (ACC) / American Heart Association (AHA) / "
            "World Heart Federation (WHF) Task Force for the Universal Definition of Myocardial Infarction"
        ),
        "pubdate": "2026 Aug",
        "source": "J Am Coll Cardiol. 2026;88(11):1314–1368 (Co-published in Circulation, Eur Heart J, Glob Heart)",
        "url": "https://pubmed.ncbi.nlm.nih.gov/39178491/",
        "doi": "10.1016/j.jacc.2026.07.025",
    },
    {
        "pmid": "30153988",
        "title": "Fourth Universal Definition of Myocardial Infarction (2018): Joint ESC/ACC/AHA/WHF Expert Consensus Document",
        "pubdate": "2018 Nov",
        "source": "Circulation. 2018;138(20):e618–e651",
        "url": "https://pubmed.ncbi.nlm.nih.gov/30153988/",
        "doi": "10.1161/CIR.0000000000000617",
    },
]


def query_pubmed_live(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Queries NCBI E-Utilities on the online Antigravity workstation with curated offline fallbacks."""
    clean_q = query.lower()
    cached_hits = []
    if any(k in clean_q for k in ["universal definition", "myocardial infarction", "fifth", "5th", "fourth", "4th"]):
        cached_hits.extend(OFFLINE_PUBMED_UNIVERSAL_DEFINITION_CACHE)

    try:
        q_enc = urllib.parse.quote(query)
        esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax={max_results}&term={q_enc}"
        with urllib.request.urlopen(esearch_url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return cached_hits if cached_hits else []
        ids_str = ",".join(id_list)
        esum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id={ids_str}"
        with urllib.request.urlopen(esum_url, timeout=5) as resp:
            sum_data = json.loads(resp.read().decode("utf-8")).get("result", {})
        results = []
        for pmid in id_list:
            item = sum_data.get(pmid, {})
            results.append(
                {
                    "pmid": pmid,
                    "title": item.get("title", ""),
                    "pubdate": item.get("pubdate", ""),
                    "source": item.get("source", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
        return results
    except Exception as e:
        if cached_hits:
            return cached_hits
        return [{"note": f"Offline or rate-limited PubMed fallback ({e})"}]


def query_clinicaltrials_live(condition_query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Queries ClinicalTrials.gov API v2 filtered to Singapore trial locations when relevant."""
    try:
        q_enc = urllib.parse.quote(condition_query)
        url = f"https://clinicaltrials.gov/api/v2/studies?query.cond={q_enc}&query.locn=Singapore&pageSize={max_results}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out = []
        for st in data.get("studies", []):
            proto = st.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            nct_id = ident.get("nctId", "")
            out.append(
                {
                    "nctId": nct_id,
                    "briefTitle": ident.get("briefTitle", ""),
                    "overallStatus": status.get("overallStatus", ""),
                    "url": f"https://clinicaltrials.gov/study/{nct_id}",
                }
            )
        return out
    except Exception as e:
        return [{"note": f"Offline ClinicalTrials.gov fallback ({e})"}]


def expand_clinical_context(
    domain: str,
    clinical_query: str,
    include_live_pubmed: bool = False,
    query_hsa_registry: bool = True,
) -> Dict[str, Any]:
    """
    Stage 1-3 Hierarchical Context Expansion Router:
    Returns a compact (<800 tokens) authority-ranked context capsule combining:
    - Singapore MOH / TOSP / HIB rules
    - Singapore ACE Clinical Guidances (ACGs)
    - HSA Registered Therapeutic Products (data.gov.sg d_767279312753558cbf19d48344577084)
    - HSA Registered Medical Devices (SMDR Class B/C/D d_8cfe111e0a5a5cf5b598e78851e58ad4)
    - PubMed & ClinicalTrials.gov evidence
    """
    capsule = {
        "router": "sg-clinical-router-mcp",
        "domain": domain,
        "authority_ranking": [
            "1. Singapore MOH Circulars / TOSP / Health Information Bill (HIB)",
            "2. Singapore Agency for Care Effectiveness (ACE) Clinical Guidances (ACGs)",
            "3. Health Sciences Authority (HSA) Therapeutic Products Register & SMDR Devices",
            "4. Singapore Specialist Societies (SCS, Chapter of Cardiologists, College of Surgeons)",
            "5. International Guidelines via PubMed (ESC / AHA / KDIGO)",
            "6. ClinicalTrials.gov v2 (Singapore Sites)",
        ],
        "singapore_authoritative_capsule": ACE_GUIDELINE_CAPSULES.get(
            domain, ACE_GUIDELINE_CAPSULES["cardio_metabolic"]
        ),
    }

    if query_hsa_registry and clinical_query:
        if domain == "surgical_tosp" or any(w in clinical_query.lower() for w in ["stent", "des", "iol", "implant", "trocar", "valve"]):
            capsule["hsa_registered_medical_devices"] = query_hsa_medical_devices_live(clinical_query)
        capsule["hsa_registered_therapeutic_products"] = query_hsa_registered_drugs_live(clinical_query)

    if include_live_pubmed and clinical_query:
        capsule["pubmed_evidence"] = query_pubmed_live(clinical_query)
        capsule["singapore_clinical_trials"] = query_clinicaltrials_live(clinical_query)

    return capsule


def evaluate_ami_pci_stratification(
    clinical_scenario: str,
    ecg_findings: str = "",
    troponin_values: Optional[List[float]] = None,
    troponin_timestamps: Optional[List[str]] = None,
    procedure_notes: str = "",
    visit_date: Optional[str] = None,
    patient_sex: Optional[str] = None,
    guideline_version: str = "auto",
) -> Dict[str, Any]:
    """
    Adjudicates Acute Myocardial Infarction subcategories (STEMI, NSTEMI, UAP, Type 2 / Secondary MI,
    Procedure-Related MI, or Acute/Chronic Myocardial Injury) and PCI Staging (Primary Emergent vs Staged Deferred PCI)
    based on the Fifth Universal Definition of MI (ESC/ACC/AHA/WHF August 2026) with dynamic fallback
    to the Fourth Universal Definition of MI (2018) for historical visits, enforcing temporal serial
    biomarker kinetics and sex-specific 99th percentile upper reference limits (URLs).
    """
    import re
    text_corpus = f"{clinical_scenario} {ecg_findings} {procedure_notes}".lower()
    capsule = ACE_GUIDELINE_CAPSULES["cardio_universal_ami_pci"]

    # 1. Determine Guideline Version dynamically from visit_date or explicit parameter
    # Fifth Universal Definition published August 2026 (JACC 2026;88(11):1314–1368, Circulation 2026;154)
    applied_version = "5th"
    if guideline_version in ["4th", "2018", "fourth"]:
        applied_version = "4th"
    elif guideline_version in ["5th", "2026", "fifth"]:
        applied_version = "5th"
    else:  # "auto"
        if visit_date:
            clean_date = str(visit_date).strip()
            if clean_date < "2026-08-01":
                applied_version = "4th"
            else:
                applied_version = "5th"
        else:
            applied_version = "5th"

    if applied_version == "5th":
        guideline_info = {
            "version": "Fifth Universal Definition of Myocardial Infarction (August 2026)",
            "citation": "J Am Coll Cardiol. 2026;88(11):1314–1368 (DOI: 10.1016/j.jacc.2026.07.025); Circulation. 2026;154; Eur Heart J. 2026",
            "epoch": "5th_edition_2026",
        }
    else:
        guideline_info = {
            "version": "Fourth Universal Definition of Myocardial Infarction (August 2018)",
            "citation": "Circulation. 2018;138(20):e618–e651 (DOI: 10.1161/CIR.0000000000000617)",
            "epoch": "4th_edition_2018",
        }

    # 2. Enforce Sex-Specific 99th Percentile Upper Reference Limit (URL)
    norm_sex = str(patient_sex or "").strip().lower()
    if norm_sex in ["f", "female", "woman", "women", "female patient"]:
        applied_url = 9.0
        sex_group = "FEMALE"
    elif norm_sex in ["m", "male", "man", "men", "male patient"]:
        applied_url = 16.0
        sex_group = "MALE"
    else:
        applied_url = 14.0
        sex_group = "OVERALL_SEX_NEUTRAL"

    # 3. Evaluate Biomarker Kinetics
    parsed_troponins: List[float] = []
    if troponin_values:
        parsed_troponins = [float(v) for v in troponin_values]
    else:
        matches = re.findall(r"(?:troponin|hs-ctn|ctn[it])\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", text_corpus)
        if matches:
            parsed_troponins = [float(m) for m in matches]

    biomarker_eval = {
        "troponin_values_ng_l": parsed_troponins,
        "timestamps": troponin_timestamps or [],
        "reference_url_ng_l": applied_url,
        "patient_sex_group": sex_group,
        "status": "UNKNOWN_TROPONIN",
        "dynamic_delta_pct": None,
        "is_dynamic_rise_or_fall": False,
        "is_elevated_above_url": False,
    }

    if parsed_troponins:
        max_c = max(parsed_troponins)
        biomarker_eval["is_elevated_above_url"] = max_c > applied_url
        if len(parsed_troponins) >= 2:
            base_c = parsed_troponins[0]
            peak_c = parsed_troponins[1]
            delta_abs = abs(peak_c - base_c)
            denom = max(1.0, base_c)
            delta_pct = (delta_abs / denom) * 100.0
            biomarker_eval["dynamic_delta_pct"] = round(delta_pct, 1)

            if base_c > applied_url:
                biomarker_eval["is_dynamic_rise_or_fall"] = delta_pct >= 20.0
            else:
                biomarker_eval["is_dynamic_rise_or_fall"] = delta_pct >= 50.0 and peak_c > applied_url

            if biomarker_eval["is_elevated_above_url"] and biomarker_eval["is_dynamic_rise_or_fall"]:
                biomarker_eval["status"] = (
                    f"ACUTE_DYNAMIC_MYOCARDIAL_NECROSIS (Dynamic delta {delta_pct:.1f}% exceeds threshold with value > {applied_url} ng/L URL)"
                )
            elif biomarker_eval["is_elevated_above_url"]:
                biomarker_eval["status"] = (
                    f"CHRONIC_STABLE_MYOCARDIAL_INJURY (Non-dynamic elevation <20% [{delta_pct:.1f}%], typical in CKD/diabetic baseline)"
                )
            else:
                biomarker_eval["status"] = f"NORMAL_TROPONIN (All serial values <= {applied_url} ng/L URL)"
        else:
            if biomarker_eval["is_elevated_above_url"]:
                biomarker_eval["status"] = (
                    f"ELEVATED_SINGLE_VALUE ({max_c} ng/L > {applied_url} ng/L URL; serial sample required to confirm dynamic rise/fall)"
                )
            else:
                biomarker_eval["status"] = f"NORMAL_SINGLE_VALUE ({max_c} ng/L <= {applied_url} ng/L URL)"

    # 4. Evaluate ECG Findings (Negation-aware)
    def _is_negated_pattern(pat: str, text: str) -> bool:
        for m in re.finditer(pat, text):
            prefix = text[max(0, m.start() - 25):m.start()]
            if not any(neg in prefix for neg in ["no ", "nil ", "without ", "negative for ", "ruled out ", "denies "]):
                return False  # Found an affirmative (non-negated) hit
        return True  # All hits were negated

    ste_pat = r"(?:st(?:-segment)?\s*elevation|stemi|ste\s+in|lead [iv\d]+ ste|new lbbb)"
    st_elevation_detected = bool(re.search(ste_pat, text_corpus)) and not _is_negated_pattern(ste_pat, text_corpus)

    std_pat = r"(?:st(?:-segment)?\s*depression|t-wave inversion|twi|std\s+in|ischemic st changes)"
    st_depression_detected = bool(re.search(std_pat, text_corpus)) and not _is_negated_pattern(std_pat, text_corpus)

    has_ischemic_symptoms = any(k in text_corpus for k in [
        "chest pain", "angina", "retrosternal", "pressure", "tightness", "shortness of breath", "sob", "dyspnea", "diaphoresis"
    ]) and not _is_negated_pattern(r"(?:chest pain|angina|dyspnea|sob)", text_corpus)

    has_objective_ischemia = (
        st_elevation_detected
        or st_depression_detected
        or has_ischemic_symptoms
        or "wall motion abnormality" in text_corpus
        or "rwma" in text_corpus
        or "culprit" in text_corpus
        or "thrombus" in text_corpus
    )

    is_procedure_related = any(k in text_corpus for k in [
        "post-pci", "periprocedural", "post cabg", "stent thrombosis", "in-stent", "within 30 days of pci", "procedural mi", "type 4", "type 5"
    ])

    is_supply_demand_mismatch = any(k in text_corpus for k in [
        "sepsis", "septic shock", "af with rvr", "tachyarrhythmia", "severe anemia", "hb 5.", "hb 6.", "hypoxemia", "hypertensive emergency", "demand ischemia", "type 2"
    ])

    # 5. Adjudicate AMI / Myocardial Injury Subcategory
    adjudicated_ami = {
        "category": "UNCERTAIN_NON_ISCHEMIC",
        "guideline_classification": None,
        "legacy_fourth_definition_type": None,
        "concept_code": None,
        "vq_token": None,
        "icd10am_code": None,
        "snomed_sg_code": None,
        "clinical_rationale": "",
    }

    if st_elevation_detected:
        sub = capsule["universal_definition_criteria"]["subcategories"]["stemi"]
        adjudicated_ami.update({
            "category": "STEMI (ST-Elevation Myocardial Infarction)",
            "guideline_classification": "Primary Myocardial Infarction (STEMI)" if applied_version == "5th" else "Type 1 Spontaneous MI (STEMI)",
            "legacy_fourth_definition_type": "Type 1 Spontaneous MI",
            "concept_code": "CONCEPT_AMI_STEMI",
            "vq_token": sub["vq_token"],
            "icd10am_code": sub["icd10am"],
            "snomed_sg_code": sub["snomed_sg"],
            "clinical_rationale": "Persistent ST-segment elevation in contiguous leads or acute coronary occlusion. Mandates emergent Primary PCI pathway.",
        })
    elif is_procedure_related and (biomarker_eval["is_elevated_above_url"] or biomarker_eval["is_dynamic_rise_or_fall"]):
        adjudicated_ami.update({
            "category": "AMI_PROCEDURAL (Procedure-Related Myocardial Infarction)",
            "guideline_classification": "Procedure-Related Myocardial Infarction (<30 days post-revascularization)" if applied_version == "5th" else "Type 4a/4b/5 Procedural MI",
            "legacy_fourth_definition_type": "Type 4a (PCI) / Type 4b (Stent Thrombosis) / Type 5 (CABG)",
            "concept_code": "CONCEPT_AMI_PROCEDURAL",
            "vq_token": "VQ_03019",
            "icd10am_code": "I97.120",
            "snomed_sg_code": "233843008",
            "clinical_rationale": "Myocardial necrosis occurring as a direct complication of coronary revascularization (PCI or CABG).",
        })
    elif is_supply_demand_mismatch and (biomarker_eval["is_elevated_above_url"] or "nstemi" in text_corpus or "demand ischemia" in text_corpus):
        sub = capsule["universal_definition_criteria"]["subcategories"]["type_2_mi"]
        adjudicated_ami.update({
            "category": "AMI_SECONDARY (Secondary Myocardial Infarction / Supply-Demand Mismatch)",
            "guideline_classification": "Secondary Myocardial Infarction (Supply-Demand Mismatch)" if applied_version == "5th" else "Type 2 Supply-Demand Mismatch MI",
            "legacy_fourth_definition_type": "Type 2 Supply-Demand Mismatch MI",
            "concept_code": "CONCEPT_AMI_SECONDARY" if applied_version == "5th" else "CONCEPT_AMI_TYPE2_OTHER",
            "vq_token": "VQ_03018" if applied_version == "5th" else sub["vq_token"],
            "icd10am_code": sub["icd10am"],
            "snomed_sg_code": sub["snomed_sg"],
            "clinical_rationale": "Acute myocardial infarction provoked by non-coronary illness (e.g. sepsis, tachyarrhythmia, severe anemia, hypoxemia) causing severe supply-demand imbalance.",
        })
    elif (biomarker_eval["is_dynamic_rise_or_fall"] and biomarker_eval["is_elevated_above_url"]) or (biomarker_eval["is_elevated_above_url"] and (st_depression_detected or has_ischemic_symptoms or "nstemi" in text_corpus)):
        if has_objective_ischemia or "nstemi" in text_corpus:
            sub = capsule["universal_definition_criteria"]["subcategories"]["nstemi"]
            adjudicated_ami.update({
                "category": "NSTEMI (Non-ST Elevation Myocardial Infarction)",
                "guideline_classification": "Primary Myocardial Infarction (NSTEMI)" if applied_version == "5th" else "Type 1 Spontaneous MI (NSTEMI)",
                "legacy_fourth_definition_type": "Type 1 Spontaneous MI",
                "concept_code": "CONCEPT_AMI_NSTEMI",
                "vq_token": sub["vq_token"],
                "icd10am_code": sub["icd10am"],
                "snomed_sg_code": sub["snomed_sg"],
                "clinical_rationale": f"Symptoms/ECG evidence of ischemia with dynamic hs-cTn elevation > URL ({applied_url} ng/L for {sex_group}). Meets criteria for acute Primary NSTEMI.",
            })
        else:
            # Dynamic troponin without ischemic evidence = Acute Myocardial Injury under 5th definition
            adjudicated_ami.update({
                "category": "ACUTE_MYOCARDIAL_INJURY",
                "guideline_classification": "Acute Myocardial Injury" if applied_version == "5th" else "Acute Non-Ischemic Myocardial Injury",
                "legacy_fourth_definition_type": "Acute Non-Ischemic Myocardial Injury",
                "concept_code": "CONCEPT_MYOCARDIAL_INJURY_ACUTE",
                "vq_token": "VQ_03020",
                "icd10am_code": "I51.89",
                "snomed_sg_code": "131148009",
                "clinical_rationale": f"Dynamic hs-cTn rise/fall above URL ({applied_url} ng/L) in the absence of overt clinical or ECG ischemic features (Troponin elevation alone = Injury, not MI under 5th definition).",
            })
    elif "unstable angina" in text_corpus or "uap" in text_corpus or (has_ischemic_symptoms and not biomarker_eval["is_elevated_above_url"] and parsed_troponins):
        sub = capsule["universal_definition_criteria"]["subcategories"]["uap"]
        adjudicated_ami.update({
            "category": "ACS_UAP (Unstable Angina Pectoris)",
            "guideline_classification": "Unstable Angina Pectoris (Acute Coronary Syndrome without necrosis)",
            "legacy_fourth_definition_type": "Unstable Angina (No Biomarker Elevation)",
            "concept_code": sub["concept_code"],
            "vq_token": sub["vq_token"],
            "icd10am_code": sub["icd10am"],
            "snomed_sg_code": sub["snomed_sg"],
            "clinical_rationale": f"Ischemic symptoms at rest/crescendo with strictly normal cardiac troponin (<= {applied_url} ng/L URL), confirming absence of acute myocardial necrosis.",
        })
    elif biomarker_eval["is_elevated_above_url"] and not biomarker_eval["is_dynamic_rise_or_fall"]:
        adjudicated_ami.update({
            "category": "CHRONIC_MYOCARDIAL_INJURY",
            "guideline_classification": "Chronic Myocardial Injury" if applied_version == "5th" else "Chronic Baseline Myocardial Injury",
            "legacy_fourth_definition_type": "Chronic Non-Dynamic Myocardial Injury",
            "concept_code": "CONCEPT_MYOCARDIAL_INJURY_CHRONIC" if applied_version == "5th" else "CONCEPT_CKD_OR_CARDIAC_INJURY",
            "vq_token": "VQ_03021" if applied_version == "5th" else "VQ_03011",
            "icd10am_code": "I51.89",
            "snomed_sg_code": "131148009",
            "clinical_rationale": f"Stable elevated hs-cTn (<20% delta) exceeding {applied_url} ng/L, indicative of chronic myocardial strain/injury (common in CKD Stage 3-5/diabetic patients) rather than acute infarction.",
        })

    # 6. Adjudicate PCI Staging
    is_staged = bool(re.search(r"(?:staged pci|plan for staged|for staged|staged to|staged procedure|deferred to stage 2|2nd stage pci|deferred non-culprit)", text_corpus))
    is_primary = bool(re.search(r"(?:primary pci|emergent pci|urgent pci|p-pci|ppci|door-to-balloon|rescue pci)", text_corpus)) or (st_elevation_detected and "pci" in text_corpus)

    pci_adjudication = {
        "staging_classification": "NONE",
        "concept_code": None,
        "vq_token": None,
        "moh_tosp_code": None,
        "achi_code": None,
        "billing_and_quality_notes": "",
    }

    if is_staged:
        rule = capsule["pci_staging_principles"]["staged_deferred_pci"]
        pci_adjudication.update({
            "staging_classification": "STAGED_DEFERRED_PCI",
            "concept_code": rule["concept_code"],
            "vq_token": rule["vq_token"],
            "moh_tosp_code": rule["moh_tosp"],
            "achi_code": rule["achi"],
            "billing_and_quality_notes": "Planned / deferred intervention on non-culprit lesion. Billed as a distinct operative episode or elective outpatient admission. Door-to-balloon quality metric does not apply.",
        })
    elif is_primary or "pci to" in text_corpus or "stent" in text_corpus:
        rule = capsule["pci_staging_principles"]["primary_emergent_pci"]
        pci_adjudication.update({
            "staging_classification": "PRIMARY_EMERGENT_PCI" if is_primary else "INDEX_PERFORMED_PCI",
            "concept_code": rule["concept_code"] if is_primary else "CONCEPT_PCI_STENT",
            "vq_token": rule["vq_token"] if is_primary else "VQ_07112",
            "moh_tosp_code": rule["moh_tosp"],
            "achi_code": rule["achi"],
            "billing_and_quality_notes": "Emergent/index revascularization of acute culprit lesion. Subject to hospital Door-to-Balloon (<=90 min) benchmarking.",
        })

    return {
        "router": "sg-clinical-router-mcp",
        "tool": "evaluate_ami_pci_stratification",
        "authority": capsule["authority"],
        "guideline_framework_applied": guideline_info,
        "temporal_parameters": {
            "visit_date_evaluated": visit_date,
            "patient_sex": norm_sex or "unspecified",
            "sex_group_evaluated": sex_group,
            "applied_99th_percentile_url_ng_l": applied_url,
        },
        "biomarker_kinetics_evaluation": biomarker_eval,
        "ecg_evaluation": {
            "st_elevation_detected": st_elevation_detected,
            "st_depression_ischemia_detected": st_depression_detected,
        },
        "adjudicated_ami_category": adjudicated_ami,
        "adjudicated_pci_staging": pci_adjudication,
    }


def handle_jsonrpc(req: Dict[str, Any]) -> Dict[str, Any]:
    method = req.get("method", "")
    req_id = req.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sg-clinical-router-mcp", "version": "1.2.0"},
            },
        }
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "expand_clinical_context",
                        "description": (
                            "Route a clinical concept or benchmark failure across Singapore ACE/MOH/TOSP guidelines, "
                            "HSA Therapeutic Products Register (data.gov.sg), HSA SMDR Medical Devices, PharmGKB, PubMed, and ClinicalTrials.gov."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "enum": [
                                        "cardio_metabolic",
                                        "surgical_tosp",
                                        "pgx_genomics",
                                        "shi_privacy",
                                        "hsa_formulary",
                                        "cardio_universal_ami_pci",
                                    ],
                                },
                                "clinical_query": {"type": "string"},
                                "include_live_pubmed": {"type": "boolean"},
                                "query_hsa_registry": {"type": "boolean"},
                            },
                            "required": ["domain", "clinical_query"],
                        },
                    },
                    {
                        "name": "evaluate_ami_pci_stratification",
                        "description": (
                            "Adjudicate Acute Myocardial Infarction subcategories (STEMI, NSTEMI, UAP, Secondary MI / Type 2, "
                            "Procedure-Related MI, or Acute/Chronic Myocardial Injury) and PCI Staging (Primary Emergent PCI vs Staged Deferred PCI) "
                            "based on temporal cardiac troponin biomarker kinetics, sex-specific 99th percentile URLs, "
                            "and dynamic guideline versioning (Fifth Universal Definition August 2026 vs Fourth Universal Definition 2018)."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "clinical_scenario": {
                                    "type": "string",
                                    "description": "Patient presentation, symptoms (chest pain, shortness of breath), timing",
                                },
                                "ecg_findings": {
                                    "type": "string",
                                    "description": "12-lead ECG findings (e.g. 'ST elevation in V1-V4', 'ST depression in II, III, aVF')",
                                },
                                "troponin_values": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "Serial cardiac troponin values in ng/L (e.g. [12.0, 85.0] or [24.0, 25.0])",
                                },
                                "troponin_timestamps": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Timestamps of troponin draws (e.g. ['0h', '3h'])",
                                },
                                "procedure_notes": {
                                    "type": "string",
                                    "description": "Catheterization report, culprit lesion, planned non-culprit staged interventions",
                                },
                                "visit_date": {
                                    "type": "string",
                                    "description": "Visit or admission date (YYYY-MM-DD). If >= 2026-08-01, applies 5th Universal Definition (2026); if earlier, applies 4th Universal Definition (2018).",
                                },
                                "patient_sex": {
                                    "type": "string",
                                    "description": "Biological sex ('female' / 'male') to apply mandated sex-specific 99th percentile URL (Female 9 ng/L, Male 16 ng/L).",
                                },
                                "guideline_version": {
                                    "type": "string",
                                    "enum": ["auto", "5th", "4th"],
                                    "description": "Guideline version override ('auto', '5th' for 2026 edition, '4th' for 2018 edition). Default: 'auto'.",
                                },
                            },
                            "required": ["clinical_scenario"],
                        },
                    },
                    {
                        "name": "query_hsa_therapeutic_products",
                        "description": (
                            "Search the official Singapore Health Sciences Authority (HSA) Register of Therapeutic Products "
                            "on data.gov.sg (Resource ID: d_767279312753558cbf19d48344577084) for product brand names, "
                            "active ingredients, forensic classification (POM/P/GSL), dosage forms, and licence numbers (SINXXXXXP)."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "drug_query": {"type": "string", "description": "Brand name, active ingredient, or SIN licence number"},
                                "max_results": {"type": "integer", "default": 5},
                            },
                            "required": ["drug_query"],
                        },
                    },
                    {
                        "name": "query_hsa_medical_devices",
                        "description": (
                            "Search the official Singapore Health Sciences Authority (HSA) Medical Device Register (SMDR) "
                            "on data.gov.sg (Resource ID: d_8cfe111e0a5a5cf5b598e78851e58ad4) for Class B/C/D medical devices "
                            "(e.g., coronary DES stents, IOL implants, pacemakers, surgical trocars/clips) matching MOH TOSP surgical tables."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "device_query": {"type": "string", "description": "Device name, category, or intended purpose"},
                                "max_results": {"type": "integer", "default": 5},
                            },
                            "required": ["device_query"],
                        },
                    },
                ]
            },
        }
    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "query_hsa_therapeutic_products":
            res = query_hsa_registered_drugs_live(
                drug_query=args.get("drug_query", ""),
                max_results=int(args.get("max_results", 5)),
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
            }

        if tool_name == "query_hsa_medical_devices":
            res = query_hsa_medical_devices_live(
                device_query=args.get("device_query", ""),
                max_results=int(args.get("max_results", 5)),
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
            }

        if tool_name == "evaluate_ami_pci_stratification":
            res = evaluate_ami_pci_stratification(
                clinical_scenario=args.get("clinical_scenario", ""),
                ecg_findings=args.get("ecg_findings", ""),
                troponin_values=args.get("troponin_values"),
                troponin_timestamps=args.get("troponin_timestamps"),
                procedure_notes=args.get("procedure_notes", ""),
                visit_date=args.get("visit_date"),
                patient_sex=args.get("patient_sex"),
                guideline_version=args.get("guideline_version", "auto"),
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
            }

        res = expand_clinical_context(
            domain=args.get("domain", "cardio_metabolic"),
            clinical_query=args.get("clinical_query", ""),
            include_live_pubmed=bool(args.get("include_live_pubmed", False)),
            query_hsa_registry=bool(args.get("query_hsa_registry", True)),
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
        }
    return {"jsonrpc": "2.0", "id": req_id, "result": {}}


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_jsonrpc(request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(
                json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(exc)}}) + "\n"
            )
            sys.stdout.flush()
