"""
6-Tier Cascading Clinical NLP Sieve & Multi-Source Derived Risk-Factor Engine.
Extracts explicit clinical concepts, ConText assertions (AFFIRMED, NEGATED, HISTORICAL, CONDITIONAL),
Singapore SDOH / Nutri-Grade / Healthier SG signals, genomic/PGx variants, Agatston CAC scores,
and triangulates implicit chronic conditions from SI-unit labs and prior/active medications.
"""
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

from hybrid_semantic_layer import HybridSemanticLayer

AGATSTON_TEXT_RE = re.compile(
    r"(?:agatston|coronary\s+calcium|cac)\s*(?:score)?\s*(?:of|=|:)?\s*(\d{1,5}(?:\.\d+)?)",
    re.IGNORECASE,
)
HBA1C_TEXT_RE = re.compile(r"hba1c\s*(?:of|=|:)?\s*(\d{1,2}\.\d+)\s*%", re.IGNORECASE)
LDL_TEXT_RE = re.compile(r"ldl(?:-c)?\s*(?:of|=|:)?\s*(\d{1,2}\.\d+)\s*mmol/l", re.IGNORECASE)
EGFR_TEXT_RE = re.compile(r"egfr\s*(?:of|=|:)?\s*(\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
BMI_TEXT_RE = re.compile(r"bmi\s*(?:of|=|:)?\s*(\d{2}(?:\.\d+)?)\s*(?:kg/m2)?", re.IGNORECASE)


class ClinicalSieveAndRiskEngine:
    def __init__(self, rules_yaml_path: Path, semantic_layer: HybridSemanticLayer, arbitrator: Optional[Any] = None):
        with rules_yaml_path.open("r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f)
        self.semantic = semantic_layer
        # Precompile synonym and SDOH boundary patterns (re's internal 512-entry cache thrashes at 200+ patterns)
        self._syn_re: Dict[str, Any] = {
            syn: re.compile(rf"(?<![a-z0-9]){re.escape(syn)}(?![a-z0-9])")
            for syn in self.semantic.synonym_to_code
        }
        self._sdoh_compiled: List[Tuple[str, Any, str]] = [
            (pat.lower(), re.compile(rf"(?<![a-z0-9]){re.escape(pat.lower())}(?![a-z0-9])"), group_cfg["code"])
            for group_cfg in self.rules.get("singapore_policy_and_sdoh_patterns", {}).values()
            for pat in group_cfg.get("patterns", [])
        ]
        # Optional Tier-5 local LLM arbitrator (16 GB profile only); None => purely deterministic
        self.arbitrator = arbitrator
        self.hedge_cues = self.rules.get("tier5_arbitration", {}).get("hedge_cues", [])

    def _detect_assertion_status(
        self, sentence_lower: str, match_start: int, temporal_status: str, concept_hint: str = ""
    ) -> str:
        window_start = max(0, match_start - 55)
        prefix_window = sentence_lower[window_start:match_start]

        # Tier 5: only hedged mentions are sent to the local LLM; failure falls through to rules
        if self.arbitrator is not None and getattr(self.arbitrator, "backend", None) and concept_hint:
            if any(h in prefix_window for h in self.hedge_cues):
                label = self.arbitrator.arbitrate(sentence_lower, concept_hint)
                if label in ("AFFIRMED", "NEGATED", "HISTORICAL", "CONDITIONAL"):
                    return label
                if label == "UNCERTAIN":
                    return "CONDITIONAL"

        for neg in self.rules["negation_and_context_triggers"]["negated_prefixes"]:
            if neg in prefix_window:
                return "NEGATED"
        for cond in self.rules["negation_and_context_triggers"]["conditional_prefixes"]:
            if cond in prefix_window:
                return "CONDITIONAL"
        for hist in self.rules["negation_and_context_triggers"]["historical_prefixes"]:
            if hist in prefix_window:
                return "HISTORICAL"
        if temporal_status in ("HISTORICAL_CARRIED_FORWARD", "HISTORICAL_MENTION"):
            return "HISTORICAL"
        return "AFFIRMED"

    def extract_from_visit(
        self,
        dedup_result: Dict[str, Any],
        structured_row: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs the 6-tier sieve across deduplicated text segments and combines with structured
        SI-unit lab columns and medication lists to derive explicit + implicit risk factors.
        """
        structured_row = structured_row or {}
        extracted: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _add_concept(
            code: str,
            source: str,
            assertion: str,
            provenance: str,
            span: str,
            extra: Optional[Dict[str, Any]] = None,
        ):
            key = (code, assertion)
            # Prefer NEW_AT_VISIT over CARRIED_FORWARD if seen in both
            if key in extracted and extracted[key]["provenance"] == "NEW_AT_VISIT":
                return
            extracted[key] = self.semantic.resolve_concept(
                concept_code=code,
                evidence_source=source,
                assertion_status=assertion,
                provenance=provenance,
                supporting_span=span,
                extra_metadata=extra,
            )

        # ------------------------------------------------------------------
        # TIER 1 & 2: Lexical Synonym Sieve + ConText Assertion over Segments
        # ------------------------------------------------------------------
        for seg_info in dedup_result.get("segments", []):
            seg_text = seg_info["text"]
            seg_lower = seg_text.lower()
            prov = seg_info["provenance"]
            t_status = seg_info["temporal_status"]

            # 1A. Match ontology synonyms (fast C-level substring gate before boundary regex)
            for syn, c_code in self.semantic.synonym_to_code.items():
                if syn not in seg_lower:
                    continue
                pat = self._syn_re.get(syn)
                if pat is None:
                    pat = self._syn_re[syn] = re.compile(rf"(?<![a-z0-9]){re.escape(syn)}(?![a-z0-9])")
                m_syn = pat.search(seg_lower)
                if m_syn:
                    idx = m_syn.start()
                    assertion = self._detect_assertion_status(seg_lower, idx, t_status, concept_hint=syn)
                    _add_concept(c_code, "TEXT_LEXICAL_SIEVE", assertion, prov, seg_text[:140])

            # 1B. Match Medication-Derived Risk Factors from text
            for med_name, target_codes in self.rules.get("medication_to_risk_inference", {}).items():
                idx = seg_lower.find(med_name)
                if idx != -1:
                    assertion = self._detect_assertion_status(seg_lower, idx, t_status)
                    if assertion != "NEGATED":
                        for tc in target_codes:
                            _add_concept(
                                tc,
                                f"MEDICATION_DERIVED({med_name})",
                                "AFFIRMED",
                                prov,
                                f"Derived from drug '{med_name}' in: {seg_text[:100]}",
                            )

            # 1C. Match Singapore Policy & SDOH patterns (Nutri-Grade, Healthier SG, CHAS, HDB, FDW)
            for pat_lower, pat_re, s_code in self._sdoh_compiled:
                if pat_lower not in seg_lower:
                    continue
                m_pat = pat_re.search(seg_lower)
                if m_pat:
                    idx = m_pat.start()
                    assertion = self._detect_assertion_status(seg_lower, idx, t_status)
                    _add_concept(s_code, "SG_POLICY_SDOH_SIEVE", assertion, prov, seg_text[:140])

            # 1D. Match Genomics & Pharmacogenomics (HLA-B*15:02, HLA-B*58:01, CYP2C19, EGFR)
            for variant_pat, var_cfg in self.rules.get("genomic_and_pgx_rules", {}).items():
                idx = seg_lower.find(variant_pat)
                if idx != -1:
                    assertion = self._detect_assertion_status(seg_lower, idx, t_status)
                    _add_concept(
                        var_cfg["code"],
                        "GENOMIC_PGX_SIEVE",
                        assertion,
                        prov,
                        seg_text[:140],
                        extra={"pgx_actionability": var_cfg["actionability"]},
                    )

            # 1E. Match Sensitive Health Information (SHI)
            for _shi_cat, shi_cfg in self.rules.get("sensitive_health_information_shi", {}).get("categories", {}).items():
                for pat in shi_cfg.get("patterns", []):
                    idx = seg_lower.find(pat)
                    if idx != -1:
                        assertion = self._detect_assertion_status(seg_lower, idx, t_status)
                        if assertion != "NEGATED":
                            _add_concept(shi_cfg["code"], "SHI_DETECTOR", assertion, prov, seg_text[:140])

            # 1F. Inline Numeric Biomarkers in Free Text (Agatston, HbA1c, LDL, eGFR, Asian BMI)
            m_cac = AGATSTON_TEXT_RE.search(seg_text)
            if m_cac:
                cac_val = float(m_cac.group(1))
                if cac_val > 0:
                    tier = "SEVERE (>=300)" if cac_val >= 300 else ("MODERATE (100-299)" if cac_val >= 100 else "MILD (1-99)")
                    _add_concept(
                        "CONCEPT_CAD_CALCIFICATION",
                        "TEXT_AGATSTON_EXTRACTOR",
                        "AFFIRMED",
                        prov,
                        f"Agatston CAC={cac_val} ({tier})",
                        extra={"agatston_score": cac_val, "agatston_risk_tier": tier},
                    )

            m_a1c = HBA1C_TEXT_RE.search(seg_text)
            if m_a1c and float(m_a1c.group(1)) >= self.rules["si_lab_derivation_rules"]["t2dm_glycemic"]["hba1c_pct_min"]:
                _add_concept(
                    "CONCEPT_DM_T2",
                    "LAB_DERIVED_INLINE_TEXT",
                    "AFFIRMED",
                    prov,
                    f"Inline HbA1c {m_a1c.group(1)}% >= 6.5%",
                )

            m_ldl = LDL_TEXT_RE.search(seg_text)
            if m_ldl and float(m_ldl.group(1)) >= self.rules["si_lab_derivation_rules"]["dyslipidemia_ascvd"]["ldl_mmol_l_high_risk_min"]:
                _add_concept(
                    "CONCEPT_HLD",
                    "LAB_DERIVED_INLINE_TEXT",
                    "AFFIRMED",
                    prov,
                    f"Inline LDL-C {m_ldl.group(1)} mmol/L elevated",
                )

            m_egfr = EGFR_TEXT_RE.search(seg_text)
            if m_egfr and float(m_egfr.group(1)) < self.rules["si_lab_derivation_rules"]["chronic_kidney_disease"]["egfr_ml_min_max"]:
                _add_concept(
                    "CONCEPT_CKD",
                    "LAB_DERIVED_INLINE_TEXT",
                    "AFFIRMED",
                    prov,
                    f"Inline eGFR {m_egfr.group(1)} < 60 mL/min/1.73m2",
                )

            m_bmi = BMI_TEXT_RE.search(seg_text)
            if m_bmi and float(m_bmi.group(1)) >= self.rules["si_lab_derivation_rules"]["asian_bmi_thresholds"]["overweight_min"]:
                bmi_val = float(m_bmi.group(1))
                cat = "Asian Obese (>=27.5)" if bmi_val >= 27.5 else "Asian Overweight (>=23.0)"
                _add_concept(
                    "CONCEPT_OBESITY_ASIAN",
                    "ANTHROPOMETRIC_ASIAN_BMI",
                    "AFFIRMED",
                    prov,
                    f"BMI {bmi_val} kg/m2 ({cat})",
                )

        # ------------------------------------------------------------------
        # TIER 3: Structured Table Columns (SI Lab Results & Medication Cell)
        # ------------------------------------------------------------------
        def _safe_float(val: Any) -> Optional[float]:
            if val is None or val == "":
                return None
            try:
                return float(str(val).strip())
            except ValueError:
                return None

        hba1c_pct = _safe_float(structured_row.get("hba1c_pct"))
        if hba1c_pct is not None and hba1c_pct >= 6.5:
            _add_concept("CONCEPT_DM_T2", "STRUCTURED_LAB_SI", "AFFIRMED", "NEW_AT_VISIT", f"Structured HbA1c={hba1c_pct}%")

        fbg_mmol = _safe_float(structured_row.get("fasting_glucose_mmol_l"))
        if fbg_mmol is not None and fbg_mmol >= 7.0:
            _add_concept("CONCEPT_DM_T2", "STRUCTURED_LAB_SI", "AFFIRMED", "NEW_AT_VISIT", f"Structured FBG={fbg_mmol} mmol/L")

        ldl_mmol = _safe_float(structured_row.get("ldl_mmol_l"))
        if ldl_mmol is not None and ldl_mmol >= 2.6:
            _add_concept("CONCEPT_HLD", "STRUCTURED_LAB_SI", "AFFIRMED", "NEW_AT_VISIT", f"Structured LDL-C={ldl_mmol} mmol/L")

        egfr_val = _safe_float(structured_row.get("egfr_ml_min"))
        if egfr_val is not None and egfr_val < 60.0:
            _add_concept("CONCEPT_CKD", "STRUCTURED_LAB_SI", "AFFIRMED", "NEW_AT_VISIT", f"Structured eGFR={egfr_val} mL/min/1.73m2")

        uacr_val = _safe_float(structured_row.get("uacr_mg_mmol"))
        if uacr_val is not None and uacr_val >= 3.0:
            _add_concept("CONCEPT_CKD", "STRUCTURED_LAB_SI", "AFFIRMED", "NEW_AT_VISIT", f"Structured uACR={uacr_val} mg/mmol")

        sbp_val = _safe_float(structured_row.get("sbp_mmhg"))
        dbp_val = _safe_float(structured_row.get("dbp_mmhg"))
        if (sbp_val is not None and sbp_val >= 140) or (dbp_val is not None and dbp_val >= 90):
            _add_concept("CONCEPT_HTN", "STRUCTURED_VITALS", "AFFIRMED", "NEW_AT_VISIT", f"Structured BP={sbp_val}/{dbp_val} mmHg")

        bmi_struct = _safe_float(structured_row.get("bmi_kg_m2"))
        if bmi_struct is not None and bmi_struct >= 23.0:
            _add_concept("CONCEPT_OBESITY_ASIAN", "STRUCTURED_VITALS_ASIAN_BMI", "AFFIRMED", "NEW_AT_VISIT", f"Structured BMI={bmi_struct} kg/m2")

        cac_struct = _safe_float(structured_row.get("agatston_score"))
        if cac_struct is not None and cac_struct > 0:
            _add_concept("CONCEPT_CAD_CALCIFICATION", "STRUCTURED_IMAGING", "AFFIRMED", "NEW_AT_VISIT", f"Structured Agatston={cac_struct}")

        med_col = str(structured_row.get("medications", "")).lower()
        if med_col:
            for med_name, target_codes in self.rules.get("medication_to_risk_inference", {}).items():
                if med_name in med_col:
                    for tc in target_codes:
                        _add_concept(tc, f"STRUCTURED_MEDICATION({med_name})", "AFFIRMED", "NEW_AT_VISIT", f"Medication list: {med_name}")

        return list(extracted.values())
