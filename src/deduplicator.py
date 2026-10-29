"""
Longitudinal Copy-Paste Deduplicator & Temporal Anchor Resolver for Singapore EHR Notes.
Addresses note bloat across Polyclinics, SOCs, ED, and Inpatient Wards (NGEMR / Sunrise SCM)
using 64-bit Rabin-Karp / SimHash shingle fingerprinting.
"""
import hashlib
import re
from typing import Dict, List, Tuple, Any

WORD_RE = re.compile(r"[a-z0-9/\*\.]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?;\n])\s+")
RELATIVE_TIME_RE = re.compile(
    r"\b(\d+/12\s+ago|\d+/52\s+ago|last\s+yr|last\s+year|in\s+20\d{2}|s/p|hx\s+of|history\s+of)\b",
    re.IGNORECASE,
)


def _compute_shingle_hashes(text: str, shingle_size: int = 8) -> set:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return set()
    if len(tokens) < shingle_size:
        shingle = " ".join(tokens)
        h = int(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).hexdigest(), 16)
        return {h}
    hashes = set()
    for i in range(len(tokens) - shingle_size + 1):
        shingle = " ".join(tokens[i : i + shingle_size])
        h = int(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).hexdigest(), 16)
        hashes.add(h)
    return hashes


def classify_visit_Care_setting(text: str, visit_type_hint: str = "") -> str:
    combined = f"{visit_type_hint} {text}".lower()
    if any(k in combined for k in ["polyclinic", "healthier sg", "nup", "nhgp", "shP", "chas"]):
        return "POLYCLINIC_HEALTHIER_SG"
    if any(k in combined for k in ["soc", "specialist outpatient", "clinic review", "tcu"]):
        return "SOC_OUTPATIENT"
    if any(k in combined for k in ["day surgery", "endoscopy suite", "ambulatory surgery"]):
        return "DAY_SURGERY_ENDOSCOPY"
    if any(k in combined for k in ["emergency department", "a&e", "triage p1", "triage p2"]):
        return "ED_ATTENDANCE"
    if any(k in combined for k in ["icu", "micu", "sicu", "high dependency", "intubated"]):
        return "ICU_HD"
    return "GENERAL_WARD_INPATIENT"


class LongitudinalDeduplicator:
    def __init__(self, shingle_size: int = 8, overlap_threshold: float = 0.85):
        self.shingle_size = shingle_size
        self.overlap_threshold = overlap_threshold
        # patient_id -> list of (visit_no, visit_date, sentence_text, shingle_hash_set)
        self.patient_history_index: Dict[str, List[Tuple[str, str, str, set]]] = {}

    def process_patient_visit(
        self,
        patient_id: str,
        visit_no: str,
        visit_date: str,
        raw_text: str,
        visit_type_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Splits a clinical note into segments, detects copy-pasted spans carried forward
        from earlier visits for the same patient, and resolves temporal anchors.
        """
        if not raw_text:
            return {
                "care_setting": classify_visit_Care_setting("", visit_type_hint),
                "segments": [],
                "novel_text": "",
                "carried_forward_text": "",
                "copy_paste_ratio": 0.0,
            }

        prior_entries = self.patient_history_index.setdefault(patient_id, [])
        raw_segments = [s.strip() for s in SENTENCE_SPLIT_RE.split(raw_text) if s.strip()]

        processed_segments = []
        novel_parts = []
        carried_parts = []

        for seg in raw_segments:
            seg_hashes = _compute_shingle_hashes(seg, self.shingle_size)
            provenance = "NEW_AT_VISIT"
            origin_visit_no = visit_no
            origin_visit_date = visit_date
            max_overlap = 0.0

            if seg_hashes and prior_entries:
                for p_visit_no, p_visit_date, _p_seg, p_hashes in prior_entries:
                    if not p_hashes:
                        continue
                    overlap = len(seg_hashes.intersection(p_hashes)) / float(len(seg_hashes))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        if overlap >= self.overlap_threshold:
                            provenance = "CARRIED_FORWARD"
                            origin_visit_no = p_visit_no
                            origin_visit_date = p_visit_date
                            break

            has_historical_marker = bool(RELATIVE_TIME_RE.search(seg))
            temporal_status = (
                "HISTORICAL_CARRIED_FORWARD"
                if provenance == "CARRIED_FORWARD"
                else ("HISTORICAL_MENTION" if has_historical_marker else "ACTIVE_INDEX_VISIT")
            )

            processed_segments.append(
                {
                    "text": seg,
                    "provenance": provenance,
                    "origin_visit_no": origin_visit_no,
                    "origin_visit_date": origin_visit_date,
                    "overlap_score": round(max_overlap, 3),
                    "temporal_status": temporal_status,
                }
            )

            if provenance == "CARRIED_FORWARD":
                carried_parts.append(seg)
            else:
                novel_parts.append(seg)
                # Index novel segments for future visit comparisons
                prior_entries.append((visit_no, visit_date, seg, seg_hashes))

        total_chars = max(1, sum(len(s["text"]) for s in processed_segments))
        carried_chars = sum(len(s) for s in carried_parts)

        return {
            "care_setting": classify_visit_Care_setting(raw_text, visit_type_hint),
            "segments": processed_segments,
            "novel_text": " ".join(novel_parts),
            "carried_forward_text": " ".join(carried_parts),
            "copy_paste_ratio": round(carried_chars / float(total_chars), 3),
        }
