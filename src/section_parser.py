"""
Multi-Pass Clinical Section Parser for SG-MEDIATOR (`@cardio-section-sieve-architect`).

Handles:
  * Executive-summary catheterization/discharge templates where CONCLUSION or IMPRESSION/PLAN
    appears on Page 1 above PROCEDURAL DETAILS and HEMODYNAMICS.
  * Multi-delimiter section headers:
      - Newline-delimited (`CONCLUSION\n1. Two-vessel CAD...`)
      - Inline colon/dash (`Conclusion: Successful PCI to RCA; staged PCI to LAD.`)
      - Numbered or Roman-numeral (`IV. CONCLUSION AND RECOMMENDATIONS:`)
      - Pipe-table / boxed (`| CONCLUSION | Plan for staged PCI to LCx |`)
  * Automatic merging of adjacent CONCLUSION/IMPRESSION + PLAN/RECOMMENDATIONS blocks so
    staged-PCI timing and vessel instructions in `Plan:` are never severed.
  * Tail-40% synthetic window (`CONCLUSION_TAIL_FALLBACK`) when a report contains no explicit headers.
"""
import re
from typing import Dict, List, Optional, Tuple

HEADER_BUCKETS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "CONCLUSION_PLAN",
        (
            "conclusion and recommendations",
            "conclusion & recommendations",
            "conclusions and plan",
            "conclusion and plan",
            "impression and plan",
            "impression / plan",
            "assessment and plan",
            "conclusions",
            "conclusion",
            "impression",
            "final diagnosis",
            "discharge diagnosis",
            "summary",
        ),
    ),
    (
        "PLAN_FOLLOWUP",
        (
            "recommendations",
            "recommendation",
            "management plan",
            "discharge plan",
            "follow-up plan",
            "plan",
        ),
    ),
    (
        "BACKGROUND_HISTORY",
        (
            "past medical history",
            "background history",
            "clinical history",
            "medical history",
            "indication",
            "indications",
            "bghx",
            "pmhx",
            "history",
        ),
    ),
    (
        "PROCEDURE_FINDINGS",
        (
            "procedural details",
            "procedure details",
            "coronary angiography",
            "angiographic findings",
            "operative findings",
            "hemodynamics",
            "haemodynamics",
            "findings",
            "procedure",
        ),
    ),
    (
        "MEDICATIONS",
        (
            "discharge medications",
            "current medications",
            "medications",
            "drugs",
        ),
    ),
    (
        "INVESTIGATIONS",
        (
            "laboratory results",
            "investigations",
            "labs",
        ),
    ),
]

PIPE_ROW_RE = re.compile(r"^\s*\|\s*([^|]{2,45}?)\s*\|\s*(.*?)\s*\|?\s*$")
ROMAN_OR_NUM_PREFIX_RE = re.compile(r"^\s*(?:[IVXLC]{1,6}|\d{1,2})[\.\)]\s*", re.IGNORECASE)


def _match_header_label(candidate: str) -> Optional[str]:
    cleaned = ROMAN_OR_NUM_PREFIX_RE.sub("", candidate.strip()).strip(" :-\t*#").lower()
    if not cleaned or len(cleaned) > 48:
        return None
    for bucket, aliases in HEADER_BUCKETS:
        for alias in aliases:
            if cleaned == alias:
                return bucket
    return None


def _parse_line_for_header(line: str) -> Tuple[Optional[str], str]:
    stripped = line.strip()
    if not stripped:
        return None, ""

    # 1. Pipe-table row: | CONCLUSION | Plan for staged PCI to LCx |
    m_pipe = PIPE_ROW_RE.match(stripped)
    if m_pipe:
        col1, col2 = m_pipe.group(1), m_pipe.group(2)
        bucket = _match_header_label(col1)
        if bucket:
            return bucket, col2.strip(" |")

    # 2. Standalone header line (with or without trailing colon/dash)
    bucket_whole = _match_header_label(stripped)
    if bucket_whole:
        return bucket_whole, ""

    # 3. Inline colon or spaced-hyphen header: `IV. Conclusion - Staged PCI...` or `Conclusion: ...`
    for sep_re in (r"^([^:]{2,52}):\s+(.+)$", r"^([^-]{2,52})\s+-\s+(.+)$"):
        m_in = re.match(sep_re, stripped)
        if m_in:
            bucket = _match_header_label(m_in.group(1))
            if bucket:
                return bucket, m_in.group(2).strip()

    return None, stripped


def parse_clinical_sections(raw_text: str) -> List[Dict[str, str]]:
    """
    Segments clinical free text into ordered `{section_type, text}` blocks, merging
    adjacent CONCLUSION_PLAN + PLAN_FOLLOWUP blocks and emitting a 40% tail fallback
    when no explicit headers exist in a multi-line document.
    """
    if not raw_text or not raw_text.strip():
        return []

    raw_blocks: List[Dict[str, str]] = []
    current_type = "GENERAL"
    current_lines: List[str] = []
    explicit_header_seen = False

    def _flush():
        nonlocal current_lines
        body = "\n".join(x for x in current_lines if x).strip()
        if body:
            raw_blocks.append({"section_type": current_type, "text": body})
        current_lines = []

    for line in raw_text.splitlines():
        hdr_bucket, remainder = _parse_line_for_header(line)
        if hdr_bucket is not None:
            explicit_header_seen = True
            _flush()
            current_type = hdr_bucket
            if remainder:
                current_lines.append(remainder)
        else:
            if remainder:
                current_lines.append(remainder)
    _flush()

    # Merge CONCLUSION_PLAN immediately followed by PLAN_FOLLOWUP (or standalone PLAN_FOLLOWUP -> CONCLUSION_PLAN)
    merged: List[Dict[str, str]] = []
    for blk in raw_blocks:
        btype = blk["section_type"]
        if btype == "PLAN_FOLLOWUP":
            if merged and merged[-1]["section_type"] == "CONCLUSION_PLAN":
                merged[-1]["text"] = merged[-1]["text"] + "\n" + blk["text"]
                continue
            blk = {"section_type": "CONCLUSION_PLAN", "text": blk["text"]}
        elif btype == "CONCLUSION_PLAN" and merged and merged[-1]["section_type"] == "CONCLUSION_PLAN":
            merged[-1]["text"] = merged[-1]["text"] + "\n" + blk["text"]
            continue
        merged.append(blk)

    # Synthetic tail-40% fallback when no explicit headers were detected on a multi-line note
    if not explicit_header_seen and len(raw_text.strip()) >= 120:
        clean_full = raw_text.strip()
        cutoff = int(len(clean_full) * 0.60)
        nl_pos = clean_full.find("\n", max(0, cutoff - 40))
        tail_start = nl_pos + 1 if 0 < nl_pos < len(clean_full) - 10 else cutoff
        tail_text = clean_full[tail_start:].strip()
        if tail_text and tail_text != clean_full:
            merged = [
                {"section_type": "GENERAL", "text": clean_full[:tail_start].strip()},
                {"section_type": "CONCLUSION_TAIL_FALLBACK", "text": tail_text},
            ]
        elif tail_text:
            merged = [{"section_type": "CONCLUSION_TAIL_FALLBACK", "text": tail_text}]

    return merged


def extract_conclusion_window(raw_text: str) -> str:
    """Returns concatenated CONCLUSION_PLAN (or CONCLUSION_TAIL_FALLBACK) blocks, or full text for short notes."""
    sections = parse_clinical_sections(raw_text)
    target = [
        s["text"] for s in sections if s["section_type"] in ("CONCLUSION_PLAN", "CONCLUSION_TAIL_FALLBACK")
    ]
    if target:
        return "\n".join(target)
    return raw_text.strip()
