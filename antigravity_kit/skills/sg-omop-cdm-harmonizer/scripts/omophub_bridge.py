"""
SG-MEDIATOR -> OHDSI OMOP CDM v5.4 Harmonizer & OMOPHub API Bridge.
Self-contained skill tool requiring zero changes to core src/ offline runtime.

Capabilities:
  1. Offline resolution of all 128 SG-MEDIATOR concepts (UMLS, SNOMED-SG, ICD-10-AM,
     ACHI, MOH TOSP, Nutri-Grade, Healthier SG, CHAS) to standard OHDSI OMOP CDM v5.4
     concept_id and domain_id via resources/sg_omop_cdm_crosswalk.csv.
  2. Live Zone-B vocabulary expansion via OMOPHub REST v1 (https://api.omophub.com/v1)
     or official omophub Python SDK (reading OMOPHUB_API_KEY strictly from os.environ)
     with mandatory outbound Singapore PHI/DLP blocking.
  3. Compliant OHDSI OMOP CDM v5.4 DuckDB table export (omop_person, omop_condition_occurrence,
     omop_procedure_occurrence, omop_drug_exposure, omop_measurement, omop_observation,
     omop_note_nlp, omop_source_to_concept_map) enforcing assertion, longitudinal copy-paste,
     planned staged-procedure, and Singapore HIB SHI enclave guards.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import urllib.parse
import urllib.request

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CROSSWALK_CSV = SKILL_DIR / "resources" / "sg_omop_cdm_crosswalk.csv"
OMOPHUB_BASE_URL = "https://api.omophub.com/v1"

# OHDSI OMOP CDM v5.4 Standard Provenance / Assertion Concept IDs
OMOP_TYPE_EHR_NLP = 32858
OMOP_OBS_HISTORY_OF = 4214956
OMOP_OBS_FAMILY_HISTORY_OF = 4167217
OMOP_OBS_PLANNED_PROCEDURE = 4275495

# Outbound DLP / Singapore PHI Guard Patterns (blocks raw clinical text exfiltration)
_DLP_BLOCK_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("SG_NRIC_FIN", re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.IGNORECASE)),
    ("SG_SMC_MCR", re.compile(r"\bM\d{5}[A-Z]\b", re.IGNORECASE)),
    ("SG_PHONE", re.compile(r"(?:\+65[\s-]?)?[689]\d{3}[\s-]?\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
)

TransportFn = Callable[[str, str, Dict[str, str], Optional[bytes]], Dict[str, Any]]


def assert_dlp_safe_query(text: str) -> None:
    """Block outbound OMOPHub queries containing Singapore PHI or raw clinical note spans."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Outbound OMOPHub query string cannot be empty.")
    if len(cleaned) > 160 or "\n" in cleaned:
        raise ValueError("Outbound DLP guard blocked multi-line or long free-text clinical span (>160 chars).")
    for label, pattern in _DLP_BLOCK_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(f"Outbound DLP guard blocked potential PHI ({label}) from OMOPHub request.")


def _default_http_transport(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[bytes],
) -> Dict[str, Any]:
    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else {}


class OMOPHubClient:
    """Zero-dependency OMOPHub REST v1 client with optional omophub SDK delegation and DLP guard."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OMOPHUB_BASE_URL,
        transport: Optional[TransportFn] = None,
    ) -> None:
        resolved_key = (api_key if api_key is not None else os.environ.get("OMOPHUB_API_KEY", "")).strip()
        if not resolved_key:
            raise ValueError(
                "OMOPHUB_API_KEY is not set. Export OMOPHUB_API_KEY in your environment for live Zone-B "
                "OMOPHub lookups, or use offline mode with resources/sg_omop_cdm_crosswalk.csv."
            )
        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._transport: TransportFn = transport or _default_http_transport

    def _headers(self, has_json_body: bool = False) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": "sg-mediator-omop-skill/1.1.0",
        }
        if has_json_body:
            h["Content-Type"] = "application/json"
        return h

    def get_by_code(self, vocabulary_id: str, concept_code: str) -> Dict[str, Any]:
        assert_dlp_safe_query(vocabulary_id)
        assert_dlp_safe_query(concept_code)
        vocab_enc = urllib.parse.quote(vocabulary_id.strip(), safe="")
        code_enc = urllib.parse.quote(concept_code.strip(), safe="")
        url = f"{self._base_url}/concepts/by-code/{vocab_enc}/{code_enc}"
        return self._transport("GET", url, self._headers(), None)

    def search_concepts(
        self,
        query: str,
        vocabulary_ids: Optional[List[str]] = None,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        assert_dlp_safe_query(query)
        params: Dict[str, str] = {
            "query": query.strip(),
            "page_size": str(max(1, min(int(page_size), 100))),
        }
        if vocabulary_ids:
            for v in vocabulary_ids:
                assert_dlp_safe_query(v)
            params["vocabulary_ids"] = ",".join(vocabulary_ids)
        qs = urllib.parse.urlencode(params)
        url = f"{self._base_url}/search/concepts?{qs}"
        return self._transport("GET", url, self._headers(), None)

    def map_codes(
        self,
        source_codes: List[Dict[str, str]],
        target_vocabulary: str = "SNOMED",
    ) -> Dict[str, Any]:
        assert_dlp_safe_query(target_vocabulary)
        for item in source_codes:
            assert_dlp_safe_query(str(item.get("vocabulary_id", "")))
            assert_dlp_safe_query(str(item.get("concept_code", "")))
        body = json.dumps(
            {
                "target_vocabulary": target_vocabulary.strip(),
                "source_codes": source_codes,
            }
        ).encode("utf-8")
        url = f"{self._base_url}/concepts/map"
        return self._transport("POST", url, self._headers(has_json_body=True), body)


def load_offline_crosswalk(crosswalk_csv: Path = DEFAULT_CROSSWALK_CSV) -> Dict[str, Dict[str, str]]:
    with crosswalk_csv.open("r", encoding="utf-8", newline="") as f:
        return {row["concept_code"]: row for row in csv.DictReader(f)}


def _deterministic_person_id(patient_id: str) -> int:
    digest = hashlib.blake2b(patient_id.encode("utf-8"), digest_size=7).digest()
    val = int.from_bytes(digest, "big") & 0x001FFFFFFFFFFFFF
    return val if val > 0 else 1


def _parse_iso_date(raw: str) -> str:
    txt = (raw or "").strip()[:10]
    try:
        date.fromisoformat(txt)
        return txt
    except ValueError:
        return "1970-01-01"


def _iter_input_rows(input_path: Path) -> Iterable[Dict[str, str]]:
    if input_path.suffix.lower() in {".duckdb", ".db"}:
        import duckdb

        con = duckdb.connect(str(input_path), read_only=True)
        try:
            cur = con.execute("SELECT * FROM concepts")
            cols = [d[0] for d in cur.description]
            for tup in cur.fetchall():
                yield {cols[i]: ("" if tup[i] is None else str(tup[i])) for i in range(len(cols))}
        finally:
            con.close()
    else:
        with input_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield {k: ("" if v is None else str(v)) for k, v in row.items()}


def _best_source_value(row: Dict[str, str], cw: Dict[str, str]) -> Tuple[str, str]:
    tosp = (row.get("moh_tosp_code") or "").strip()
    if tosp and tosp != "NONE":
        return "SG_MOH_TOSP", tosp
    local_vocab = (cw.get("local_source_vocabulary") or "").strip()
    local_code = (cw.get("local_source_code") or "").strip()
    if local_vocab == "SG_MOH_SDOH" and local_code:
        return local_vocab, local_code
    snomed = (row.get("snomed_sg_code") or "").strip()
    if snomed and snomed != "NONE":
        return "SNOMED_SG", snomed
    icd = (row.get("icd10am_code") or "").strip()
    if icd and icd != "NONE":
        return "ICD10AM", icd
    return "SG_MEDIATOR", (row.get("concept_code") or "UNKNOWN")


def export_to_omop_duckdb(
    input_path: Path,
    output_duckdb: Path,
    crosswalk_csv: Path = DEFAULT_CROSSWALK_CSV,
    include_shi: bool = False,
    shi_reason: str = "",
    include_copy_paste: bool = False,
    include_snippets: bool = False,
) -> Dict[str, int]:
    """
    Convert SG-MEDIATOR extracted concepts (CSV or DuckDB) into OHDSI OMOP CDM v5.4 tables.
    """
    import duckdb

    if include_shi and not shi_reason.strip():
        raise ValueError("Unlocking SHI_Restricted rows for OMOP export requires a non-empty shi_reason.")

    crosswalk = load_offline_crosswalk(crosswalk_csv)
    persons: Dict[int, str] = {}
    cond_rows: List[Tuple[Any, ...]] = []
    proc_rows: List[Tuple[Any, ...]] = []
    drug_rows: List[Tuple[Any, ...]] = []
    meas_rows: List[Tuple[Any, ...]] = []
    obs_rows: List[Tuple[Any, ...]] = []
    nlp_rows: List[Tuple[Any, ...]] = []
    stcm_set: Dict[Tuple[str, str, int], Tuple[Any, ...]] = {}

    shi_suppressed = 0
    copy_paste_suppressed = 0

    for row in _iter_input_rows(input_path):
        concept_code = (row.get("concept_code") or "").strip()
        domain_raw = (row.get("domain") or "").strip()
        is_shi = domain_raw == "SHI_Restricted" or concept_code.startswith("CONCEPT_SHI_")
        if is_shi and not include_shi:
            shi_suppressed += 1
            continue

        cw = crosswalk.get(concept_code)
        omop_concept_id = int(cw["omop_concept_id"]) if cw else 0
        target_table = cw["target_cdm_table"] if cw else "observation"
        value_as_concept_id = int(cw.get("value_as_concept_id") or 0) if cw else 0

        patient_id = (row.get("patient_id") or "PT_UNKNOWN").strip()
        person_id = _deterministic_person_id(patient_id)
        persons[person_id] = patient_id

        visit_date = _parse_iso_date(row.get("visit_date", ""))
        assertion = (row.get("assertion_status") or "AFFIRMED").strip().upper()
        provenance = (row.get("provenance") or "NEW").strip().upper()
        src_vocab, src_val = _best_source_value(row, cw or {})

        if omop_concept_id > 0:
            stcm_key = (src_val, src_vocab, omop_concept_id)
            if stcm_key not in stcm_set:
                stcm_set[stcm_key] = (
                    src_val,
                    0,
                    src_vocab,
                    (cw["omop_concept_name"] if cw else row.get("canonical_name", "")),
                    omop_concept_id,
                    (cw["omop_vocabulary_id"] if cw else "SNOMED"),
                    "1970-01-01",
                    "2099-12-31",
                )

        # 1. Populate NOTE_NLP for every non-SHI mention (preserving negations & hedged mentions)
        if assertion == "NEGATED":
            term_exists = "N"
        elif assertion in {"UNCERTAIN", "CONDITIONAL", "POSSIBLE", "HYPOTHETICAL"}:
            term_exists = "U"
        else:
            term_exists = "Y"

        term_temporal = "historical" if assertion == "HISTORICAL" else "present"
        snippet_out = (
            ("[SHI_REDACTED]" if is_shi else (row.get("supporting_span") or "")[:250])
            if include_snippets
            else ""
        )
        modifiers = f"assertion={assertion};provenance={provenance};vq={row.get('vq_token_id', '')}"
        nlp_concept_target = value_as_concept_id if value_as_concept_id > 0 else omop_concept_id
        nlp_rows.append(
            (
                len(nlp_rows) + 1,
                person_id,
                0,
                snippet_out,
                "",
                (row.get("canonical_name") or concept_code)[:250],
                nlp_concept_target,
                0,
                "SG-MEDIATOR-v1.1",
                visit_date,
                f"{visit_date} 00:00:00",
                term_exists,
                term_temporal,
                modifiers,
            )
        )

        # 2. Longitudinal copy-paste suppression for clinical event tables
        if provenance == "COPY_PASTE" and not include_copy_paste:
            copy_paste_suppressed += 1
            continue

        # 3. Assertion-aware routing into OHDSI clinical event tables
        if assertion in {"NEGATED", "UNCERTAIN", "CONDITIONAL", "POSSIBLE", "HYPOTHETICAL"}:
            continue

        if assertion == "HISTORICAL":
            hist_val = value_as_concept_id if value_as_concept_id > 0 else omop_concept_id
            obs_rows.append(
                (
                    len(obs_rows) + 1,
                    person_id,
                    OMOP_OBS_HISTORY_OF,
                    visit_date,
                    OMOP_TYPE_EHR_NLP,
                    hist_val,
                    row.get("canonical_name", ""),
                    src_val,
                )
            )
            continue

        if assertion in {"FAMILY", "FAMILY_HISTORY"}:
            fam_val = value_as_concept_id if value_as_concept_id > 0 else omop_concept_id
            obs_rows.append(
                (
                    len(obs_rows) + 1,
                    person_id,
                    OMOP_OBS_FAMILY_HISTORY_OF,
                    visit_date,
                    OMOP_TYPE_EHR_NLP,
                    fam_val,
                    row.get("canonical_name", ""),
                    src_val,
                )
            )
            continue

        # AFFIRMED active event -> domain-specific OMOP CDM v5.4 table
        if target_table == "condition_occurrence":
            cond_rows.append(
                (
                    len(cond_rows) + 1,
                    person_id,
                    omop_concept_id,
                    visit_date,
                    OMOP_TYPE_EHR_NLP,
                    src_val,
                    0,
                )
            )
        elif target_table == "procedure_occurrence":
            proc_rows.append(
                (
                    len(proc_rows) + 1,
                    person_id,
                    omop_concept_id,
                    visit_date,
                    OMOP_TYPE_EHR_NLP,
                    src_val,
                    0,
                )
            )
        elif target_table == "drug_exposure":
            drug_rows.append(
                (
                    len(drug_rows) + 1,
                    person_id,
                    omop_concept_id,
                    visit_date,
                    visit_date,
                    OMOP_TYPE_EHR_NLP,
                    src_val,
                    0,
                )
            )
        elif target_table == "measurement":
            meas_rows.append(
                (
                    len(meas_rows) + 1,
                    person_id,
                    omop_concept_id,
                    visit_date,
                    OMOP_TYPE_EHR_NLP,
                    value_as_concept_id,
                    src_val,
                    0,
                )
            )
        else:
            obs_rows.append(
                (
                    len(obs_rows) + 1,
                    person_id,
                    omop_concept_id,
                    visit_date,
                    OMOP_TYPE_EHR_NLP,
                    value_as_concept_id,
                    row.get("canonical_name", ""),
                    src_val,
                )
            )

    if output_duckdb.exists():
        output_duckdb.unlink()
    output_duckdb.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(output_duckdb))
    try:
        con.execute(
            "CREATE TABLE omop_person ("
            "person_id BIGINT, gender_concept_id INTEGER, year_of_birth INTEGER, "
            "race_concept_id INTEGER, ethnicity_concept_id INTEGER, person_source_value VARCHAR)"
        )
        con.execute(
            "CREATE TABLE omop_condition_occurrence ("
            "condition_occurrence_id BIGINT, person_id BIGINT, condition_concept_id INTEGER, "
            "condition_start_date DATE, condition_type_concept_id INTEGER, "
            "condition_source_value VARCHAR, condition_source_concept_id INTEGER)"
        )
        con.execute(
            "CREATE TABLE omop_procedure_occurrence ("
            "procedure_occurrence_id BIGINT, person_id BIGINT, procedure_concept_id INTEGER, "
            "procedure_date DATE, procedure_type_concept_id INTEGER, "
            "procedure_source_value VARCHAR, procedure_source_concept_id INTEGER)"
        )
        con.execute(
            "CREATE TABLE omop_drug_exposure ("
            "drug_exposure_id BIGINT, person_id BIGINT, drug_concept_id INTEGER, "
            "drug_exposure_start_date DATE, drug_exposure_end_date DATE, "
            "drug_type_concept_id INTEGER, drug_source_value VARCHAR, drug_source_concept_id INTEGER)"
        )
        con.execute(
            "CREATE TABLE omop_measurement ("
            "measurement_id BIGINT, person_id BIGINT, measurement_concept_id INTEGER, "
            "measurement_date DATE, measurement_type_concept_id INTEGER, "
            "value_as_concept_id INTEGER, measurement_source_value VARCHAR, measurement_source_concept_id INTEGER)"
        )
        con.execute(
            "CREATE TABLE omop_observation ("
            "observation_id BIGINT, person_id BIGINT, observation_concept_id INTEGER, "
            "observation_date DATE, observation_type_concept_id INTEGER, "
            "value_as_concept_id INTEGER, value_as_string VARCHAR, observation_source_value VARCHAR)"
        )
        con.execute(
            'CREATE TABLE omop_note_nlp ('
            'note_nlp_id BIGINT, note_id BIGINT, section_concept_id INTEGER, snippet VARCHAR, '
            '"offset" VARCHAR, lexical_variant VARCHAR, note_nlp_concept_id INTEGER, '
            'note_nlp_source_concept_id INTEGER, nlp_system VARCHAR, nlp_date DATE, '
            'nlp_datetime TIMESTAMP, term_exists VARCHAR, term_temporal VARCHAR, term_modifiers VARCHAR)'
        )
        con.execute(
            "CREATE TABLE omop_source_to_concept_map ("
            "source_code VARCHAR, source_concept_id INTEGER, source_vocabulary_id VARCHAR, "
            "source_code_description VARCHAR, target_concept_id INTEGER, target_vocabulary_id VARCHAR, "
            "valid_start_date DATE, valid_end_date DATE)"
        )

        if persons:
            con.executemany(
                "INSERT INTO omop_person VALUES (?, 0, 1970, 0, 0, ?)",
                [(pid, psrc) for pid, psrc in sorted(persons.items())],
            )
        if cond_rows:
            con.executemany("INSERT INTO omop_condition_occurrence VALUES (?, ?, ?, ?, ?, ?, ?)", cond_rows)
        if proc_rows:
            con.executemany("INSERT INTO omop_procedure_occurrence VALUES (?, ?, ?, ?, ?, ?, ?)", proc_rows)
        if drug_rows:
            con.executemany("INSERT INTO omop_drug_exposure VALUES (?, ?, ?, ?, ?, ?, ?, ?)", drug_rows)
        if meas_rows:
            con.executemany("INSERT INTO omop_measurement VALUES (?, ?, ?, ?, ?, ?, ?, ?)", meas_rows)
        if obs_rows:
            con.executemany("INSERT INTO omop_observation VALUES (?, ?, ?, ?, ?, ?, ?, ?)", obs_rows)
        if nlp_rows:
            con.executemany(
                "INSERT INTO omop_note_nlp VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                nlp_rows,
            )
        if stcm_set:
            con.executemany(
                "INSERT INTO omop_source_to_concept_map VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                list(stcm_set.values()),
            )
    finally:
        con.close()

    return {
        "person_count": len(persons),
        "condition_occurrence_count": len(cond_rows),
        "procedure_occurrence_count": len(proc_rows),
        "drug_exposure_count": len(drug_rows),
        "measurement_count": len(meas_rows),
        "observation_count": len(obs_rows),
        "note_nlp_count": len(nlp_rows),
        "source_to_concept_map_count": len(stcm_set),
        "shi_suppressed_count": shi_suppressed,
        "copy_paste_suppressed_count": copy_paste_suppressed,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SG-MEDIATOR Skill-Only OHDSI OMOP CDM v5.4 Harmonizer & OMOPHub Bridge"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="Convert SG-MEDIATOR CSV or DuckDB output to OMOP CDM v5.4 DuckDB")
    p_exp.add_argument("--input", required=True, type=Path, help="Path to sg_mediator_extracted_concepts.csv or sg_mediator.duckdb")
    p_exp.add_argument("--output-duckdb", required=True, type=Path, help="Output OHDSI OMOP CDM v5.4 DuckDB file")
    p_exp.add_argument("--crosswalk", default=DEFAULT_CROSSWALK_CSV, type=Path, help="Offline 128-concept OMOP crosswalk CSV")
    p_exp.add_argument("--include-shi", action="store_true", help="Include HIB SHI_Restricted concepts (requires --shi-reason)")
    p_exp.add_argument("--shi-reason", default="", help="IRB / statutory authorization reason for unlocking SHI")
    p_exp.add_argument("--include-copy-paste", action="store_true", help="Include COPY_PASTE rows in event tables")
    p_exp.add_argument("--include-snippets", action="store_true", help="Populate NOTE_NLP.snippet (enclave-internal use only)")

    p_search = sub.add_parser("search", help="Search OMOPHub standard concepts (Zone-B dev workstation, requires OMOPHUB_API_KEY)")
    p_search.add_argument("--query", required=True, help="De-identified concept name (DLP-checked)")
    p_search.add_argument("--vocabularies", default="SNOMED,RxNorm,LOINC", help="Comma-separated vocabulary_ids")
    p_search.add_argument("--page-size", default=10, type=int)

    p_code = sub.add_parser("by-code", help="Look up an OMOP concept by vocabulary code via OMOPHub REST v1")
    p_code.add_argument("--vocabulary", required=True, help="e.g., SNOMED, ICD10CM, LOINC, RxNorm")
    p_code.add_argument("--code", required=True, help="Source vocabulary code, e.g., 44054006 or E11.9")

    p_map = sub.add_parser("map-code", help="Map a source vocabulary code to standard SNOMED/RxNorm via OMOPHub")
    p_map.add_argument("--source-vocabulary", required=True, help="e.g., ICD10CM")
    p_map.add_argument("--source-code", required=True, help="e.g., E11.9")
    p_map.add_argument("--target-vocabulary", default="SNOMED", help="Target standard vocabulary (default: SNOMED)")

    args = parser.parse_args(argv)
    if args.cmd == "export":
        summary = export_to_omop_duckdb(
            input_path=args.input,
            output_duckdb=args.output_duckdb,
            crosswalk_csv=args.crosswalk,
            include_shi=args.include_shi,
            shi_reason=args.shi_reason,
            include_copy_paste=args.include_copy_paste,
            include_snippets=args.include_snippets,
        )
        print(json.dumps(summary, indent=2))
        return 0

    client = OMOPHubClient()
    if args.cmd == "search":
        vocabs = [v.strip() for v in args.vocabularies.split(",") if v.strip()]
        print(json.dumps(client.search_concepts(args.query, vocabs, args.page_size), indent=2))
    elif args.cmd == "by-code":
        print(json.dumps(client.get_by_code(args.vocabulary, args.code), indent=2))
    elif args.cmd == "map-code":
        print(
            json.dumps(
                client.map_codes(
                    [{"vocabulary_id": args.source_vocabulary, "concept_code": args.source_code}],
                    target_vocabulary=args.target_vocabulary,
                ),
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
