"""
SG-MEDIATOR Main Portable CLI Entry Point.
Processes a folder of multimodal clinical inputs (CSV/TSV tables, PDFs/notes, 12-lead ECG XMLs,
DICOM files) under strict 8GB or 16GB CPU memory bounds and outputs harmonized MEDIATOR VQ tokens
and Singapore ontology codes (UMLS, SNOMED-SG, ICD-10-AM, ACHI, MOH TOSP).

Execution model:
  * Records are grouped by patient_id and sorted chronologically by (visit_date, visit_no); a patient's
    whole timeline is processed by ONE worker so longitudinal copy-paste deduplication stays correct.
  * Patients are distributed over a multiprocessing pool whose size / maxtasksperchild come from
    config/hardware_profiles.yaml (8gb: 2 workers, 16gb: 4 workers).
  * Results are loaded into DuckDB (memory_limit / threads from the profile) and exported from it.
  * SHI enclave writes are queued in workers and flushed sequentially by the parent process.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Iterable

import yaml

from deduplicator import LongitudinalDeduplicator
from hybrid_semantic_layer import HybridSemanticLayer
from ingest_multimodal import ingest_folder, is_supported_clinical_file
from shi_enclave import SHIEnclaveGate
from sieve_and_risk import ClinicalSieveAndRiskEngine
from tier5_llm_arbitrator import Tier5Arbitrator

ROOT_DIR = Path(__file__).resolve().parent.parent

CSV_FIELDS = [
    "patient_id", "visit_no", "visit_date", "modality", "care_setting", "copy_paste_ratio",
    "concept_code", "vq_token_id", "canonical_name", "domain", "assertion_status", "provenance",
    "evidence_source", "umls_cui", "snomed_sg_code", "icd10am_code", "achi_code", "moh_tosp_code",
    "tosp_table_tier", "supporting_span",
]

# Per-process state, built once by the pool initializer
_W: Dict[str, Any] = {}


def load_profile(profile: str) -> Dict[str, Any]:
    cfg = yaml.safe_load((ROOT_DIR / "config" / "hardware_profiles.yaml").read_text(encoding="utf-8"))
    return cfg["profiles"][profile]


def _init_worker(profile: str, unlock_shi: bool, shi_reason: str, output_dir: str) -> None:
    config_dir = ROOT_DIR / "config"
    prof = load_profile(profile)
    os.environ["OMP_NUM_THREADS"] = str(prof["omp_num_threads"])
    os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(prof["ort_intra_op_num_threads"])
    semantic = HybridSemanticLayer(config_dir / "sg_ontology_seed.csv")
    arbitrator = Tier5Arbitrator(enabled=bool(prof.get("enable_tier5_llm_arbitrator")), root_dir=ROOT_DIR)
    _W["semantic"] = semantic
    _W["arbitrator"] = arbitrator
    _W["sieve"] = ClinicalSieveAndRiskEngine(config_dir / "sg_clinical_rules.yaml", semantic, arbitrator=arbitrator)
    _W["shi"] = SHIEnclaveGate(Path(output_dir), unlock_shi=unlock_shi, shi_reason=shi_reason, defer_writes=True)


def _process_patient(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Processes one patient's chronologically sorted records with a fresh deduplicator."""
    semantic, sieve, shi_gate = _W["semantic"], _W["sieve"], _W["shi"]
    deduplicator = LongitudinalDeduplicator(shingle_size=8, overlap_threshold=0.85)
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    shi_total = 0
    shi_gate.pending = []
    calls_before = _W["arbitrator"].calls

    for rec in records:
        patient_id, visit_no, visit_date = rec["patient_id"], rec["visit_no"], rec["visit_date"]
        dedup_res = deduplicator.process_patient_visit(
            patient_id=patient_id,
            visit_no=visit_no,
            visit_date=visit_date,
            raw_text=rec["free_text"],
            visit_type_hint=rec.get("visit_type_hint", ""),
        )
        concepts = sieve.extract_from_visit(dedup_result=dedup_res, structured_row=rec.get("structured_fields", {}))

        for mm in rec.get("multimodal_findings", []):
            resolved_mm = semantic.resolve_concept(
                concept_code=mm["concept_code"],
                evidence_source=mm["evidence_source"],
                assertion_status="AFFIRMED",
                provenance="NEW_AT_VISIT",
                supporting_span=mm["supporting_span"],
                extra_metadata=mm.get("extra"),
            )
            if not any(c["concept_code"] == resolved_mm["concept_code"] for c in concepts):
                concepts.append(resolved_mm)

        safe_concepts, shi_count = shi_gate.partition_and_mask_concepts(patient_id, visit_no, concepts)
        shi_total += shi_count

        for c in safe_concepts:
            rows.append({
                "patient_id": patient_id, "visit_no": visit_no, "visit_date": visit_date,
                "modality": rec["modality"], "care_setting": dedup_res["care_setting"],
                "copy_paste_ratio": dedup_res["copy_paste_ratio"], **c,
            })
        summaries.append({
            "patient_id": patient_id, "visit_no": visit_no, "visit_date": visit_date,
            "source_file": rec["source_file"], "modality": rec["modality"],
            "care_setting": dedup_res["care_setting"], "copy_paste_ratio": dedup_res["copy_paste_ratio"],
            "extracted_concept_count": len(safe_concepts), "shi_enclave_count": shi_count,
            "vq_token_sequence": [c["vq_token_id"] for c in safe_concepts if c["assertion_status"] == "AFFIRMED"],
        })

    return {
        "rows": rows, "summaries": summaries, "shi_total": shi_total,
        "shi_pending": list(shi_gate.pending), "llm_calls": _W["arbitrator"].calls - calls_before,
        "pid": mp.current_process().pid,
    }


def _group_by_patient(records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    by_patient: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_patient.setdefault(r["patient_id"], []).append(r)
    return [sorted(v, key=lambda r: (str(r["visit_date"]), str(r["visit_no"]))) for _, v in sorted(by_patient.items())]


def _stream_results_to_staging(
    result_iter: Iterable[Dict[str, Any]],
    tmp_concepts: Path,
    tmp_vt: Path,
    jsonl_out: Path,
    shi_flush: SHIEnclaveGate,
) -> Tuple[int, int, int, List[int]]:
    vt_cols = [
        "patient_id", "visit_no", "visit_date", "source_file", "modality", "care_setting",
        "copy_paste_ratio", "extracted_concept_count", "shi_enclave_count", "vq_token_sequence",
    ]
    total_concepts = 0
    total_shi = 0
    llm_calls = 0
    worker_pids = set()

    with (
        tmp_concepts.open("w", encoding="utf-8", newline="") as fc,
        tmp_vt.open("w", encoding="utf-8", newline="") as fv,
        jsonl_out.open("w", encoding="utf-8") as fj,
    ):
        wc = csv.writer(fc)
        wv = csv.writer(fv)
        wc.writerow(CSV_FIELDS)
        wv.writerow(vt_cols)

        for res in result_iter:
            worker_pids.add(res["pid"])
            for r in res["rows"]:
                wc.writerow(["" if r.get(c) is None else str(r.get(c, "")) for c in CSV_FIELDS])
            total_concepts += len(res["rows"])

            for s in res["summaries"]:
                wv.writerow([
                    s["patient_id"], s["visit_no"], s["visit_date"], s["source_file"], s["modality"],
                    s["care_setting"], s["copy_paste_ratio"], s["extracted_concept_count"],
                    s["shi_enclave_count"], json.dumps(s["vq_token_sequence"]),
                ])
                fj.write(json.dumps(s, ensure_ascii=False) + "\n")

            total_shi += res["shi_total"]
            llm_calls += res["llm_calls"]
            shi_flush.flush_entries(res["shi_pending"])

    return total_concepts, total_shi, llm_calls, sorted(worker_pids)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _compute_input_hashes(input_dir: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for p in sorted(input_dir.iterdir()):
        if is_supported_clinical_file(p):
            hashes[p.name] = _sha256_file(p)
    return hashes


def _load_checkpoint_hashes(db_path: Path) -> Optional[Dict[str, str]]:
    if not db_path.exists():
        return None
    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        if not {"processed_files", "concepts", "visit_timelines"}.issubset(tables):
            return None
        return {
            str(r[0]): str(r[1])
            for r in con.execute("SELECT source_file, sha256 FROM processed_files").fetchall()
        }
    except Exception:
        return None
    finally:
        con.close()


def _finalize_duckdb(
    output_dir: Path,
    prof: Dict[str, Any],
    tmp_concepts: Path,
    tmp_vt: Path,
    file_hashes: Dict[str, str],
    dirty_files: Set[str],
    stale_prev_files: Set[str],
    incremental: bool = False,
    has_new_rows: bool = True,
) -> Tuple[Path, int]:
    import duckdb

    db_path = output_dir / "sg_mediator.duckdb"
    if not incremental and db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"PRAGMA memory_limit='{prof['duckdb_memory_limit']}'")
        con.execute(f"PRAGMA threads={int(prof['duckdb_threads'])}")

        if not incremental:
            con.execute(
                f"CREATE TABLE ontology AS SELECT * FROM read_csv_auto('{(ROOT_DIR / 'config' / 'sg_ontology_seed.csv').as_posix()}')"
            )
            cols = ", ".join(f"{c} VARCHAR" for c in CSV_FIELDS)
            con.execute(f"CREATE TABLE concepts ({cols})")
            con.execute(
                "CREATE TABLE visit_timelines (patient_id VARCHAR, visit_no VARCHAR, visit_date VARCHAR, "
                "source_file VARCHAR, modality VARCHAR, care_setting VARCHAR, copy_paste_ratio DOUBLE, "
                "extracted_concept_count INTEGER, shi_enclave_count INTEGER, vq_token_sequence VARCHAR)"
            )
            con.execute(
                "CREATE TABLE processed_files (source_file VARCHAR, sha256 VARCHAR, processed_at VARCHAR)"
            )
            con.execute(
                "CREATE VIEW patient_problem_list AS SELECT patient_id, concept_code, canonical_name, "
                "min(visit_date) AS first_seen, max(visit_date) AS last_seen, count(*) AS n_mentions "
                "FROM concepts WHERE assertion_status='AFFIRMED' GROUP BY patient_id, concept_code, canonical_name"
            )
        elif stale_prev_files:
            placeholders = ", ".join("?" for _ in stale_prev_files)
            stale_list = sorted(stale_prev_files)
            con.execute(
                f"DELETE FROM concepts WHERE (patient_id, visit_no) IN "
                f"(SELECT patient_id, visit_no FROM visit_timelines WHERE source_file IN ({placeholders}))",
                stale_list,
            )
            con.execute(f"DELETE FROM visit_timelines WHERE source_file IN ({placeholders})", stale_list)
            con.execute(f"DELETE FROM processed_files WHERE source_file IN ({placeholders})", stale_list)

        if has_new_rows and tmp_concepts.exists():
            con.execute(
                f"INSERT INTO concepts SELECT * FROM read_csv('{tmp_concepts.as_posix()}', header=true, all_varchar=true)"
            )
        tmp_concepts.unlink(missing_ok=True)

        if has_new_rows and tmp_vt.exists():
            con.execute(f"INSERT INTO visit_timelines SELECT * FROM read_csv('{tmp_vt.as_posix()}', header=true)")
        tmp_vt.unlink(missing_ok=True)

        now_iso = datetime.now(timezone.utc).isoformat()
        to_record = dirty_files if incremental else set(file_hashes.keys())
        if to_record:
            rows_pf = [(fn, file_hashes[fn], now_iso) for fn in sorted(to_record) if fn in file_hashes]
            if rows_pf:
                con.executemany("INSERT INTO processed_files VALUES (?, ?, ?)", rows_pf)

        csv_out = output_dir / "sg_mediator_extracted_concepts.csv"
        con.execute(
            f"COPY (SELECT * FROM concepts ORDER BY patient_id, visit_date, visit_no) "
            f"TO '{csv_out.as_posix()}' (HEADER, DELIMITER ',')"
        )

        if incremental:
            jsonl_out = output_dir / "sg_mediator_visit_timelines.jsonl"
            vt_rows = con.execute(
                "SELECT patient_id, visit_no, visit_date, source_file, modality, care_setting, "
                "copy_paste_ratio, extracted_concept_count, shi_enclave_count, vq_token_sequence "
                "FROM visit_timelines ORDER BY patient_id, visit_date, visit_no"
            ).fetchall()
            with jsonl_out.open("w", encoding="utf-8") as fj:
                for r in vt_rows:
                    seq_val = json.loads(r[9]) if r[9] else []
                    fj.write(
                        json.dumps(
                            {
                                "patient_id": r[0],
                                "visit_no": r[1],
                                "visit_date": r[2],
                                "source_file": r[3],
                                "modality": r[4],
                                "care_setting": r[5],
                                "copy_paste_ratio": r[6],
                                "extracted_concept_count": r[7],
                                "shi_enclave_count": r[8],
                                "vq_token_sequence": seq_val,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        total_db_concepts = int(con.execute("SELECT count(*) FROM concepts").fetchone()[0])
        return csv_out, total_db_concepts
    finally:
        con.close()


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    profile: str = "8gb",
    unlock_shi: bool = False,
    shi_reason: str = "",
    resume: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prof = load_profile(profile)
    prev_omp = os.environ.get("OMP_NUM_THREADS")
    prev_ort = os.environ.get("ORT_INTRA_OP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(prof["omp_num_threads"])
    os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(prof["ort_intra_op_num_threads"])

    try:
        return _run_pipeline_inner(input_dir, output_dir, prof, profile, unlock_shi, shi_reason, resume)
    finally:
        if prev_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev_omp
        if prev_ort is None:
            os.environ.pop("ORT_INTRA_OP_NUM_THREADS", None)
        else:
            os.environ["ORT_INTRA_OP_NUM_THREADS"] = prev_ort


def _run_pipeline_inner(
    input_dir: Path,
    output_dir: Path,
    prof: Dict[str, Any],
    profile: str,
    unlock_shi: bool,
    shi_reason: str,
    resume: bool,
) -> Dict[str, Any]:

    file_hashes = _compute_input_hashes(input_dir)
    prior_hashes = _load_checkpoint_hashes(output_dir / "sg_mediator.duckdb") if resume else None
    incremental = prior_hashes is not None

    if incremental and prior_hashes is not None:
        unchanged_files = {fn for fn, sha in file_hashes.items() if prior_hashes.get(fn) == sha}
        dirty_files = {fn for fn in file_hashes if fn not in unchanged_files}
        stale_prev_files = {fn for fn in prior_hashes if fn not in unchanged_files}
        skipped_cached_files = len(unchanged_files)
        raw_records = ingest_folder(input_dir, only_files=dirty_files) if dirty_files else []
    else:
        dirty_files = set(file_hashes.keys())
        stale_prev_files = set()
        skipped_cached_files = 0
        raw_records = ingest_folder(input_dir)

    groups = _group_by_patient(raw_records)
    n_workers = max(1, min(int(prof["worker_processes"]), max(1, len(groups))))
    chunksize = max(1, min(16, len(groups) // (n_workers * 4))) if groups else 1
    initargs = (profile, unlock_shi, shi_reason, str(output_dir))

    tmp_concepts = output_dir / "_concepts_load.tmp.csv"
    tmp_vt = output_dir / "_visits_load.tmp.csv"
    jsonl_out = output_dir / ("_visits_delta.tmp.jsonl" if incremental else "sg_mediator_visit_timelines.jsonl")
    shi_flush = SHIEnclaveGate(output_dir, unlock_shi=unlock_shi, shi_reason=shi_reason)

    if not groups:
        total_concepts_delta, total_shi, llm_calls, worker_pids = 0, 0, 0, []
    elif n_workers == 1 or len(groups) <= 1:
        _init_worker(*initargs)
        total_concepts_delta, total_shi, llm_calls, worker_pids = _stream_results_to_staging(
            (_process_patient(g) for g in groups), tmp_concepts, tmp_vt, jsonl_out, shi_flush
        )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=initargs,
            maxtasksperchild=int(prof["max_tasks_per_child"]),
        ) as pool:
            total_concepts_delta, total_shi, llm_calls, worker_pids = _stream_results_to_staging(
                pool.imap(_process_patient, groups, chunksize=chunksize),
                tmp_concepts, tmp_vt, jsonl_out, shi_flush,
            )

    if incremental:
        jsonl_out.unlink(missing_ok=True)

    csv_out, total_db_concepts = _finalize_duckdb(
        output_dir=output_dir,
        prof=prof,
        tmp_concepts=tmp_concepts,
        tmp_vt=tmp_vt,
        file_hashes=file_hashes,
        dirty_files=dirty_files,
        stale_prev_files=stale_prev_files,
        incremental=incremental,
        has_new_rows=bool(groups),
    )
    final_jsonl = output_dir / "sg_mediator_visit_timelines.jsonl"

    sapbert_dir = ROOT_DIR / "models_onnx_int8" / "sapbert_pubmedbert_int8"
    if "semantic" in _W:
        vq_backend = _W["semantic"].encoder_backend
    elif all((sapbert_dir / fn).exists() for fn in ("model_quantized.onnx", "tokenizer.json", "vq_codebook_16384.npz")):
        vq_backend = "onnx_int8_sapbert_vq16384"
    else:
        vq_backend = "hash_trigram_fallback"

    probe = Tier5Arbitrator(enabled=bool(prof.get("enable_tier5_llm_arbitrator")), root_dir=ROOT_DIR)
    summary = {
        "profile": profile,
        "resume_mode": bool(resume),
        "skipped_cached_files": skipped_cached_files,
        "worker_processes": n_workers,
        "distinct_worker_pids_observed": len(worker_pids),
        "patients": len(groups),
        "records_processed": len(raw_records),
        "total_concepts_extracted": total_db_concepts if incremental else total_concepts_delta,
        "total_shi_enclave_protected": total_shi,
        "tier5_llm_backend": probe.backend,
        "tier5_llm_calls": llm_calls,
        "vq_encoder_backend": vq_backend,
        "duckdb_memory_limit": prof["duckdb_memory_limit"],
        "output_csv": str(csv_out),
        "output_timelines_jsonl": str(final_jsonl),
        "output_duckdb": str(output_dir / "sg_mediator.duckdb"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SG-MEDIATOR Portable Multimodal Clinical CLI")
    parser.add_argument("--input-dir", required=True, help="Folder containing PDFs, XMLs, DICOMs, CSVs")
    parser.add_argument("--output-dir", required=True, help="Folder to write extracted outputs")
    parser.add_argument("--profile", default="8gb", choices=["8gb", "16gb"], help="Hardware RAM ceiling profile")
    parser.add_argument("--unlock-shi", action="store_true", help="Unlock Sensitive Health Information in standard export")
    parser.add_argument("--shi-reason", default="", help="Mandatory audit reason when --unlock-shi is set")
    parser.add_argument("--resume", action="store_true", help="Skip unchanged files using DuckDB SHA-256 checkpoints")
    args = parser.parse_args()

    summary = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        profile=args.profile,
        unlock_shi=args.unlock_shi,
        shi_reason=args.shi_reason,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
