"""
SG-MEDIATOR Main Portable CLI Entry Point.
Processes a folder of multimodal clinical inputs (CSV/TSV tables, PDFs/notes, 12-lead ECG XMLs,
DICOM files) under strict 8GB or 16GB CPU memory bounds and outputs harmonized MEDIATOR VQ tokens
and Singapore ontology codes (UMLS, SNOMED-SG, ICD-10-AM, ACHI, MOH TOSP).
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

from deduplicator import LongitudinalDeduplicator
from hybrid_semantic_layer import HybridSemanticLayer
from ingest_multimodal import ingest_folder
from shi_enclave import SHIEnclaveGate
from sieve_and_risk import ClinicalSieveAndRiskEngine
from tier5_llm_arbitrator import Tier5Arbitrator


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    profile: str = "8gb",
    unlock_shi: bool = False,
    shi_reason: str = "",
) -> Dict[str, Any]:
    root_dir = Path(__file__).resolve().parent.parent
    config_dir = root_dir / "config"
    rules_path = config_dir / "sg_clinical_rules.yaml"
    ontology_path = config_dir / "sg_ontology_seed.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    semantic = HybridSemanticLayer(ontology_path)
    deduplicator = LongitudinalDeduplicator(shingle_size=8, overlap_threshold=0.85)
    # Tier-5 local LLM arbitrator: 16gb profile only; silently inactive if Ollama/llama-cli is absent
    arbitrator = Tier5Arbitrator(enabled=(profile == "16gb"), root_dir=root_dir)
    sieve = ClinicalSieveAndRiskEngine(rules_path, semantic, arbitrator=arbitrator)
    shi_gate = SHIEnclaveGate(output_dir, unlock_shi=unlock_shi, shi_reason=shi_reason)

    raw_records = ingest_folder(input_dir)
    all_extracted_rows: List[Dict[str, Any]] = []
    visit_summaries: List[Dict[str, Any]] = []
    total_shi_flagged = 0

    for rec in raw_records:
        patient_id = rec["patient_id"]
        visit_no = rec["visit_no"]
        visit_date = rec["visit_date"]

        dedup_res = deduplicator.process_patient_visit(
            patient_id=patient_id,
            visit_no=visit_no,
            visit_date=visit_date,
            raw_text=rec["free_text"],
            visit_type_hint=rec.get("visit_type_hint", ""),
        )

        concepts = sieve.extract_from_visit(
            dedup_result=dedup_res,
            structured_row=rec.get("structured_fields", {}),
        )

        # Merge multimodal direct findings (ECG XML / DICOM SR)
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
        total_shi_flagged += shi_count

        for c in safe_concepts:
            row_out = {
                "patient_id": patient_id,
                "visit_no": visit_no,
                "visit_date": visit_date,
                "modality": rec["modality"],
                "care_setting": dedup_res["care_setting"],
                "copy_paste_ratio": dedup_res["copy_paste_ratio"],
                **c,
            }
            all_extracted_rows.append(row_out)

        visit_summaries.append(
            {
                "patient_id": patient_id,
                "visit_no": visit_no,
                "visit_date": visit_date,
                "source_file": rec["source_file"],
                "modality": rec["modality"],
                "care_setting": dedup_res["care_setting"],
                "copy_paste_ratio": dedup_res["copy_paste_ratio"],
                "extracted_concept_count": len(safe_concepts),
                "shi_enclave_count": shi_count,
                "vq_token_sequence": [c["vq_token_id"] for c in safe_concepts if c["assertion_status"] == "AFFIRMED"],
            }
        )

    # Write CSV output
    csv_out_path = output_dir / "sg_mediator_extracted_concepts.csv"
    fieldnames = [
        "patient_id",
        "visit_no",
        "visit_date",
        "modality",
        "care_setting",
        "copy_paste_ratio",
        "concept_code",
        "vq_token_id",
        "canonical_name",
        "domain",
        "assertion_status",
        "provenance",
        "evidence_source",
        "umls_cui",
        "snomed_sg_code",
        "icd10am_code",
        "achi_code",
        "moh_tosp_code",
        "tosp_table_tier",
        "supporting_span",
    ]
    with csv_out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_extracted_rows)

    # Write JSONL summary
    jsonl_out_path = output_dir / "sg_mediator_visit_timelines.jsonl"
    with jsonl_out_path.open("w", encoding="utf-8") as f:
        for vs in visit_summaries:
            f.write(json.dumps(vs, ensure_ascii=False) + "\n")

    summary = {
        "profile": profile,
        "records_processed": len(raw_records),
        "total_concepts_extracted": len(all_extracted_rows),
        "total_shi_enclave_protected": total_shi_flagged,
        "tier5_llm_backend": arbitrator.backend,
        "vq_encoder_backend": semantic.encoder_backend,
        "tier5_llm_calls": arbitrator.calls,
        "output_csv": str(csv_out_path),
        "output_timelines_jsonl": str(jsonl_out_path),
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
    args = parser.parse_args()

    summary = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        profile=args.profile,
        unlock_shi=args.unlock_shi,
        shi_reason=args.shi_reason,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
