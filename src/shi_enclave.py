"""
Singapore Health Information Bill (HIB) & NEHR Sensitive Health Information (SHI) Enclave Gate.
Protects HIV/STD status, psychiatric diagnoses, substance abuse, termination of pregnancy,
and hereditary genetic test results with double-login flag enforcement and SHA-256 audit logging.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SHIEnclaveGate:
    def __init__(self, output_dir: Path, unlock_shi: bool = False, shi_reason: str = ""):
        self.output_dir = output_dir
        self.unlock_shi = unlock_shi
        self.shi_reason = shi_reason.strip()
        self.enclave_file = output_dir / "shi_enclave_restricted.jsonl"
        self.audit_log_file = output_dir / "shi_access_audit.log"

    def partition_and_mask_concepts(
        self,
        patient_id: str,
        visit_no: str,
        extracted_concepts: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Separates SHI concepts from general clinical concepts.
        Writes SHI concepts to the restricted enclave file, logs audit trail, and masks them
        in the main export unless both --unlock-shi and --shi-reason are provided.
        """
        safe_concepts: List[Dict[str, Any]] = []
        shi_count = 0

        for item in extracted_concepts:
            is_shi = item.get("domain") == "SHI_Restricted" or item.get("concept_code", "").startswith("CONCEPT_SHI_")
            if not is_shi:
                safe_concepts.append(item)
                continue

            shi_count += 1
            self._write_enclave_record(patient_id, visit_no, item)

            if self.unlock_shi and self.shi_reason:
                item_copy = dict(item)
                item_copy["shi_access_authorized"] = True
                item_copy["shi_access_reason"] = self.shi_reason
                safe_concepts.append(item_copy)
            else:
                # Emit masked placeholder in standard clinical table so downstream analysts know
                # an SHI flag exists without exposing protected details
                safe_concepts.append(
                    {
                        "concept_code": "SHI_MASKED_ENCLAVE_RECORD",
                        "vq_token_id": "VQ_SHI_REDACTED",
                        "canonical_name": "[SHI REDACTED - REQUIRES HIB DOUBLE-LOGIN AUTHORIZATION]",
                        "domain": "SHI_Restricted",
                        "umls_cui": "REDACTED",
                        "snomed_sg_code": "REDACTED",
                        "icd10am_code": "REDACTED",
                        "achi_code": "REDACTED",
                        "moh_tosp_code": "REDACTED",
                        "tosp_table_tier": "REDACTED",
                        "assertion_status": item.get("assertion_status", "AFFIRMED"),
                        "provenance": item.get("provenance", "NEW_AT_VISIT"),
                        "evidence_source": "SHI_ENCLAVE_GATE",
                        "supporting_span": "[REDACTED_UNDER_SINGAPORE_HEALTH_INFORMATION_BILL]",
                    }
                )

        return safe_concepts, shi_count

    def _write_enclave_record(self, patient_id: str, visit_no: str, item: Dict[str, Any]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp_utc": ts,
            "patient_id": patient_id,
            "visit_no": visit_no,
            "unlocked_in_standard_output": bool(self.unlock_shi and self.shi_reason),
            "access_reason": self.shi_reason or "LOCKED_DEFAULT_ENCLAVE_ROUTING",
            "shi_concept": item,
        }
        line = json.dumps(payload, ensure_ascii=False)
        digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
        with self.enclave_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        with self.audit_log_file.open("a", encoding="utf-8") as f:
            f.write(
                f"{ts} | patient={patient_id} | visit={visit_no} | code={item.get('concept_code')} "
                f"| unlocked={bool(self.unlock_shi and self.shi_reason)} | sha256={digest}\n"
            )
