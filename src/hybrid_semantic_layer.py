"""
SG-MEDIATOR Dual-Head Hybrid Semantic Layer:
- Head 1: Open-Vocabulary Discrete Neural Interlingua VQ Codebook Tokenizer (T: X -> [K=16384])
  inspired by MahmoodLab's MEDIATOR (arXiv:2610.27047). Uses deterministic character/word n-gram
  random projection + vector quantization when offline ONNX encoder weights are not yet attached,
  or ONNX Runtime INT8 encoder when present in models_onnx_int8/.
- Head 2: Deterministic & Fuzzy Multi-Ontology Crosswalk (UMLS CUI, SNOMED CT-SG, ICD-10-AM,
  ACHI, and Singapore MOH TOSP 1A-7C surgical codes).
"""
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


class OnnxPhraseEncoder:
    """INT8 ONNX sentence encoder (MiniLM-L6) -> mean-pool -> L2 -> orthonormal projection -> 14 sign bits.

    The 14-bit sign code indexes the shared K=16384 discrete codebook (random-hyperplane VQ over
    real semantic embeddings, so paraphrases land in the same or neighbouring buckets). This is NOT a
    learned MEDIATOR codebook; it is a fixed, reproducible quantizer over an off-the-shelf encoder.
    Loaded lazily with ORT thread limits from the 8gb/16gb profile; returns None if weights are absent.
    """

    def __init__(self, model_dir: Path, bits: int = 14, seed: int = 20261029):
        self.session = None
        self.tokenizer = None
        self.bits = bits
        model_path = model_dir / "model_quantized.onnx"
        tok_path = model_dir / "tokenizer.json"
        if not (model_path.exists() and tok_path.exists()):
            return
        try:
            import os
            import onnxruntime as ort
            from tokenizers import Tokenizer

            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "1"))
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
            self.input_names = {i.name for i in self.session.get_inputs()}
            self.tokenizer = Tokenizer.from_file(str(tok_path))
            self.tokenizer.enable_truncation(max_length=64)
            self.tokenizer.enable_padding()
            rng = np.random.default_rng(seed)
            q, _ = np.linalg.qr(rng.standard_normal((384, 384)).astype(np.float32))
            self.projection = q[:, :bits].astype(np.float32)  # orthonormal columns
        except Exception:
            self.session = None

    @property
    def available(self) -> bool:
        return self.session is not None

    def bucket(self, phrase: str) -> Optional[int]:
        if not self.available:
            return None
        enc = self.tokenizer.encode(phrase)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feeds)[0]  # (1, T, 384)
        m = mask[..., None].astype(np.float32)
        vec = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)
        vec = vec[0]
        vec = vec / max(float(np.linalg.norm(vec)), 1e-9)
        code = (vec @ self.projection) >= 0
        return int("".join("1" if b else "0" for b in code), 2) % (1 << self.bits)


class HybridSemanticLayer:
    def __init__(self, ontology_csv_path: Path, codebook_size: int = 16384, dim: int = 128,
                 model_dir: Optional[Path] = None):
        self.codebook_size = codebook_size
        self.dim = dim
        self.concepts_by_code: Dict[str, Dict[str, Any]] = {}
        self.synonym_to_code: Dict[str, str] = {}
        self._load_ontology_seed(ontology_csv_path)
        model_dir = model_dir or (Path(__file__).resolve().parent.parent / "models_onnx_int8" / "minilm_l6_int8")
        self.encoder = OnnxPhraseEncoder(model_dir)
        self.encoder_backend = "onnx_int8_minilm" if self.encoder.available else "hash_trigram_fallback"

    def _load_ontology_seed(self, csv_path: Path):
        if not csv_path.exists():
            return
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c_code = row["concept_code"].strip()
                synonyms = [s.strip().lower() for s in row.get("synonyms", "").split("|") if s.strip()]
                row["synonyms_list"] = synonyms
                self.concepts_by_code[c_code] = row
                for syn in synonyms:
                    self.synonym_to_code[syn] = c_code
                self.synonym_to_code[row["canonical_name"].strip().lower()] = c_code

    def quantize_open_vocab_text(self, phrase: str, known_concept_code: Optional[str] = None) -> str:
        """
        Maps any clinical phrase into the discrete VQ codebook [K=16384].
        If aligned to a seeded concept cluster, returns its canonical VQ prototype ID;
        otherwise computes a deterministic L2-normalized n-gram projection to the nearest VQ bucket.
        """
        if known_concept_code and known_concept_code in self.concepts_by_code:
            return self.concepts_by_code[known_concept_code]["vq_token_id"]

        norm = phrase.strip().lower()
        if norm in self.synonym_to_code:
            c_code = self.synonym_to_code[norm]
            return self.concepts_by_code[c_code]["vq_token_id"]

        # Preferred: real INT8 ONNX encoder -> orthonormal projection -> discrete bucket
        onnx_bucket = self.encoder.bucket(norm)
        if onnx_bucket is not None:
            return f"VQ_{onnx_bucket % self.codebook_size:05d}"

        # Fallback (weights absent): deterministic char-trigram projection into [0, K-1]
        vec = np.zeros(self.dim, dtype=np.float32)
        padded = f"<{norm}>"
        for i in range(max(1, len(padded) - 2)):
            trigram = padded[i : i + 3]
            digest = hashlib.blake2b(trigram.encode("utf-8"), digest_size=4).digest()
            idx = int.from_bytes(digest[:2], "little") % self.dim
            sign = 1.0 if (digest[2] & 1) == 0 else -1.0
            vec[idx] += sign
        norm_val = float(np.linalg.norm(vec))
        if norm_val > 0:
            vec /= norm_val
        # Hash quantized direction to discrete codebook ID [1..K]
        q_bits = "".join("1" if x >= 0 else "0" for x in vec[:14])
        bucket = int(q_bits, 2) % self.codebook_size
        return f"VQ_{bucket:05d}"

    def resolve_concept(
        self,
        concept_code: str,
        evidence_source: str,
        assertion_status: str = "AFFIRMED",
        provenance: str = "NEW_AT_VISIT",
        supporting_span: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = self.concepts_by_code.get(concept_code, {})
        vq_token = self.quantize_open_vocab_text(supporting_span or concept_code, concept_code)
        record = {
            "concept_code": concept_code,
            "vq_token_id": vq_token,
            "canonical_name": meta.get("canonical_name", concept_code),
            "domain": meta.get("domain", "Unclassified"),
            "umls_cui": meta.get("umls_cui", "UNMAPPED"),
            "snomed_sg_code": meta.get("snomed_sg_code", "UNMAPPED"),
            "icd10am_code": meta.get("icd10am_code", "UNMAPPED"),
            "achi_code": meta.get("achi_code", "NONE"),
            "moh_tosp_code": meta.get("moh_tosp_code", "NONE"),
            "tosp_table_tier": meta.get("tosp_table_tier", "NONE"),
            "assertion_status": assertion_status,
            "provenance": provenance,
            "evidence_source": evidence_source,
            "supporting_span": supporting_span,
        }
        if extra_metadata:
            record.update(extra_metadata)
        return record
