"""
SG-MEDIATOR Dual-Head Hybrid Semantic Layer:
- Head 1: Open-Vocabulary Discrete Neural Interlingua VQ Codebook Tokenizer (T: X -> [K=16384])
  inspired by MahmoodLab's MEDIATOR (arXiv:2610.27047). Uses an INT8 ONNX biomedical concept
  encoder (`SapBERT-from-PubMedBERT-fulltext`, UMLS self-aligned `[CLS]` embeddings) projected into
  a mean-centered, variance-balanced 128-D spherical latent space and quantized via a trained
  K=16,384 centroid codebook (`vq_codebook_16384.npz`) with seeded ontology prototype anchoring.
  Falls back deterministically to character-trigram hashing only if offline model files are absent.
- Head 2: Deterministic & Semantic Multi-Ontology Crosswalk (UMLS CUI, SNOMED CT-SG, ICD-10-AM,
  ACHI, and Singapore MOH TOSP 1A-7C surgical codes).
"""
import csv
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import numpy as np


class OnnxPhraseEncoder:
    """INT8 ONNX SapBERT biomedical encoder -> [CLS] L2 -> centered 128-D PCA -> K=16384 VQ lookup."""

    def __init__(self, model_dir: Path):
        self.session = None
        self.tokenizer = None
        self.mean: Optional[np.ndarray] = None
        self.proj: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None
        self.anchor_index_to_code: Dict[int, str] = {}

        model_path = model_dir / "model_quantized.onnx"
        tok_path = model_dir / "tokenizer.json"
        cb_path = model_dir / "vq_codebook_16384.npz"
        if not (model_path.exists() and tok_path.exists() and cb_path.exists()):
            return
        try:
            import os
            import onnxruntime as ort
            from tokenizers import Tokenizer

            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "1"))
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
            self.input_names = {i.name for i in self.session.get_inputs()}
            self.tokenizer = Tokenizer.from_file(str(tok_path))
            self.tokenizer.enable_truncation(max_length=32)
            self.tokenizer.no_padding()

            with np.load(cb_path, allow_pickle=False) as cb:
                self.mean = cb["mean"].astype(np.float32)
                self.proj = cb["proj"].astype(np.float32)
                raw_c = cb["centroids"].astype(np.float32)
                norms = np.maximum(np.linalg.norm(raw_c, axis=1, keepdims=True), 1e-9)
                self.centroids = (raw_c / norms).astype(np.float32)
                if "anchor_indices" in cb and "anchor_codes" in cb:
                    for idx, code in zip(cb["anchor_indices"].tolist(), cb["anchor_codes"].tolist()):
                        self.anchor_index_to_code[int(idx)] = str(code)
        except Exception:
            self.session = None
            self.centroids = None

    @property
    def available(self) -> bool:
        return self.session is not None and self.centroids is not None

    @property
    def codebook_shape(self) -> Optional[Tuple[int, int]]:
        if self.centroids is None:
            return None
        return (int(self.centroids.shape[0]), int(self.centroids.shape[1]))

    def embed_projected(self, phrase: str) -> Optional[np.ndarray]:
        if not self.available:
            return None
        enc = self.tokenizer.encode(phrase)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feeds)[0]  # (1, T, 768)
        cls_vec = hidden[0, 0].astype(np.float32)
        cls_vec = cls_vec / max(float(np.linalg.norm(cls_vec)), 1e-9)
        z = (cls_vec - self.mean) @ self.proj
        z = z / max(float(np.linalg.norm(z)), 1e-9)
        return z.astype(np.float32)

    def quantize(self, phrase: str) -> Optional[Tuple[int, float]]:
        z = self.embed_projected(phrase)
        if z is None:
            return None
        sims = self.centroids @ z
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx])

    def bucket(self, phrase: str) -> Optional[int]:
        res = self.quantize(phrase)
        return res[0] if res is not None else None


class HybridSemanticLayer:
    def __init__(
        self,
        ontology_csv_path: Path,
        codebook_size: int = 16384,
        dim: int = 128,
        model_dir: Optional[Path] = None,
    ):
        self.codebook_size = codebook_size
        self.dim = dim
        self.concepts_by_code: Dict[str, Dict[str, Any]] = {}
        self.synonym_to_code: Dict[str, str] = {}
        self._vq_cache: Dict[str, str] = {}
        self._load_ontology_seed(ontology_csv_path)
        model_dir = model_dir or (
            Path(__file__).resolve().parent.parent / "models_onnx_int8" / "sapbert_pubmedbert_int8"
        )
        self.encoder = OnnxPhraseEncoder(model_dir)
        self.encoder_backend = (
            "onnx_int8_sapbert_vq16384" if self.encoder.available else "hash_trigram_fallback"
        )

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
        otherwise encodes with INT8 SapBERT and performs nearest-centroid lookup over K=16,384
        unit-norm biomedical centroids (or deterministic char-trigram fallback if offline weights absent).
        """
        if known_concept_code and known_concept_code in self.concepts_by_code:
            return self.concepts_by_code[known_concept_code]["vq_token_id"]

        norm = phrase.strip().lower()
        if norm in self.synonym_to_code:
            c_code = self.synonym_to_code[norm]
            return self.concepts_by_code[c_code]["vq_token_id"]

        cached = self._vq_cache.get(norm)
        if cached is not None:
            return cached

        if len(self._vq_cache) >= 16384:
            self._vq_cache.clear()

        # Primary: INT8 ONNX SapBERT -> 128-D centered PCA -> nearest K=16384 centroid
        onnx_bucket = self.encoder.bucket(norm)
        if onnx_bucket is not None:
            tok = f"VQ_{onnx_bucket % self.codebook_size:05d}"
            self._vq_cache[norm] = tok
            return tok

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
        q_bits = "".join("1" if x >= 0 else "0" for x in vec[:14])
        bucket = int(q_bits, 2) % self.codebook_size
        tok = f"VQ_{bucket:05d}"
        self._vq_cache[norm] = tok
        return tok

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
