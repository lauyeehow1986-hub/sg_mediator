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
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np


class OnnxGlinerSpanExtractor:
    """
    Optional INT8 ONNX biomedical span extractor (`Ihor/gliner-biomed-bi-small-v1.0`).
    Looks for `model_quantized.onnx`, `tokenizer.json`, and `gliner_config.json` inside
    `models_onnx_int8/gliner_biomed_int8/`. Remains cleanly inactive (`.available == False`,
    zero RAM overhead) when optional weights are not installed.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.session = None
        self.tokenizer = None
        self.config: Dict[str, Any] = {}
        model_dir = model_dir or (
            Path(__file__).resolve().parent.parent / "models_onnx_int8" / "gliner_biomed_int8"
        )
        model_path = model_dir / "model_quantized.onnx"
        tok_path = model_dir / "tokenizer.json"
        cfg_path = model_dir / "gliner_config.json"
        if not (model_path.exists() and tok_path.exists() and cfg_path.exists()):
            return
        try:
            import os
            import onnxruntime as ort
            from tokenizers import Tokenizer

            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "2"))
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
            self.tokenizer = Tokenizer.from_file(str(tok_path))
            self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            self.session = None
            self.tokenizer = None

    @property
    def available(self) -> bool:
        return self.session is not None and self.tokenizer is not None

    def extract_spans(self, text: str) -> List[Tuple[int, int, str]]:
        if not self.available or not text.strip():
            return []
        try:
            enc = self.tokenizer.encode(text)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            input_names = {i.name for i in self.session.get_inputs()}
            feeds: Dict[str, Any] = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = ids
            if "attention_mask" in input_names:
                feeds["attention_mask"] = mask
            if not feeds:
                return []
            outputs = self.session.run(None, feeds)
            if not outputs or outputs[0] is None:
                return []
            logits = np.asarray(outputs[0])
            # Decode high-confidence token spans using fast offset mapping when bi-encoder head is present
            offsets = enc.offsets
            spans: List[Tuple[int, int, str]] = []
            if logits.ndim >= 2 and len(offsets) == logits.shape[1]:
                scores = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -20.0, 20.0)))
                token_hits = np.max(scores, axis=-1) if scores.ndim > 1 else scores
                in_span = False
                s_char = 0
                e_char = 0
                for tok_idx, sc in enumerate(token_hits):
                    c0, c1 = offsets[tok_idx]
                    if c1 <= c0:
                        continue
                    if float(sc) >= 0.65:
                        if not in_span:
                            s_char = c0
                            in_span = True
                        e_char = c1
                    elif in_span:
                        spans.append((s_char, e_char, text[s_char:e_char]))
                        in_span = False
                if in_span and e_char > s_char:
                    spans.append((s_char, e_char, text[s_char:e_char]))
            return spans
        except Exception:
            return []


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
            so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "2"))
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
            # Pre-warm ONNX CPU execution arena so first live clinical span has deterministic <15 ms latency
            self.embed_projected("clinical concept encoder warmup")
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
        self.concept_digit_signatures: Dict[str, set] = {}
        self.synonym_to_code: Dict[str, str] = {}
        self._vq_cache: Dict[str, str] = {}
        self._anchor_match_cache: Dict[Tuple[str, float], Optional[Tuple[str, float]]] = {}
        self._load_ontology_seed(ontology_csv_path)
        base_models_dir = Path(__file__).resolve().parent.parent / "models_onnx_int8"
        model_dir = model_dir or (base_models_dir / "sapbert_pubmedbert_int8")
        self.encoder = OnnxPhraseEncoder(model_dir)
        self.gliner_extractor = OnnxGlinerSpanExtractor(base_models_dir / "gliner_biomed_int8")
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
                canon_low = row["canonical_name"].strip().lower()
                self.synonym_to_code[canon_low] = c_code
                # Strip administrative year tags like '(Fifth Universal Definition 2026)' when indexing clinical digits
                canon_clean = canon_low.split("(")[0].strip()
                sigs = {
                    "".join(ch for ch in t if ch.isdigit())
                    for t in [canon_clean, *synonyms]
                    if any(ch.isdigit() for ch in t)
                }
                self.concept_digit_signatures[c_code] = sigs

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

    def match_anchored_concept(
        self, phrase: str, min_cosine: float = 0.68
    ) -> Optional[Tuple[str, float]]:
        """
        Zero-shot biomedical span linker: projects `phrase` via INT8 SapBERT into the 128-D
        Voronoi-carved VQ latent space and returns `(concept_code, cosine)` ONLY if its nearest
        centroid among all K=16,384 centroids is a seeded ontology anchor with `cosine >= min_cosine`.
        """
        norm = " ".join(phrase.strip().lower().split())
        if not norm or not self.encoder.available:
            return None
        if norm in self.synonym_to_code:
            return (self.synonym_to_code[norm], 1.0)

        cache_key = (norm, round(float(min_cosine), 4))
        if cache_key in self._anchor_match_cache:
            return self._anchor_match_cache[cache_key]
        if len(self._anchor_match_cache) >= 16384:
            self._anchor_match_cache.clear()

        q_res = self.encoder.quantize(norm)
        result: Optional[Tuple[str, float]] = None
        if q_res is not None:
            best_idx, sim = q_res
            c_code = self.encoder.anchor_index_to_code.get(best_idx)
            if c_code is not None and sim >= min_cosine:
                meta = self.concepts_by_code.get(c_code, {})
                domain = str(meta.get("domain", ""))
                cand_digits = "".join(ch for ch in norm if ch.isdigit())
                allowed_digits = self.concept_digit_signatures.get(c_code, set())
                # Clinical safety guard: never hallucinate PGx/somatic alleles or cross-match conflicting numbers
                if not domain.startswith("Genomics") and (
                    not cand_digits or not allowed_digits or cand_digits in allowed_digits
                ):
                    result = (c_code, float(sim))

        self._anchor_match_cache[cache_key] = result
        return result

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
