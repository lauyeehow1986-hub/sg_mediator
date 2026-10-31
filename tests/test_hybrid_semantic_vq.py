"""
TDD Verification Suite for Head 1 Biomedical INT8 ONNX Encoder (SapBERT) + Trained K=16384 VQ Codebook.
"""
import sys
import time
import unittest
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from hybrid_semantic_layer import HybridSemanticLayer  # noqa: E402


class TestBiomedicalSapbertVQCodebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology_csv = ROOT_DIR / "config" / "sg_ontology_seed.csv"
        cls.layer = HybridSemanticLayer(cls.ontology_csv)

    def test_encoder_backend_is_biomedical_sapbert_vq16384(self):
        self.assertEqual(self.layer.encoder_backend, "onnx_int8_sapbert_vq16384")
        self.assertTrue(self.layer.encoder.available)
        self.assertEqual(self.layer.encoder.codebook_shape, (16384, 128))

    def test_codebook_has_16384_unit_norm_centroids(self):
        centroids = self.layer.encoder.centroids
        self.assertIsNotNone(centroids)
        self.assertEqual(centroids.shape, (16384, 128))
        norms = np.linalg.norm(centroids, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-2))

    def test_unseen_seeded_paraphrases_map_to_canonical_ontology_vq_tokens(self):
        trainer_src = (ROOT_DIR / "scripts" / "train_vq_codebook.py").read_text(encoding="utf-8").lower()
        ont_src = self.ontology_csv.read_text(encoding="utf-8").lower()
        unseen_paraphrases = [
            ("maturity onset diabetes of adults", {"VQ_01042"}),
            ("persistent elevation of systemic blood pressure", {"VQ_01105"}),
            ("endoscopic minimally invasive excision of gallbladder", {"VQ_07304"}),
            ("transcatheter coronary stent placement", {"VQ_07112", "VQ_07113", "VQ_07114"}),
            ("progressive decline in kidney glomerular function", {"VQ_02019"}),
            ("chaotic supraventricular rhythm lacking p waves", {"VQ_09011"}),
            ("surgical extraction of opaque crystalline lens with intraocular implant", {"VQ_07512"}),
            ("chronic psychotic illness with delusions", {"VQ_15002"}),
        ]
        exact_hits = 0
        details = []
        for phrase, expected_tokens in unseen_paraphrases:
            p_low = phrase.lower()
            self.assertNotIn(p_low, self.layer.synonym_to_code)
            self.assertNotIn(p_low, trainer_src, f"Data leakage: '{phrase}' found in train_vq_codebook.py")
            self.assertNotIn(p_low, ont_src, f"Data leakage: '{phrase}' found in sg_ontology_seed.csv")
            vq_tok = self.layer.quantize_open_vocab_text(phrase)
            details.append((phrase, vq_tok, sorted(expected_tokens)))
            if vq_tok in expected_tokens:
                exact_hits += 1
        self.assertGreaterEqual(
            exact_hits,
            6,
            f"Expected >=6/8 strictly held-out clinical paraphrases to hit anchored ontology VQ slots, got {exact_hits}/8: {details}",
        )

    def test_open_vocab_non_seeded_concepts_avoid_false_collision_and_preserve_geometry(self):
        seeded_tokens = {row["vq_token_id"] for row in self.layer.concepts_by_code.values()}
        open_vocab_pairs = [
            ("community-acquired bacterial pneumonia", "acute lobar pneumonia"),
            ("acute appendicitis", "acute inflammation of vermiform appendix"),
            ("distal radius wrist fracture", "colles fracture of distal radius"),
            ("iron deficiency microcytic anemia", "sideropenic microcytic anaemia"),
        ]
        unrelated_pairs = [
            ("community-acquired bacterial pneumonia", "distal radius wrist fracture"),
            ("acute appendicitis", "iron deficiency microcytic anemia"),
            ("acute lobar pneumonia", "colles fracture of distal radius"),
        ]

        pos_cos = []
        for a, b in open_vocab_pairs:
            tok_a = self.layer.quantize_open_vocab_text(a)
            tok_b = self.layer.quantize_open_vocab_text(b)
            self.assertNotIn(tok_a, seeded_tokens, f"Open-vocab '{a}' falsely collided with seeded {tok_a}")
            self.assertNotIn(tok_b, seeded_tokens, f"Open-vocab '{b}' falsely collided with seeded {tok_b}")
            idx_a = int(tok_a.split("_")[1])
            idx_b = int(tok_b.split("_")[1])
            pos_cos.append(float(self.layer.encoder.centroids[idx_a] @ self.layer.encoder.centroids[idx_b]))

        neg_cos = []
        for a, b in unrelated_pairs:
            idx_a = int(self.layer.quantize_open_vocab_text(a).split("_")[1])
            idx_b = int(self.layer.quantize_open_vocab_text(b).split("_")[1])
            neg_cos.append(float(self.layer.encoder.centroids[idx_a] @ self.layer.encoder.centroids[idx_b]))

        gap = float(np.mean(pos_cos) - np.mean(neg_cos))
        self.assertGreaterEqual(
            gap, 0.45, f"Expected quantized centroid cosine separation >= 0.45, got {gap:.3f} (pos={np.mean(pos_cos):.3f}, neg={np.mean(neg_cos):.3f})"
        )

    def test_graceful_fallback_when_model_dir_missing(self):
        fallback_layer = HybridSemanticLayer(self.ontology_csv, model_dir=ROOT_DIR / "models_onnx_int8" / "_missing_")
        self.assertEqual(fallback_layer.encoder_backend, "hash_trigram_fallback")
        tok = fallback_layer.quantize_open_vocab_text("acute cholecystitis")
        self.assertTrue(tok.startswith("VQ_"))
        idx = int(tok.split("_")[1])
        self.assertTrue(0 <= idx < 16384)

    def test_cpu_inference_latency_and_cache(self):
        phrase = "spontaneous coronary artery dissection"
        t0 = time.perf_counter()
        tok1 = self.layer.quantize_open_vocab_text(phrase)
        cold_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        tok2 = self.layer.quantize_open_vocab_text(phrase)
        warm_ms = (time.perf_counter() - t1) * 1000.0

        self.assertEqual(tok1, tok2)
        self.assertLess(cold_ms, 30.0)
        self.assertLess(warm_ms, 0.2)


if __name__ == "__main__":
    unittest.main()
