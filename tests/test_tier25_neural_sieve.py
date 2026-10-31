import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deduplicator import LongitudinalDeduplicator
from hybrid_semantic_layer import HybridSemanticLayer, OnnxGlinerSpanExtractor
from sieve_and_risk import ClinicalSieveAndRiskEngine


class TestTier25NeuralSieve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantic = HybridSemanticLayer(ROOT / "config" / "sg_ontology_seed.csv")
        cls.sieve = ClinicalSieveAndRiskEngine(
            ROOT / "config" / "sg_clinical_rules.yaml",
            cls.semantic,
        )

    def _run_note(self, text: str):
        dedup = LongitudinalDeduplicator()
        dres = dedup.process_patient_visit("PAT_T25", "V01", "2026-10-31", text)
        return self.sieve.extract_from_visit(dres)

    def test_rapidfuzz_recovers_ocr_corrupted_clinical_concepts(self):
        note = (
            "Conclusion: Plan for stged PCl to proximal LAD in 4 weeks. "
            "Background of essential hypertensn. Genomic allele HLA-B*15:09 noted."
        )
        concepts = self._run_note(note)
        by_code = {c["concept_code"]: c for c in concepts}

        self.assertIn("CONCEPT_PCI_STAGED", by_code)
        self.assertEqual(by_code["CONCEPT_PCI_STAGED"]["assertion_status"], "AFFIRMED")
        self.assertEqual(by_code["CONCEPT_PCI_STAGED"]["evidence_source"], "TIER25_RAPIDFUZZ_OCR")

        self.assertIn("CONCEPT_HTN", by_code)
        self.assertEqual(by_code["CONCEPT_HTN"]["assertion_status"], "AFFIRMED")
        self.assertEqual(by_code["CONCEPT_HTN"]["evidence_source"], "TIER25_RAPIDFUZZ_OCR")

        # Digit-preservation guard: HLA-B*15:09 must NEVER fuzzy-match HLA-B*15:02 or HLA-B*58:01
        self.assertNotIn("CONCEPT_PGX_HLAB1502_POS", by_code)
        self.assertNotIn("CONCEPT_PGX_HLAB5801_POS", by_code)

    def test_zero_shot_sapbert_span_linker_extracts_unseen_paraphrases_and_respects_negation(self):
        # Verify none of these paraphrases exist verbatim in synonym_to_code
        unseen_phrases = [
            "maturity onset diabetes of adults",
            "calcific narrowing of aortic valve",
            "chronic renal glomerular insufficiency",
        ]
        for p in unseen_phrases:
            self.assertNotIn(p, self.semantic.synonym_to_code)

        note = (
            "68M referred with maturity onset diabetes of adults and "
            "calcific narrowing of aortic valve; denies chronic renal glomerular insufficiency. "
            "Admitted for symptomatic gallstones."
        )
        concepts = self._run_note(note)
        by_code = {(c["concept_code"], c["assertion_status"]): c for c in concepts}

        self.assertIn(("CONCEPT_DM_T2", "AFFIRMED"), by_code)
        self.assertEqual(
            by_code[("CONCEPT_DM_T2", "AFFIRMED")]["evidence_source"], "TIER25_SAPBERT_SPAN"
        )
        self.assertIn(("CONCEPT_AORTIC_STENOSIS", "AFFIRMED"), by_code)
        self.assertIn(("CONCEPT_CKD", "NEGATED"), by_code)

        # Distinct anatomy guard: gallbladder stones ("symptomatic gallstones") != CBD choledocholithiasis
        all_codes = {c["concept_code"] for c in concepts}
        self.assertNotIn("CONCEPT_CHOLEDOCHOLITHIASIS", all_codes)

    def test_match_anchored_concept_threshold_calibration(self):
        hit = self.semantic.match_anchored_concept("maturity onset diabetes of adults", min_cosine=0.68)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "CONCEPT_DM_T2")
        self.assertGreaterEqual(hit[1], 0.68)

        self.assertIsNone(self.semantic.match_anchored_concept("bukit panjang polyclinic", min_cosine=0.68))
        self.assertIsNone(self.semantic.match_anchored_concept("symptomatic gallstones", min_cosine=0.68))

    def test_optional_gliner_biomed_int8_slot_graceful_fallback_and_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_gliner = OnnxGlinerSpanExtractor(Path(tmp) / "gliner_biomed_int8")
            self.assertFalse(empty_gliner.available)
            self.assertEqual(empty_gliner.extract_spans("Patient has severe aortic stenosis."), [])

        # Simulate an active GLiNER-BioMed span extractor supplying a biomedical span
        class _StubGliner:
            available = True

            def extract_spans(self, text: str):
                idx = text.lower().find("maturity onset diabetes of adults")
                if idx != -1:
                    return [(idx, idx + len("maturity onset diabetes of adults"), "maturity onset diabetes of adults")]
                return []

        custom_sieve = ClinicalSieveAndRiskEngine(
            ROOT / "config" / "sg_clinical_rules.yaml",
            self.semantic,
            gliner_extractor=_StubGliner(),
        )
        dedup = LongitudinalDeduplicator()
        dres = dedup.process_patient_visit(
            "PAT_GLINER", "V01", "2026-10-31", " Clinic note: maturity onset diabetes of adults."
        )
        res = custom_sieve.extract_from_visit(dres)
        codes = {c["concept_code"]: c for c in res}
        self.assertIn("CONCEPT_DM_T2", codes)
        self.assertEqual(codes["CONCEPT_DM_T2"]["evidence_source"], "TIER25_GLINER_SAPBERT")

    def test_tier1_aho_corasick_automaton_active(self):
        self.assertIsNotNone(getattr(self.sieve, "_ac_automaton", None))


if __name__ == "__main__":
    unittest.main()
