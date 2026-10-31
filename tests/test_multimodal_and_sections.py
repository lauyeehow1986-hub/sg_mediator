"""
TDD Verification Suite for Task W1.2:
1. Multi-pass clinical PDF section parser (`section_parser.py` — Page-1 conclusions, multi-delimiter headers,
   CONCLUSION + PLAN merging, and 40% tail fallback).
2. Scanned image-only PDF `RapidOCR` fallback in `ingest_multimodal.parse_pdf_or_text_document` (< 40 native chars).
3. Zero-copy `.parquet` clinical table ingestion via DuckDB in `ingest_multimodal.ingest_folder`.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pymupdf as fitz

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from ingest_multimodal import ingest_folder, parse_pdf_or_text_document  # noqa: E402
from section_parser import extract_conclusion_window, parse_clinical_sections  # noqa: E402


class TestMultimodalAndClinicalSections(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="sg_med_mm_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_page1_executive_conclusion_and_plan_merged(self):
        report = (
            "CATHETERIZATION REPORT - PAGE 1\n"
            "CONCLUSION:\n"
            "1. Severe two-vessel coronary artery disease.\n"
            "2. Successful primary PCI to culprit RCA.\n"
            "PLAN:\n"
            "Plan for staged PCI to proximal LAD in 4 weeks.\n"
            "PROCEDURAL DETAILS:\n"
            "6F radial sheath inserted. Heparin 7000 IU given.\n"
            "HEMODYNAMICS:\n"
            "Ao 128/74 mmHg, LVEDP 14 mmHg.\n"
        )
        sections = parse_clinical_sections(report)
        conc_blocks = [s for s in sections if s["section_type"] == "CONCLUSION_PLAN"]
        self.assertGreaterEqual(len(conc_blocks), 1)
        merged_text = " ".join(s["text"] for s in conc_blocks).lower()
        self.assertIn("two-vessel coronary artery disease", merged_text)
        self.assertIn("plan for staged pci to proximal lad", merged_text)
        self.assertNotIn("6f radial sheath", merged_text)

    def test_multi_delimiter_headers_roman_inline_and_pipe_table(self):
        roman_doc = "I. INDICATION:\nChest pain.\nIV. CONCLUSION AND RECOMMENDATIONS:\nStaged PCI to LCx."
        pipe_doc = "| HEADER | DETAILS |\n| CONCLUSION | Plan for staged PCI to LCx in 6 weeks |"
        inline_doc = "History: HTN.\nConclusion - Successful DES to RCA; for staged PCI to LAD."

        self.assertIn("staged pci to lcx", extract_conclusion_window(roman_doc).lower())
        self.assertIn("plan for staged pci to lcx", extract_conclusion_window(pipe_doc).lower())
        self.assertIn("for staged pci to lad", extract_conclusion_window(inline_doc).lower())

    def test_headerless_report_extracts_tail_40pct_fallback(self):
        lines = [f"Routine procedural narration line {i} with catheter advancement." for i in range(15)]
        lines.append("Successful stent deployment to RCA; staged intervention to LAD scheduled.")
        raw = "\n".join(lines)
        sections = parse_clinical_sections(raw)
        types = {s["section_type"] for s in sections}
        self.assertIn("CONCLUSION_TAIL_FALLBACK", types)
        self.assertIn("staged intervention to lad", extract_conclusion_window(raw).lower())

    def test_scanned_image_only_pdf_triggers_rapidocr_fallback(self):
        # Step 1: Create a temporary text PDF, rasterize to a PNG pixmap, and embed ONLY the image in a new PDF
        src_doc = fitz.open()
        p = src_doc.new_page(width=595, height=400)
        clinical_lines = (
            "Patient_ID: PATOCR01\n"
            "Visit_No: VOCR01\n"
            "Visit_Date: 2026-03-10\n"
            "Conclusion: Plan for staged PCI to LAD. HbA1c 7.8%."
        )
        p.insert_text((36, 60), clinical_lines, fontsize=16)
        pix = p.get_pixmap(dpi=200)
        png_bytes = pix.tobytes("png")
        src_doc.close()

        scanned_pdf = self.tmp_dir / "scanned_cath_report.pdf"
        img_doc = fitz.open()
        img_page = img_doc.new_page(width=595, height=400)
        img_page.insert_image(img_page.rect, stream=png_bytes)
        self.assertEqual(img_page.get_text("text").strip(), "")  # confirm 0 native text chars
        img_doc.save(str(scanned_pdf))
        img_doc.close()

        records = parse_pdf_or_text_document(scanned_pdf)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertTrue(rec.get("structured_fields", {}).get("ocr_used"), f"Expected OCR fallback on {rec}")
        low = rec["free_text"].lower()
        self.assertIn("staged pci", low)
        self.assertEqual(rec["patient_id"], "PATOCR01")

    def test_parquet_table_ingested_via_duckdb(self):
        pq_path = self.tmp_dir / "polyclinic_visits.parquet"
        con = duckdb.connect(":memory:")
        con.execute(
            f"COPY (SELECT 'PAT_PQ_01' AS patient_id, 'VPQ1' AS visit_no, '2026-04-02' AS visit_date, "
            f"'POLYCLINIC' AS visit_type, 'Essential HTN on amlodipine. Kopi O kosong daily.' AS free_text, "
            f"7.2 AS hba1c_pct, 148 AS sbp_mmhg, 92 AS dbp_mmhg) TO '{pq_path.as_posix()}' (FORMAT PARQUET)"
        )
        con.close()

        records = ingest_folder(self.tmp_dir)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["modality"], "TABLE_PARQUET")
        self.assertEqual(rec["patient_id"], "PAT_PQ_01")
        self.assertEqual(rec["visit_no"], "VPQ1")
        self.assertIn("kopi o kosong", rec["free_text"].lower())
        self.assertAlmostEqual(float(rec["structured_fields"]["hba1c_pct"]), 7.2, places=2)


if __name__ == "__main__":
    unittest.main()
