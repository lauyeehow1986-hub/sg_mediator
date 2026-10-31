import csv
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cli import run_pipeline
from ingest_multimodal import iter_dicom_2d_slices_lazy, parse_dicom_file, parse_ecg_waveform_xml


def _synth_qrs_train(fs: int, duration_s: float, r_times_s, r_amp_uv: float = 1200.0, s_amp_uv: float = -300.0):
    n = int(fs * duration_s)
    t = np.arange(n, dtype=np.float32) / float(fs)
    sig = 15.0 * np.sin(2.0 * np.pi * 0.25 * t)  # gentle respiratory baseline wander
    for rt in r_times_s:
        sig += r_amp_uv * np.exp(-0.5 * ((t - rt) / 0.012) ** 2)
        sig += s_amp_uv * np.exp(-0.5 * ((t - (rt + 0.028)) / 0.014) ** 2)
    return np.round(sig).astype(int)


class TestEcgDspDicomAndCheckpoints(unittest.TestCase):
    def test_raw_ecg_xml_blank_statement_detects_afib_via_rr_irregularity_and_sinus_negative(self):
        fs = 500
        # Irregularly irregular RR intervals (CV_RR ~ 0.30)
        afib_rrs = [0.46, 0.94, 0.50, 1.08, 0.44, 0.91, 0.52, 1.14, 0.47, 0.89, 0.51]
        afib_times = np.cumsum([0.4] + afib_rrs)
        lead_ii_af = _synth_qrs_train(fs, 10.0, afib_times, r_amp_uv=1250.0, s_amp_uv=-250.0)

        # Normal sinus control (regular 0.80s RR intervals)
        sinus_times = np.arange(0.5, 9.5, 0.80)
        lead_ii_nsr = _synth_qrs_train(fs, 10.0, sinus_times, r_amp_uv=1250.0, s_amp_uv=-250.0)

        with tempfile.TemporaryDirectory() as tmp:
            af_xml = Path(tmp) / "raw_afib.xml"
            af_xml.write_text(
                f"""<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <PatientDemographics><PatientID>PAT_AF</PatientID></PatientDemographics>
  <OrderInfo><VisitNumber>V_AF</VisitNumber><AcquisitionDate>2026-05-01</AcquisitionDate></OrderInfo>
  <GlobalMeasurements>
    <HeartRate>86</HeartRate><QRSDuration>90</QRSDuration><QTCorrected>412</QTCorrected>
    <InterpretationStatement></InterpretationStatement>
  </GlobalMeasurements>
  <Waveform>
    <LeadID>II</LeadID><SampleRate>{fs}</SampleRate>
    <WaveformData>{" ".join(str(x) for x in lead_ii_af)}</WaveformData>
  </Waveform>
</RestingECG>""",
                encoding="utf-8",
            )
            recs_af = parse_ecg_waveform_xml(af_xml)
            mm_af = {m["concept_code"]: m for m in recs_af[0]["multimodal_findings"]}
            self.assertIn("CONCEPT_ECG_AFIB", mm_af)
            self.assertEqual(mm_af["CONCEPT_ECG_AFIB"]["evidence_source"], "ECG_DSP_RR_IRREGULARITY")
            self.assertNotIn("CONCEPT_ECG_LVH", mm_af)

            nsr_xml = Path(tmp) / "raw_sinus.xml"
            nsr_xml.write_text(
                f"""<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <PatientDemographics><PatientID>PAT_NSR</PatientID></PatientDemographics>
  <OrderInfo><VisitNumber>V_NSR</VisitNumber><AcquisitionDate>2026-05-01</AcquisitionDate></OrderInfo>
  <GlobalMeasurements>
    <HeartRate>75</HeartRate><QRSDuration>88</QRSDuration><QTCorrected>405</QTCorrected>
    <InterpretationStatement></InterpretationStatement>
  </GlobalMeasurements>
  <Waveform>
    <LeadID>II</LeadID><SampleRate>{fs}</SampleRate>
    <WaveformData>{" ".join(str(x) for x in lead_ii_nsr)}</WaveformData>
  </Waveform>
</RestingECG>""",
                encoding="utf-8",
            )
            recs_nsr = parse_ecg_waveform_xml(nsr_xml)
            self.assertEqual(recs_nsr[0]["multimodal_findings"], [])

    def test_raw_ecg_xml_blank_statement_detects_sokolow_lyon_lvh(self):
        fs = 500
        beat_times = np.arange(0.5, 6.0, 0.80)
        # V1: small R (+250 uV), deep S (-1850 uV); V5: tall R (+2150 uV) => |SV1| + RV5 = 4000 uV >= 3500 uV
        v1 = _synth_qrs_train(fs, 6.5, beat_times, r_amp_uv=250.0, s_amp_uv=-1850.0)
        v5 = _synth_qrs_train(fs, 6.5, beat_times, r_amp_uv=2150.0, s_amp_uv=-200.0)

        with tempfile.TemporaryDirectory() as tmp:
            lvh_xml = Path(tmp) / "raw_lvh.xml"
            lvh_xml.write_text(
                f"""<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <PatientDemographics><PatientID>PAT_LVH</PatientID></PatientDemographics>
  <OrderInfo><VisitNumber>V_LVH</VisitNumber><AcquisitionDate>2026-05-02</AcquisitionDate></OrderInfo>
  <GlobalMeasurements>
    <HeartRate>75</HeartRate><QRSDuration>98</QRSDuration><QTCorrected>420</QTCorrected>
    <InterpretationStatement></InterpretationStatement>
  </GlobalMeasurements>
  <Waveform>
    <LeadID>V1</LeadID><SampleRate>{fs}</SampleRate>
    <WaveformData>{" ".join(str(x) for x in v1)}</WaveformData>
  </Waveform>
  <Waveform>
    <LeadID>V5</LeadID><SampleRate>{fs}</SampleRate>
    <WaveformData>{" ".join(str(x) for x in v5)}</WaveformData>
  </Waveform>
</RestingECG>""",
                encoding="utf-8",
            )
            recs = parse_ecg_waveform_xml(lvh_xml)
            mm = {m["concept_code"]: m for m in recs[0]["multimodal_findings"]}
            self.assertIn("CONCEPT_ECG_LVH", mm)
            self.assertEqual(mm["CONCEPT_ECG_LVH"]["evidence_source"], "ECG_DSP_SOKOLOW_LYON")
            self.assertNotIn("CONCEPT_ECG_AFIB", mm)

    def test_deeply_nested_binary_dicom_tid3902_sr_and_lazy_2d_slice_iterator(self):
        def _code_seq(meaning: str) -> Sequence:
            c = Dataset()
            c.CodeValue = "12345"
            c.CodingSchemeDesignator = "SCT"
            c.CodeMeaning = meaning
            return Sequence([c])

        def _num_item(meaning: str, val: float) -> Dataset:
            item = Dataset()
            item.ValueType = "NUM"
            item.ConceptNameCodeSequence = _code_seq(meaning)
            mv = Dataset()
            mv.NumericValue = str(val)
            item.MeasuredValueSequence = Sequence([mv])
            return item

        def _vessel_container(vessel_meaning: str, score: float) -> Dataset:
            v_cont = Dataset()
            v_cont.ValueType = "CONTAINER"
            v_cont.ConceptNameCodeSequence = _code_seq("Vessel Calcium Section")
            mod = Dataset()
            mod.ValueType = "CODE"
            mod.ConceptNameCodeSequence = _code_seq("Finding Site")
            mod.ConceptCodeSequence = _code_seq(vessel_meaning)
            v_cont.ContentSequence = Sequence([mod, _num_item("Calcium Score", score)])
            return v_cont

        with tempfile.TemporaryDirectory() as tmp:
            dcm_sr_path = Path(tmp) / "nested_tid3902.dcm"
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.33"
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds = FileDataset(str(dcm_sr_path), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.PatientID = "PAT_DCM_NESTED"
            ds.AccessionNumber = "ACC_NESTED_3902"
            ds.StudyDate = "20260615"
            ds.Modality = "SR"
            ds.StudyDescription = "Cardiac CT Calcium TID 3902"

            summary_cont = Dataset()
            summary_cont.ValueType = "CONTAINER"
            summary_cont.ConceptNameCodeSequence = _code_seq("Overall Summary")
            summary_cont.ContentSequence = Sequence([_num_item("Global Agatston Calcium Score", 368.4)])

            vessels_cont = Dataset()
            vessels_cont.ValueType = "CONTAINER"
            vessels_cont.ConceptNameCodeSequence = _code_seq("Vessels")
            vessels_cont.ContentSequence = Sequence(
                [
                    _vessel_container("Left Anterior Descending Coronary Artery", 248.4),
                    _vessel_container("Right Coronary Artery", 120.0),
                ]
            )

            root_cont = Dataset()
            root_cont.ValueType = "CONTAINER"
            root_cont.ConceptNameCodeSequence = _code_seq("TID 3902 Root")
            root_cont.ContentSequence = Sequence([summary_cont, vessels_cont])
            ds.ContentSequence = Sequence([root_cont])
            ds.save_as(str(dcm_sr_path), enforce_file_format=True)

            recs = parse_dicom_file(dcm_sr_path)
            self.assertEqual(len(recs), 1)
            self.assertAlmostEqual(recs[0]["structured_fields"]["agatston_score"], 368.4, places=1)
            mm = recs[0]["multimodal_findings"][0]
            self.assertEqual(mm["concept_code"], "CONCEPT_CAD_CALCIFICATION")
            self.assertAlmostEqual(mm["extra"]["vessel_agatston"].get("LAD", 0.0), 248.4, places=1)
            self.assertAlmostEqual(mm["extra"]["vessel_agatston"].get("RCA", 0.0), 120.0, places=1)

            # Verify lazy 2D slice generator on a 3-slice CT dataset
            ct_path = Path(tmp) / "tiny_ct.dcm"
            ct_meta = FileMetaDataset()
            ct_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            ct_meta.MediaStorageSOPInstanceUID = generate_uid()
            ct_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ct_ds = FileDataset(str(ct_path), {}, file_meta=ct_meta, preamble=b"\0" * 128)
            ct_ds.Modality = "CT"
            ct_ds.Rows = 8
            ct_ds.Columns = 8
            ct_ds.NumberOfFrames = 3
            ct_ds.SamplesPerPixel = 1
            ct_ds.PhotometricInterpretation = "MONOCHROME2"
            ct_ds.BitsAllocated = 16
            ct_ds.BitsStored = 16
            ct_ds.HighBit = 15
            ct_ds.PixelRepresentation = 0
            vol = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
            ct_ds.PixelData = vol.tobytes()
            ct_ds.save_as(str(ct_path), enforce_file_format=True)

            slices = list(iter_dicom_2d_slices_lazy(ct_path))
            self.assertEqual(len(slices), 3)
            self.assertEqual(slices[0].shape, (8, 8))

    def test_duckdb_sha256_checkpoint_resume_skips_unchanged_and_updates_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            inbox = Path(tmp) / "inbox"
            out = Path(tmp) / "out"
            inbox.mkdir()

            (inbox / "pat_r1.txt").write_text(
                "Patient_ID: PAT_R1\nVisit_No: V1\nVisit_Date: 2026-04-01\nAdmitted with ST-elevation myocardial infarction.\n",
                encoding="utf-8",
            )
            (inbox / "pat_r2.txt").write_text(
                "Patient_ID: PAT_R2\nVisit_No: V1\nVisit_Date: 2026-04-02\nClinic follow-up for bronchial asthma.\n",
                encoding="utf-8",
            )

            s1 = run_pipeline(inbox, out, profile="8gb", resume=False)
            self.assertEqual(s1.get("skipped_cached_files", 0), 0)
            self.assertEqual(s1["records_processed"], 2)

            # Second run with resume=True and identical files: 0 reprocessed, all rows preserved
            s2 = run_pipeline(inbox, out, profile="8gb", resume=True)
            self.assertEqual(s2["skipped_cached_files"], 2)
            self.assertEqual(s2["records_processed"], 0)

            with (out / "sg_mediator_extracted_concepts.csv").open("r", encoding="utf-8") as f:
                rows2 = list(csv.DictReader(f))
            by_pat2 = {(r["patient_id"], r["concept_code"]) for r in rows2}
            self.assertIn(("PAT_R1", "CONCEPT_AMI_STEMI"), by_pat2)
            self.assertIn(("PAT_R2", "CONCEPT_ASTHMA_BRONCHIAL"), by_pat2)

            # Modify pat_r2.txt and add pat_r3.txt
            (inbox / "pat_r2.txt").write_text(
                "Patient_ID: PAT_R2\nVisit_No: V1\nVisit_Date: 2026-04-02\nClinic follow-up for chronic obstructive pulmonary disease.\n",
                encoding="utf-8",
            )
            (inbox / "pat_r3.txt").write_text(
                "Patient_ID: PAT_R3\nVisit_No: V1\nVisit_Date: 2026-04-03\nConfirmed dengue fever.\n",
                encoding="utf-8",
            )

            s3 = run_pipeline(inbox, out, profile="8gb", resume=True)
            self.assertEqual(s3["skipped_cached_files"], 1)
            self.assertEqual(s3["records_processed"], 2)

            with (out / "sg_mediator_extracted_concepts.csv").open("r", encoding="utf-8") as f:
                rows3 = list(csv.DictReader(f))
            by_pat3 = {(r["patient_id"], r["concept_code"]) for r in rows3}
            self.assertIn(("PAT_R1", "CONCEPT_AMI_STEMI"), by_pat3)
            self.assertIn(("PAT_R2", "CONCEPT_COPD_AIRWAY"), by_pat3)
            self.assertNotIn(("PAT_R2", "CONCEPT_ASTHMA_BRONCHIAL"), by_pat3)
            self.assertIn(("PAT_R3", "CONCEPT_DENGUE_FEVER"), by_pat3)


if __name__ == "__main__":
    unittest.main()
