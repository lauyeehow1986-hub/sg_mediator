"""
Folder-Driven Multimodal Ingestion Router for SG-MEDIATOR (8GB / 16GB RAM CPU).
Supports:
1. Structured & Semi-Structured Tables (.csv, .tsv, .parquet) with patient_id, visit_no, free_text, and lab columns
2. Clinical PDFs (.pdf) via PyMuPDF with lazy per-worker RapidOCR ONNX fallback on scanned pages (< 40 chars)
3. 12-Lead ECG Waveform XMLs (.xml — HL7 aECG, Philips Sierra, GE MUSE) + rhythm analysis
4. DICOM Files (.dcm / .dcm.json — Header + TID 3902 Coronary Calcium Agatston Structured Reports)
"""
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Any, Optional, Set, Tuple
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from section_parser import extract_conclusion_window, parse_clinical_sections

_OCR_ENGINE: Optional[Any] = None
_ECG_ONNX_SESSION: Optional[Any] = None
_ECG_ONNX_CFG: Dict[str, Any] = {}


def _get_ocr_engine() -> Optional[Any]:
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            _OCR_ENGINE = RapidOCR()
        except Exception:
            _OCR_ENGINE = False
    return _OCR_ENGINE if _OCR_ENGINE is not False else None


def _row_to_visit_record(norm_row: Dict[str, Any], idx: int, source_name: str, modality: str) -> Dict[str, Any]:
    patient_id = (
        norm_row.get("patient_id")
        or norm_row.get("patientid")
        or norm_row.get("case_id")
        or f"PAT_TBL_{idx:04d}"
    )
    visit_no = (
        norm_row.get("visit_no")
        or norm_row.get("visit_id")
        or norm_row.get("visitnumber")
        or f"VIS_{idx:04d}"
    )
    visit_date = str(norm_row.get("visit_date") or norm_row.get("date") or "2026-01-01")
    free_text = str(
        norm_row.get("free_text")
        or norm_row.get("clinical_note")
        or norm_row.get("note_text")
        or norm_row.get("freetext")
        or ""
    )
    return {
        "source_file": source_name,
        "modality": modality,
        "patient_id": str(patient_id),
        "visit_no": str(visit_no),
        "visit_date": visit_date,
        "visit_type_hint": str(norm_row.get("visit_type", "") or ""),
        "free_text": free_text,
        "structured_fields": norm_row,
        "multimodal_findings": [],
    }


def parse_csv_or_tsv_table(file_path: Path) -> List[Dict[str, Any]]:
    delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
    records = []
    with file_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for idx, row in enumerate(reader, start=1):
            norm_row = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
            records.append(_row_to_visit_record(norm_row, idx, file_path.name, "TABLE_CSV"))
    return records


def parse_parquet_table(file_path: Path) -> List[Dict[str, Any]]:
    import duckdb

    con = duckdb.connect(":memory:")
    try:
        cur = con.execute(f"SELECT * FROM read_parquet('{file_path.as_posix()}')")
        cols = [d[0].strip().lower() for d in cur.description]
        records = []
        for idx, vals in enumerate(cur.fetchall(), start=1):
            norm_row = {
                c: (v.strip() if isinstance(v, str) else ("" if v is None else v))
                for c, v in zip(cols, vals)
            }
            records.append(_row_to_visit_record(norm_row, idx, file_path.name, "TABLE_PARQUET"))
        return records
    finally:
        con.close()


def parse_pdf_or_text_document(file_path: Path) -> List[Dict[str, Any]]:
    text_content = ""
    ocr_used = False
    if file_path.suffix.lower() == ".pdf":
        try:
            try:
                import pymupdf as fitz  # type: ignore
            except ImportError:
                import fitz  # type: ignore

            doc = fitz.open(str(file_path))
            try:
                pages = []
                for page in doc:
                    native_txt = page.get_text("text") or ""
                    if len(native_txt.strip()) < 40 and (page.get_images(full=False) or page.get_drawings()):
                        ocr = _get_ocr_engine()
                        if ocr is not None:
                            pix = page.get_pixmap(dpi=200)
                            ocr_res, _ = ocr(pix.tobytes("png"))
                            if ocr_res:
                                native_txt = "\n".join(str(item[1]) for item in ocr_res if len(item) >= 2)
                                ocr_used = True
                    pages.append(native_txt)
                text_content = "\n".join(pages)
            finally:
                doc.close()
        except ImportError:
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(file_path))
                text_content = "\n".join((p.extract_text() or "") for p in reader.pages)
            except Exception as e:
                text_content = f"[PDF extraction error: {e}]"
    else:
        text_content = file_path.read_text(encoding="utf-8", errors="replace")

    # Extract optional header metadata from text (tolerant to OCR underscores/spaces)
    patient_id = file_path.stem.split("_")[0].upper()
    visit_no = f"DOC_{file_path.stem.upper()}"
    visit_date = "2026-01-15"
    for line in text_content.splitlines()[:12]:
        m_id = re.match(r"^\s*patient[_\s]*id\s*[:\-]\s*(\S+)", line, re.IGNORECASE)
        if m_id:
            patient_id = m_id.group(1).strip()
            continue
        m_vn = re.match(r"^\s*visit[_\s]*no\s*[:\-]\s*(\S+)", line, re.IGNORECASE)
        if m_vn:
            visit_no = m_vn.group(1).strip()
            continue
        m_vd = re.match(r"^\s*visit[_\s]*date\s*[:\-]\s*(\S+)", line, re.IGNORECASE)
        if m_vd:
            visit_date = m_vd.group(1).strip()

    sections = parse_clinical_sections(text_content)
    return [
        {
            "source_file": file_path.name,
            "modality": "PDF_OR_TEXT_DOC",
            "patient_id": patient_id,
            "visit_no": visit_no,
            "visit_date": visit_date,
            "visit_type_hint": "DISCHARGE_OR_REPORT",
            "free_text": text_content,
            "structured_fields": {
                "ocr_used": ocr_used,
                "section_count": len(sections),
                "conclusion_window": extract_conclusion_window(text_content),
            },
            "multimodal_findings": [],
        }
    ]


def _normalize_lead_name(raw_name: str) -> str:
    s = raw_name.strip().upper()
    for prefix in ("MDC_ECG_LEAD_", "ECG_LEAD_", "LEAD_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s


def _parse_numeric_samples(raw_text: str, scale: float = 1.0) -> Optional[np.ndarray]:
    nums = [
        float(x) * scale
        for x in raw_text.replace(",", " ").split()
        if x.strip().lstrip("-").replace(".", "", 1).isdigit()
    ]
    return np.asarray(nums, dtype=np.float32) if nums else None


def _get_optional_ecg_onnx() -> Tuple[Optional[Any], Dict[str, Any]]:
    global _ECG_ONNX_SESSION, _ECG_ONNX_CFG
    if _ECG_ONNX_SESSION is None:
        ecg_dir = Path(__file__).resolve().parent.parent / "models_onnx_int8" / "ecg_1dcnn_int8"
        m_path = ecg_dir / "model_quantized.onnx"
        c_path = ecg_dir / "ecg_config.json"
        if m_path.exists() and c_path.exists():
            try:
                import os
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "1"))
                so.inter_op_num_threads = 1
                _ECG_ONNX_SESSION = ort.InferenceSession(str(m_path), so, providers=["CPUExecutionProvider"])
                _ECG_ONNX_CFG = json.loads(c_path.read_text(encoding="utf-8"))
            except Exception:
                _ECG_ONNX_SESSION = False
        else:
            _ECG_ONNX_SESSION = False
    return (_ECG_ONNX_SESSION if _ECG_ONNX_SESSION is not False else None), _ECG_ONNX_CFG


def parse_ecg_waveform_xml(file_path: Path) -> List[Dict[str, Any]]:
    """
    Streams 12-lead ECG XML files (Philips / GE MUSE / HL7 aECG), extracts global intervals
    (Heart Rate, PR, QRS, QT, QTc), decodes per-lead waveform arrays, runs raw R-peak RR-interval
    irregularity DSP (AFib) and Sokolow-Lyon voltage criteria (LVH) when statement text is absent,
    and optionally invokes an INT8 1D-CNN ONNX classifier if installed.
    """
    tree = ET.parse(str(file_path))
    root = tree.getroot()

    def _find_text(tag_suffix: str, default: str = "") -> str:
        for el in root.iter():
            local_name = el.tag.split("}")[-1].lower()
            if local_name == tag_suffix.lower() and el.text:
                return el.text.strip()
        return default

    patient_id = _find_text("patientid", "PAT_ECG_001")
    visit_no = _find_text("visitnumber", f"ECG_{file_path.stem}")
    visit_date = _find_text("acquisitiondate", "2026-02-01")
    hr_bpm = float(_find_text("heartrate", "72") or 72)
    qrs_ms = float(_find_text("qrsduration", "88") or 88)
    qtc_ms = float(_find_text("qtcorrected", "410") or 410)
    statement = _find_text("interpretationstatement", "")

    fs_hz = float(_find_text("samplerate", _find_text("samplefrequency", "500")) or 500.0)
    leads_uv: Dict[str, np.ndarray] = {}

    # 1. Extract multi-lead containers (Philips / GE MUSE <Waveform> or <LeadData>, HL7 aECG <sequence>)
    for container in root.iter():
        cname = container.tag.split("}")[-1].lower()
        if cname in ("waveform", "leaddata", "sequence"):
            lead_id = ""
            scale = 1.0
            samples_txt = ""
            for ch in container.iter():
                ch_name = ch.tag.split("}")[-1].lower()
                if ch_name == "leadid" and ch.text:
                    lead_id = _normalize_lead_name(ch.text)
                elif ch_name == "code" and ch.attrib.get("code"):
                    lead_id = _normalize_lead_name(ch.attrib["code"])
                elif ch_name in ("samplerate", "samplefrequency") and ch.text:
                    try:
                        fs_hz = float(ch.text.strip())
                    except ValueError:
                        pass
                elif ch_name in ("amplitudeunitsperbit", "leadamplitudeunitsperbit") and ch.text:
                    try:
                        scale = float(ch.text.strip())
                    except ValueError:
                        pass
                elif ch_name == "scale" and ch.attrib.get("value"):
                    try:
                        scale = float(ch.attrib["value"])
                    except ValueError:
                        pass
                elif ch_name in ("waveformdata", "digits") and ch.text:
                    samples_txt = ch.text
            if samples_txt:
                parsed_arr = _parse_numeric_samples(samples_txt, scale=scale)
                if parsed_arr is not None:
                    leads_uv[lead_id or f"LEAD_{len(leads_uv)+1}"] = parsed_arr

    # Fallback if bare <WaveformData>/<digits> exists without a container
    if not leads_uv:
        for el in root.iter():
            if el.tag.split("}")[-1].lower() in ("waveformdata", "digits") and el.text:
                parsed_arr = _parse_numeric_samples(el.text)
                if parsed_arr is not None:
                    leads_uv["II"] = parsed_arr
                    break

    primary_arr = leads_uv.get("II")
    if primary_arr is None:
        primary_arr = next(iter(leads_uv.values()), np.zeros(500, dtype=np.float32))
    signal_rms_uv = float(np.sqrt(np.mean(np.square(primary_arr)))) if primary_arr.size else 0.0

    multimodal_findings: List[Dict[str, Any]] = []
    found_codes: Set[str] = set()
    stmt_lower = statement.lower()

    if "atrial fibrillation" in stmt_lower or "afib" in stmt_lower or "irregularly irregular" in stmt_lower:
        multimodal_findings.append(
            {
                "concept_code": "CONCEPT_ECG_AFIB",
                "evidence_source": "ECG_XML_12LEAD_ANALYZER",
                "supporting_span": f"12-Lead ECG XML: {statement} (HR={hr_bpm} bpm, RMS={signal_rms_uv:.1f} uV)",
            }
        )
        found_codes.add("CONCEPT_ECG_AFIB")
    if "left ventricular hypertrophy" in stmt_lower or "lvh" in stmt_lower:
        multimodal_findings.append(
            {
                "concept_code": "CONCEPT_ECG_LVH",
                "evidence_source": "ECG_XML_12LEAD_ANALYZER",
                "supporting_span": f"12-Lead ECG XML: {statement} (QRS={qrs_ms} ms)",
            }
        )
        found_codes.add("CONCEPT_ECG_LVH")
    if qtc_ms >= 460.0 or "st depression" in stmt_lower or "ischemia" in stmt_lower or "t wave inversion" in stmt_lower:
        multimodal_findings.append(
            {
                "concept_code": "CONCEPT_ECG_ISCHEMIA",
                "evidence_source": "ECG_XML_12LEAD_ANALYZER",
                "supporting_span": f"12-Lead ECG XML: QTc={qtc_ms} ms; {statement}",
            }
        )
        found_codes.add("CONCEPT_ECG_ISCHEMIA")

    # 2. Raw Waveform DSP Fallback when XML InterpretationStatement omits rhythm/voltage labels
    min_rhythm_len = int(2.5 * fs_hz)
    if "CONCEPT_ECG_AFIB" not in found_codes:
        rhythm_sig = leads_uv.get("II")
        if rhythm_sig is None or rhythm_sig.size < min_rhythm_len:
            rhythm_sig = max(leads_uv.values(), key=lambda a: a.size, default=np.zeros(0, dtype=np.float32))
        if rhythm_sig.size >= min_rhythm_len:
            win = max(3, int(0.6 * fs_hz))
            sig_c = rhythm_sig.astype(np.float64) - uniform_filter1d(rhythm_sig.astype(np.float64), size=win, mode="nearest")
            min_dist = max(1, int(0.25 * fs_hz))
            pos_max = float(np.max(sig_c))
            peaks, _ = find_peaks(sig_c, distance=min_dist, prominence=max(200.0, 0.40 * pos_max))
            if len(peaks) < 5:
                abs_sig = np.abs(sig_c)
                peaks, _ = find_peaks(abs_sig, distance=min_dist, prominence=max(200.0, 0.40 * float(np.max(abs_sig))))
            if len(peaks) >= 5:
                rr_sec = np.diff(peaks).astype(np.float64) / fs_hz
                rr_valid = rr_sec[(rr_sec >= 0.25) & (rr_sec <= 2.2)]
                if rr_valid.size >= 4:
                    mean_rr = float(np.mean(rr_valid))
                    cv_rr = float(np.std(rr_valid) / max(mean_rr, 1e-6))
                    rmssd_ratio = float(np.sqrt(np.mean(np.diff(rr_valid) ** 2)) / max(mean_rr, 1e-6))
                    if cv_rr >= 0.18 and rmssd_ratio >= 0.18:
                        multimodal_findings.append(
                            {
                                "concept_code": "CONCEPT_ECG_AFIB",
                                "evidence_source": "ECG_DSP_RR_IRREGULARITY",
                                "supporting_span": (
                                    f"12-Lead ECG Waveform DSP: Irregularly irregular R-R intervals "
                                    f"(CV_RR={cv_rr:.2f}, RMSSD/RR={rmssd_ratio:.2f}, beats={len(peaks)})"
                                ),
                            }
                        )
                        found_codes.add("CONCEPT_ECG_AFIB")

    if "CONCEPT_ECG_LVH" not in found_codes and "V1" in leads_uv and ("V5" in leads_uv or "V6" in leads_uv):
        min_v_len = int(1.5 * fs_hz)
        v1 = leads_uv["V1"]
        if v1.size >= min_v_len:
            win = max(3, int(0.6 * fs_hz))
            min_dist = max(1, int(0.25 * fs_hz))
            v1_c = v1.astype(np.float64) - uniform_filter1d(v1.astype(np.float64), size=win, mode="nearest")
            s_troughs, _ = find_peaks(-v1_c, distance=min_dist, prominence=150.0)
            s_v1_uv = float(np.median(-v1_c[s_troughs])) if len(s_troughs) >= 2 else float(max(0.0, -np.min(v1_c)))

            r_lat_uv = 0.0
            for lat_name in ("V5", "V6"):
                if lat_name in leads_uv and leads_uv[lat_name].size >= min_v_len:
                    vx = leads_uv[lat_name].astype(np.float64)
                    vx_c = vx - uniform_filter1d(vx, size=win, mode="nearest")
                    r_pks, _ = find_peaks(vx_c, distance=min_dist, prominence=150.0)
                    r_val = float(np.median(vx_c[r_pks])) if len(r_pks) >= 2 else float(max(0.0, np.max(vx_c)))
                    r_lat_uv = max(r_lat_uv, r_val)

            sokolow_uv = s_v1_uv + r_lat_uv
            if sokolow_uv >= 3500.0:
                multimodal_findings.append(
                    {
                        "concept_code": "CONCEPT_ECG_LVH",
                        "evidence_source": "ECG_DSP_SOKOLOW_LYON",
                        "supporting_span": (
                            f"12-Lead ECG Waveform DSP: Sokolow-Lyon positive "
                            f"(|SV1|={s_v1_uv:.0f} uV + max(RV5,RV6)={r_lat_uv:.0f} uV = {sokolow_uv:.0f} uV >= 3500 uV)"
                        ),
                    }
                )
                found_codes.add("CONCEPT_ECG_LVH")

    # 3. Optional INT8 1D-CNN ONNX ECG Classifier Hook
    ecg_sess, ecg_cfg = _get_optional_ecg_onnx()
    if ecg_sess is not None and leads_uv:
        try:
            target_len = int(ecg_cfg.get("sequence_length", 5000))
            lead_order = ecg_cfg.get(
                "leads", ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            )
            tensor = np.zeros((1, len(lead_order), target_len), dtype=np.float32)
            for li, lname in enumerate(lead_order):
                arr_l = leads_uv.get(lname.upper())
                if arr_l is not None and arr_l.size > 0:
                    take = min(target_len, arr_l.size)
                    tensor[0, li, :take] = arr_l[:take]
            inp_name = ecg_sess.get_inputs()[0].name
            probs = np.asarray(ecg_sess.run(None, {inp_name: tensor})[0])[0]
            for idx_lbl, c_code in enumerate(ecg_cfg.get("label_concepts", [])):
                if idx_lbl < probs.size and float(probs[idx_lbl]) >= float(ecg_cfg.get("threshold", 0.7)) and c_code not in found_codes:
                    multimodal_findings.append(
                        {
                            "concept_code": c_code,
                            "evidence_source": "ECG_ONNX_1DCNN_INT8",
                            "supporting_span": f"12-Lead ECG 1D-CNN ONNX score={float(probs[idx_lbl]):.3f}",
                        }
                    )
                    found_codes.add(c_code)
        except Exception:
            pass

    synthesized_text = (
        f"12-Lead ECG Report ({visit_date}): Ventricular rate {hr_bpm:.0f} bpm, QRS {qrs_ms:.0f} ms, "
        f"QTc {qtc_ms:.0f} ms. Interpretation: {statement}"
    )

    return [
        {
            "source_file": file_path.name,
            "modality": "ECG_WAVEFORM_XML",
            "patient_id": patient_id,
            "visit_no": visit_no,
            "visit_date": visit_date,
            "visit_type_hint": "CARDIOLOGY_DIAGNOSTIC",
            "free_text": synthesized_text,
            "structured_fields": {"hr_bpm": hr_bpm, "qrs_ms": qrs_ms, "qtc_ms": qtc_ms},
            "multimodal_findings": multimodal_findings,
        }
    ]


def _format_dicom_date(raw_date: Any) -> str:
    """Converts DICOM DA date format YYYYMMDD to ISO YYYY-MM-DD."""
    s = str(raw_date or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s or "2026-01-15"


def _vessel_key_from_text(text: str) -> str:
    low = text.lower()
    if "left anterior descending" in low or re.search(r"\blad\b", low):
        return "LAD"
    if "left main" in low or re.search(r"\blm\b", low):
        return "LM"
    if "circumflex" in low or re.search(r"\blcx\b", low):
        return "LCx"
    if "right coronary" in low or re.search(r"\brca\b", low):
        return "RCA"
    return ""


def _walk_tid3902_sequence(
    seq: Any,
    vessel_scores: Dict[str, float],
    total_candidates: List[float],
    inherited_vessel: str = "",
) -> None:
    """Recursively walks multi-depth DICOM SR ContentSequence for TID 3902 total & vessel Agatston scores."""
    local_vessel = inherited_vessel
    # First pass over siblings to capture Finding Site / Target Region modifier codes
    for item in seq:
        cname_seq = getattr(item, "ConceptNameCodeSequence", None)
        cmeaning = str(getattr(cname_seq[0], "CodeMeaning", "")) if cname_seq and len(cname_seq) > 0 else ""
        ccode_seq = getattr(item, "ConceptCodeSequence", None)
        val_meaning = str(getattr(ccode_seq[0], "CodeMeaning", "")) if ccode_seq and len(ccode_seq) > 0 else ""
        tval = str(getattr(item, "TextValue", "") or "")
        v_hit = _vessel_key_from_text(f"{cmeaning} {val_meaning} {tval}")
        if v_hit and ("finding site" in cmeaning.lower() or "vessel" in cmeaning.lower() or "artery" in val_meaning.lower()):
            local_vessel = v_hit

    for item in seq:
        cname_seq = getattr(item, "ConceptNameCodeSequence", None)
        code_meaning = str(getattr(cname_seq[0], "CodeMeaning", "")).lower() if cname_seq and len(cname_seq) > 0 else ""
        item_text = str(getattr(item, "TextValue", "") or "").strip()
        item_vessel = _vessel_key_from_text(code_meaning) or local_vessel

        if "agatston" in code_meaning or "calcium" in code_meaning:
            measured_val: Optional[float] = None
            num_seq = getattr(item, "MeasuredValueSequence", None)
            if num_seq and len(num_seq) > 0 and hasattr(num_seq[0], "NumericValue"):
                try:
                    measured_val = float(num_seq[0].NumericValue)
                except (TypeError, ValueError):
                    pass
            elif item_text.replace(".", "", 1).isdigit():
                measured_val = float(item_text)

            if measured_val is not None:
                is_global = any(w in code_meaning for w in ("total", "global", "whole", "overall"))
                if item_vessel and not is_global:
                    vessel_scores[item_vessel] = measured_val
                else:
                    total_candidates.append(measured_val)

        child_seq = getattr(item, "ContentSequence", None)
        if child_seq:
            _walk_tid3902_sequence(child_seq, vessel_scores, total_candidates, inherited_vessel=item_vessel)


def iter_dicom_2d_slices_lazy(file_path: Path) -> Iterator[np.ndarray]:
    """
    Optional memory-safe 2D slice iterator for imaging ONNX models: yields one 2D frame `(H, W)`
    at a time so downstream callers never retain 3D volumes across files.
    """
    import pydicom  # type: ignore

    ds = pydicom.dcmread(str(file_path), force=True)
    if not hasattr(ds, "PixelData"):
        return
    arr = ds.pixel_array
    if arr.ndim == 2:
        yield np.asarray(arr)
    elif arr.ndim >= 3:
        for i in range(arr.shape[0]):
            yield np.asarray(arr[i])


def parse_dicom_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Reads DICOM (.dcm) headers and arbitrarily nested TID 3902 Coronary Calcium Structured Reports
    with stop_before_pixels=True (< 2 ms, zero pixel memory bloat), or parses .dcm.json sidecar
    representations when testing without binary DICOM datasets.
    """
    patient_id = "PAT_DCM_001"
    visit_no = f"DCM_{file_path.stem}"
    visit_date = "2026-02-10"
    study_desc = "Gated Non-Contrast Cardiac CT Calcium Score"
    modality = "DICOM_IMAGING_SR"
    institution = "Singapore Public Healthcare System"
    total_agatston = 0.0
    vessel_scores: Dict[str, float] = {}

    if file_path.name.endswith(".dcm.json"):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        patient_id = data.get("PatientID", patient_id)
        visit_no = data.get("AccessionNumber", visit_no)
        visit_date = _format_dicom_date(data.get("StudyDate", visit_date))
        study_desc = data.get("StudyDescription", study_desc)
        total_agatston = float(data.get("TotalAgatstonScore", 0.0))
        vessel_scores = data.get("VesselAgatstonScores", {})
        modality = f"DICOM_{data.get('Modality', 'SR')}"
    else:
        try:
            import pydicom  # type: ignore
            # Fast-path: read metadata, VR headers, and SR ContentSequence without loading raw pixels into RAM
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
            patient_id = str(getattr(ds, "PatientID", patient_id))
            visit_no = str(getattr(ds, "AccessionNumber", visit_no))
            visit_date = _format_dicom_date(getattr(ds, "StudyDate", visit_date))
            study_desc = str(getattr(ds, "StudyDescription", study_desc))
            institution = str(getattr(ds, "InstitutionName", institution))
            mod_code = str(getattr(ds, "Modality", "CT"))
            modality = f"DICOM_{mod_code}"

            # Check standard DICOM ImageComments
            if hasattr(ds, "ImageComments") and "agatston" in str(ds.ImageComments).lower():
                for tok in str(ds.ImageComments).split():
                    if tok.replace(".", "", 1).isdigit():
                        total_agatston = float(tok)
                        break

            # Recursively walk multi-depth TID 3902 Structured Report ContentSequence
            if hasattr(ds, "ContentSequence"):
                total_candidates: List[float] = []
                _walk_tid3902_sequence(ds.ContentSequence, vessel_scores, total_candidates)
                if total_candidates:
                    total_agatston = max(total_candidates)
                elif vessel_scores and total_agatston <= 0.0:
                    total_agatston = round(sum(vessel_scores.values()), 2)
        except Exception as e:
            study_desc = f"DICOM Header Read ({e})"

    multimodal_findings = []
    if total_agatston > 0:
        tier = "SEVERE (>=300)" if total_agatston >= 300 else ("MODERATE (100-299)" if total_agatston >= 100 else "MILD (1-99)")
        vessel_str = ", ".join(f"{k}={v}" for k, v in vessel_scores.items())
        multimodal_findings.append(
            {
                "concept_code": "CONCEPT_CAD_CALCIFICATION",
                "evidence_source": "DICOM_SR_TID3902_CALCIUM",
                "supporting_span": f"DICOM CT Calcium SR ({study_desc} @ {institution}): Total Agatston Score = {total_agatston} ({tier}) [{vessel_str}]",
                "extra": {"agatston_score": total_agatston, "vessel_agatston": vessel_scores, "agatston_risk_tier": tier},
            }
        )

    free_text = (
        f"DICOM Study: {study_desc}. Facility: {institution}. Total Agatston Coronary Calcium Score: {total_agatston}. "
        f"Vessel Breakdown: {json.dumps(vessel_scores)}"
    )

    return [
        {
            "source_file": file_path.name,
            "modality": modality,
            "patient_id": patient_id,
            "visit_no": visit_no,
            "visit_date": visit_date,
            "visit_type_hint": "RADIOLOGY_CT_CALCIUM",
            "free_text": free_text,
            "structured_fields": {"agatston_score": total_agatston},
            "multimodal_findings": multimodal_findings,
        }
    ]


SUPPORTED_SUFFIXES = (".csv", ".tsv", ".parquet", ".pdf", ".txt", ".xml", ".dcm", ".dcm.json")


def is_supported_clinical_file(path: Path) -> bool:
    if path.is_dir() or path.name.lower() == "gold_labels.csv":
        return False
    return path.name.lower().endswith(SUPPORTED_SUFFIXES)


def ingest_folder(input_dir: Path, only_files: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    for path in sorted(input_dir.iterdir()):
        if not is_supported_clinical_file(path):
            continue
        if only_files is not None and path.name not in only_files:
            continue
        name_low = path.name.lower()
        if name_low.endswith((".csv", ".tsv")):
            all_records.extend(parse_csv_or_tsv_table(path))
        elif name_low.endswith(".parquet"):
            all_records.extend(parse_parquet_table(path))
        elif name_low.endswith((".pdf", ".txt")):
            all_records.extend(parse_pdf_or_text_document(path))
        elif name_low.endswith(".xml"):
            all_records.extend(parse_ecg_waveform_xml(path))
        elif name_low.endswith((".dcm", ".dcm.json")):
            all_records.extend(parse_dicom_file(path))

    # Sort chronologically per patient_id so copy-paste deduplication is strictly causal
    all_records.sort(key=lambda r: (r["patient_id"], r["visit_date"], r["visit_no"]))
    return all_records
