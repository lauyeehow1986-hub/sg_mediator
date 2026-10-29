---
name: sg-multimodal-dicom-ecg
description: Low-memory CPU multimodal ingestion for 12-lead ECG waveform XMLs (HL7 aECG, Philips, GE MUSE), DICOM headers and TID 3902 Coronary Calcium Agatston Structured Reports, scanned/native clinical PDFs, and CSV/Parquet tables.
---
# Multimodal DICOM, ECG XML, PDF & Table Ingestion Skill

## When to Use
Use this skill when extending `src/ingest_multimodal.py` or adding INT8 ONNX models (`models_onnx_int8/`) for ECG waveforms or DICOM imaging slices.

## Memory-Safe Multimodal Rules
1. **DICOM Fast-Path (`stop_before_pixels=True`)**: Always read DICOM headers and `TID 3902` Coronary Calcium Structured Reports without loading pixel arrays into RAM. Only stream 2D slices lazily when pixel inference is required.
2. **12-Lead ECG XML Streaming**: Parse HL7 aECG / Philips / GE MUSE XMLs via `xml.etree.ElementTree` or `lxml.etree.iterparse`, extracting `HeartRate`, `PR`, `QRS`, `QTc`, and Lead II/V1-V6 waveform samples.
3. **PDF Hybrid Parser**: Extract native text/tables via `PyMuPDF` (`fitz`) first; invoke `rapidocr-onnxruntime` only on scanned pages with `< 40` characters.
