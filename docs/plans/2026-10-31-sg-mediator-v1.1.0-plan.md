# SG-MEDIATOR v1.1.0 Implementation Plan

> **For Agent:** REQUIRED SUB-SKILL: Use `executing-plans` and `test-driven-development` to implement this plan wave-by-wave, pausing at each Wave boundary to prompt the user to adjust **Gemini 4 Argon** reasoning effort.

**Goal:** Upgrade SG-MEDIATOR from `v1.0.0` to `v1.1.0` with scanned-PDF RapidOCR + multi-pass clinical section segmentation + Parquet ingestion, hardened Singapore PHI scrubbing + 120+ concept MOH TOSP/ICD-10-AM/SNOMED-SG ontology expansion, Tier-2.5 zero-shot SapBERT + RapidFuzz span linking, raw 12-lead ECG DSP + recursive DICOM TID 3902 SR + resumable DuckDB checkpoints, optional air-gapped Tier-5 `llama-cli` / ONNX hooks, and automated CI.

**Architecture:** Tasks are grouped into **3 Reasoning-Effort Waves (`Medium` -> `High` -> `Medium`)** calibrated by `@effort-harness-router` so the user only switches **Gemini 4 Argon** effort twice across the entire milestone. Every feature is built under strict RED -> GREEN -> REFACTOR TDD (`unittest`) within the bundled Python 3.11.9 Embeddable runtime (`runtime/python/python.exe`) with zero PowerShell or admin dependencies on target workstations.

**Tech Stack:** Python 3.11.9 Embeddable, DuckDB 2.0, ONNX Runtime 1.31 (`SapBERT` INT8 + `RapidOCR` INT8), PyMuPDF (`fitz`), `pydicom`, `scipy`, `rapidfuzz`, Portable Ollama / `llama-cli.exe` (`Qwen2.5-3B-Instruct-Q4_K_M`).

---

## Effort-Wave Schedule & Checkpoints

| Wave | Argon Effort | Included Items (Core + Optional) | Boundary Action |
| :--- | :--- | :--- | :--- |
| **Wave 1** | **`Medium`** *(Current)* | **Task W1.1 (Item 5):** Release asset cleanup + `.github/workflows/ci.yml`<br>**Task W1.2 (Item 2):** Multi-pass clinical section parser (`src/section_parser.py`), scanned PDF `RapidOCR` fallback, `.parquet` DuckDB ingestion<br>**Task W1.3 (Item 4):** Hardened Singapore PHI scrubber (`NRIC`, `MCR`, `Postal`, `Phone`, `Date`, `blake2b` IDs) + 120+ concept SG ontology expansion & codebook sync | **PAUSE & PROMPT:** Ask user to switch **Gemini 4 Argon** from `Medium` -> **`High`** |
| **Wave 2** | **`High`** | **Task W2.1 (Item 1 + Optional O1):** Tier-2.5 Neural Span Sieve (clinical chunker + `RapidFuzz` OCR recovery + calibrated SapBERT VQ zero-shot concept linker + optional `GLiNER-BioMed` INT8 ONNX slot)<br>**Task W2.2 (Item 3 + Optional O4):** Raw 12-lead ECG R-peak / RR-irregularity AFib & Sokolow-Lyon LVH DSP + recursive nested DICOM TID 3902 SR + crash-safe DuckDB `processed_files` SHA-256 checkpointing + optional 1D-CNN/pixel hook<br>**Task W2.3 (Optional O2 & O3):** Zero-socket `setup_tier5_llamacli.bat` + `pack_tier5_addon.bat` (`<2.0 GB` optional 16 GB LLM add-on packager) | **PAUSE & PROMPT:** Ask user to switch **Gemini 4 Argon** from `High` -> **`Medium`** |
| **Wave 3** | **`Medium`** | **Task W3.1:** Full regression suite (`unittest`, `benchmark_local.bat`, 15k-visit `8gb`/`16gb` RSS verification)<br>**Task W3.2:** Secret/PHI audit, rebuild `sg_mediator_portable_offline_bundle_x64.zip`, clean-room fresh-unzip offline test, git tag `v1.1.0`, and GitHub Release `v1.1.0` upload | Milestone Complete |

---

## Wave 1 Tasks (`Gemini 4 Argon` — `Medium` Effort)

### Task W1.1: GitHub Release Cleanup & Automated CI Workflow (Item 5)

**Files:**
- Create: `.github/workflows/ci.yml`
- Remote action: Delete legacy `60 KB` `sg_mediator_release_bundle.zip` from Release `v1.0.0`

**Steps:**
1. Create `.github/workflows/ci.yml` running on `windows-latest` with Python 3.11 to execute `python -m unittest discover -s tests -v` and `python src/benchmark_engine.py --input-dir sample_inbox --gold-file sample_inbox/gold_labels.csv --output-dir output_benchmark`.
2. Delete stale `sg_mediator_release_bundle.zip` from `v1.0.0` via `gh release delete-asset` and verify only `sg_mediator_portable_offline_bundle_x64.zip` (`315.5 MB`) remains on `v1.0.0`.

---

### Task W1.2: Scanned PDF RapidOCR + Multi-Pass Section Parser + Parquet Ingestion (Item 2)

**Skills:** `@cardio-section-sieve-architect`, `@sg-multimodal-dicom-ecg`, `@test-driven-development`

**Files:**
- Create: `src/section_parser.py`
- Modify: `src/ingest_multimodal.py`
- Modify: `src/deduplicator.py`
- Test: `tests/test_multimodal_and_sections.py`

**Steps:**
1. **RED:** Write `tests/test_multimodal_and_sections.py` covering:
   - Page-1 executive `CONCLUSION:` / `IMPRESSION AND PLAN:` above `PROCEDURAL DETAILS`, inline/Roman/pipe-table headers, `CONCLUSION` + `PLAN` merging, and `conclusion_tail_fallback`.
   - Image-only scanned PDF generated via `fitz` (`0` native text chars) triggering `RapidOCR` fallback and extracting clinical text cleanly.
   - Zero-copy `.parquet` clinical visit table ingestion via `duckdb` inside `ingest_folder()`.
2. **Verify RED:** Run `.\runtime\python\python.exe -m unittest tests/test_multimodal_and_sections.py -v` and confirm expected failures (`ModuleNotFoundError: No module named 'section_parser'`).
3. **GREEN:** Implement `src/section_parser.py` and update `src/ingest_multimodal.py` + `src/deduplicator.py` (lazy singleton `RapidOCR` per worker only when a PDF page has `< 40` native chars and embedded images/drawings, explicit `doc.close()`, and DuckDB `.parquet` reader).
4. **Verify GREEN:** Re-run `tests/test_multimodal_and_sections.py` and `benchmark_local.bat` (`41/41` TP, `0` FP, `0` FN).

---

### Task W1.3: Hardened Singapore PHI Scrubber & 120+ Concept Multi-Ontology Expansion (Item 4)

**Skills:** `@sg-local-benchmark-reprompter`, `@sg-mediator-architect`, `@test-driven-development`

**Files:**
- Modify: `src/benchmark_engine.py`
- Modify: `config/sg_ontology_seed.csv`
- Modify: `models_onnx_int8/sapbert_pubmedbert_int8/vq_codebook_16384.npz` (via `scripts/train_vq_codebook.py`)
- Test: `tests/test_phi_and_ontology.py`

**Steps:**
1. **RED:** Write `tests/test_phi_and_ontology.py` asserting:
   - `scrub_phi_text` redacts SG NRIC/FIN (`[STFGM]\d{7}[A-Z]`), SG phone numbers (`+65 9123 4567` / `8123-4567`), SMC MCR numbers (`M12345A`), SG 6-digit postal codes (`Singapore 169608`, `S(560123)`), ISO/slash/named dates, and emails.
   - `anonymize_patient_id` uses deterministic `blake2b` hashing (stable across processes regardless of `PYTHONHASHSEED`).
   - `config/sg_ontology_seed.csv` contains `>= 120` unique Singapore clinical concepts with distinct `VQ_xxxxx` IDs in `[0, 16383]` and populated crosswalk fields.
2. **Verify RED:** Run `.\runtime\python\python.exe -m unittest tests/test_phi_and_ontology.py -v` and confirm failure.
3. **GREEN:** Update `src/benchmark_engine.py`, expand `config/sg_ontology_seed.csv` to 120+ concepts across MOH TOSP `1A`–`7C` / ICD-10-AM / SNOMED-SG / HSA SDL, and re-run `scripts/train_vq_codebook.py` so all 120+ seeded prototypes are anchored in `vq_codebook_16384.npz`.
4. **Verify GREEN:** Run full `unittest discover -s tests -v` and `benchmark_local.bat` (`41/41` TP, `0` FP, `0` FN).

---

## Wave 2 Tasks (`Gemini 4 Argon` — `High` Effort)

### Task W2.1: Tier-2.5 Neural Span Sieve + RapidFuzz OCR Recovery + Optional GLiNER-BioMed Slot (Item 1 + O1)
- Create `tests/test_tier25_neural_sieve.py` (RED -> GREEN).
- Add clinical candidate span chunker + `RapidFuzz` OCR typo recovery (`stged PCl` -> `CONCEPT_PCI_STAGED`) + calibrated SapBERT anchored-centroid cosine matcher (`cos >= 0.68`) + optional `GLiNER-BioMed` INT8 ONNX hook (`models_onnx_int8/gliner_biomed_int8/`) in `src/sieve_and_risk.py` and `src/hybrid_semantic_layer.py`.

### Task W2.2: Raw 12-Lead ECG DSP + Recursive DICOM TID 3902 SR + Crash-Safe DuckDB Checkpoints (Item 3 + O4)
- Create `tests/test_ecg_dsp_dicom_and_checkpoints.py` (RED -> GREEN).
- Implement R-peak detection + RR-interval CV irregularity (`AFib`) + Sokolow-Lyon voltage (`SV1 + RV5/V6 >= 35 mm` -> `LVH`) in `parse_ecg_waveform_xml`, recursive multi-depth `ContentSequence` traversal in `parse_dicom_file` (preserving `stop_before_pixels=True` + optional lazy slice hook), and resumable `processed_files` SHA-256 checkpointing in `src/cli.py` (`--resume`).

### Task W2.3: Zero-Socket `llama-cli.exe` Setup & Optional Tier-5 Add-On Packager (Optional O2 & O3)
- Create `setup_tier5_llamacli.bat` and `pack_tier5_addon.bat` (pure `cmd.exe`) + unit tests verifying `Tier5Arbitrator` fallback order (`ollama` -> `llama-cli` -> deterministic rules).

---

## Wave 3 Tasks (`Gemini 4 Argon` — `Medium` Effort)

### Task W3.1 & W3.2: Full Verification, Portable Bundle Rebuild, Clean-Room Unzip Test & `v1.1.0` Release
- Run all unit tests, `benchmark_local.bat`, 15,000-visit `8gb`/`16gb` scale & RSS checks, secret/PHI scan, rebuild `dist/sg_mediator_portable_offline_bundle_x64.zip`, verify from fresh offline unzip, commit, tag `v1.1.0`, and upload to GitHub Releases.
