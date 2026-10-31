# SG-MEDIATOR: Portable Multimodal Clinical Interlingua & Extraction CLI + Antigravity IDE Kit

SG-MEDIATOR is a zero-admin, zero-PowerShell, 100% offline-capable multimodal clinical NLP and ontology harmonization toolkit for **8 GB and 16 GB RAM CPU-only Windows workstations**, paired with an **Antigravity IDE** development and federated clinical context-expansion kit.

## Key Capabilities (`v1.1.0`)

1. **Dual-Head Hybrid Semantic Layer (`src/hybrid_semantic_layer.py`, `scripts/train_vq_codebook.py`)**:
   - **Head 1 (Discrete Biomedical Neural Interlingua, `K=16,384`)**: Encodes open-vocabulary clinical phrases via an INT8 quantized **SapBERT** transformer (`Xenova/SapBERT-from-PubMedBERT-fulltext`, UMLS self-aligned `[CLS]` representation), projects into a mean-centered 128-D spherical PCA space, and quantizes against 16,384 Voronoi-carved unit-norm centroids (`vq_codebook_16384.npz`) anchored to **128 Singapore ontology prototypes**.
   - **Head 2 (128-Concept Singapore Multi-Ontology Crosswalk)**: Maps concepts deterministically and zero-shot across **128 unique clinical concepts** (including **37 MOH TOSP `1A`–`7C` procedures**) to `UMLS CUI`, `SNOMED CT-SG`, `ICD-10-AM`, `ACHI`, and **Singapore MOH TOSP**, including temporal **5th Universal Definition of Myocardial Infarction (2026)** and **4th UDMI (2018)** stratification (`Primary` / `Secondary` / `Procedural MI`, sex-specific hs-cTn URLs, `Primary` vs `Staged PCI`).
2. **Multi-Pass Clinical Section Parser, Longitudinal Deduplication & 6-Tier Sieve (`src/section_parser.py`, `src/deduplicator.py`, `src/sieve_and_risk.py`)**:
   - **Multi-Pass Section Parser**: Captures Page-1 executive `CONCLUSION` / `IMPRESSION AND PLAN` blocks above `PROCEDURAL DETAILS`, inline/Roman/pipe-table headers, merged `CONCLUSION` + `PLAN` windows, and 40% `CONCLUSION_TAIL_FALLBACK`.
   - **64-bit SimHash Deduplication**: 8-word shingle alignment tracks `NEW_AT_VISIT` vs `CARRIED_FORWARD` clinical text chronologically per patient across multiprocess workers.
   - **Tier-1 Aho-Corasick + Tier-2.5 Neural Span Sieve**: Single-pass C-level `pyahocorasick` automaton, `RapidFuzz` OCR typo recovery (`stged PCl`, `essential hypertensn` with strict digit-preservation guards), zero-shot `SapBERT` anchored-centroid span linking (`cos >= 0.68` with genomics/allele safety guards), and optional `GLiNER-BioMed` INT8 ONNX span extraction (`fetch_optional_gliner.bat`).
3. **Multimodal PDF/RapidOCR, 12-Lead ECG DSP, Recursive DICOM SR & Parquet Ingestion (`src/ingest_multimodal.py`)**:
   - **Scanned PDF Hybrid OCR**: Extracts native PDF text via `PyMuPDF` (`fitz`) first and triggers per-worker `RapidOCR` INT8 ONNX fallback only on scanned pages (`< 40` native chars with images/drawings).
   - **Raw 12-Lead ECG Waveform DSP**: Multi-lead HL7 aECG / Philips / GE MUSE decoding with R-peak detection + RR-interval irregularity (`CV_RR` & `RMSSD/RR`) for AFib, Sokolow-Lyon voltage ($\lvert S_{V1}\rvert + \max(R_{V5}, R_{V6}) \ge 3500\text{ }\mu\text{V}$) for LVH when XML `InterpretationStatement` is blank, plus optional INT8 1D-CNN ONNX hook.
   - **Recursive DICOM `TID 3902` SR & Zero-Copy `.parquet`**: Arbitrary-depth `ContentSequence` traversal for total & vessel (`LM`, `LAD`, `LCx`, `RCA`) Agatston calcium scores with `stop_before_pixels=True`, optional lazy 2D slice iterator (`iter_dicom_2d_slices_lazy`), and DuckDB `.parquet` table ingestion.
4. **Optional Tier-5 Local LLM Assertion Arbitrator (`src/tier5_llm_arbitrator.py`, `16gb` profile only)**:
   - Resolves hedged clinical mentions (`?NSTEMI`, `r/o ACS`, `likely`, `suspect`) via loopback Ollama (`127.0.0.1:11434`, `qwen2.5:3b-instruct-q4_K_M`, `OLLAMA_NUM_PARALLEL=1`) or zero-socket `llama-cli.exe` (`setup_tier5_llamacli.bat`) with a 6-shot clinical prompt, strict loopback host enforcement, and lexical negation-evidence guard (`pack_tier5_addon.bat` builds an optional `< 2.0 GB` add-on zip).
5. **Resumable DuckDB Streaming + Hardened SG PHI Scrubber + HIB SHI Enclave (`src/cli.py`, `src/shi_enclave.py`, `src/benchmark_engine.py`)**:
   - Streams worker outputs into DuckDB (`sg_mediator.duckdb`) with crash-safe SHA-256 `processed_files` checkpointing (`--resume`), deterministic `blake2b` patient ID pseudonymization, Singapore PHI scrubbing (`NRIC/FIN`, SMC `MCR`, SG `Postal`, `+65 Phone`, `Date`, `Email`), and HIB Sensitive Health Information enclave gating.

## Validated CPU Scale & Memory Ceilings

| Profile | Workers / Threads | DuckDB Limit | 15,000-Visit Wall Time | Measured Peak RSS | Target Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`8gb`** (Default, Full Tier-1 + Tier-2.5 SapBERT/RapidFuzz) | `2` workers / `1` thread | `1536MB` | `17.5s` (~857 visits/s) | **`685 MB`** | `< 4,200 MB` |
| **`16gb`** (Offline Rules + Tier-2.5 SapBERT/RapidFuzz) | `4` workers / `2` threads | `4096MB` | `14.5s` (~1,034 visits/s) | **`1,196 MB`** | `< 8,500 MB` |
| **`16gb`** (+ Live Ollama `qwen2.5:3b`) | `4` workers / `2` threads | `4096MB` | `~26s` (cached per worker) | **`~1.2 GB`** Python (`~3.5 GB` total) | `< 8,500 MB` |

Gold benchmark (`benchmark_local.bat`): **`41/41` TP, `0` FP, `0` FN (`F1 = 1.0`, Copy-Paste Provenance Accuracy `1.0`)**.

## Directory Layout

```text
sg_mediator/
├── ANTIGRAVITY_MASTER_PROMPT.md        # Master prompt for Antigravity IDE context expansion
├── README.md                           # Quickstart & architecture guide
├── run_sg_mediator.bat                 # Pure cmd.exe CLI launcher (zero PowerShell, zero Admin)
├── benchmark_local.bat                 # Pure cmd.exe local benchmark & PHI-scrubbed reprompt generator
├── build_portable_bundle.bat           # Dev PC script: bundles Python 3.11 Embeddable + wheels + models
├── fetch_models.bat                    # Downloads INT8 ONNX SapBERT & trains K=16,384 VQ codebook
├── fetch_optional_gliner.bat           # Optional GLiNER-BioMed INT8 ONNX span extractor setup
├── setup_tier5_ollama.bat              # Optional 16GB Tier-5 Portable Ollama (qwen2.5:3b-instruct-q4_K_M)
├── setup_tier5_llamacli.bat            # Optional 16GB zero-socket llama-cli.exe + GGUF setup
├── pack_tier5_addon.bat                # Packages optional Tier-5 LLM binaries/weights (<2.0 GB zip)
├── .github/workflows/ci.yml            # Automated Windows CPU unit test & F1=1.0 benchmark CI
├── config/
│   ├── hardware_profiles.yaml          # Memory, worker, and DuckDB caps for 8GB and 16GB profiles
│   ├── sg_clinical_rules.yaml          # Singapore SI lab rules, HSA drugs, Nutri-Grade/SDOH, PGx, 5th UDMI, SHI
│   └── sg_ontology_seed.csv            # 128-concept crosswalk: VQ ID <-> UMLS, SNOMED-SG, ICD-10-AM, ACHI, MOH TOSP
├── scripts/
│   └── train_vq_codebook.py            # Reproducible SapBERT K=16,384 spherical VQ codebook trainer
├── src/
│   ├── cli.py                          # Multiprocessing + DuckDB streaming & SHA-256 --resume orchestrator
│   ├── section_parser.py               # Multi-pass clinical report section & Page-1 conclusion parser
│   ├── ingest_multimodal.py            # Folder router for PDF/RapidOCR, 12-lead ECG DSP, recursive DICOM SR, CSV/Parquet
│   ├── deduplicator.py                 # Longitudinal 64-bit SimHash copy-paste detector & temporal resolver
│   ├── hybrid_semantic_layer.py        # INT8 SapBERT VQ Interlingua + optional GLiNER slot + SG crosswalk
│   ├── sieve_and_risk.py               # 6-tier Aho-Corasick + RapidFuzz + SapBERT clinical NLP sieve
│   ├── tier5_llm_arbitrator.py         # Optional loopback/zero-socket LLM arbitrator & add-on packager
│   ├── shi_enclave.py                  # Singapore HIB Sensitive Health Information SHA-256 enclave gate
│   └── benchmark_engine.py             # Local evaluator & hardened SG PHI-redacted reprompt exporter
├── tests/                              # 26-test TDD suite (VQ, sections/OCR/Parquet, PHI/TOSP, Tier-2.5, ECG/DICOM/resume, Tier-5)
├── antigravity_kit/
│   ├── mcp_servers/
│   │   ├── mcp_config.json             # MCP server configuration for Antigravity IDE
│   │   └── sg_clinical_router_mcp.py   # Federated Clinical Router MCP (5th/4th UDMI, MOH TOSP, PubMed, UMLS)
│   └── skills/                         # Antigravity workspace skills
└── sample_inbox/                       # Synthetic multimodal test cases + gold_labels.csv
```

## Quickstart on Offline Target Workstation (`cmd.exe`, Zero Admin, Zero PowerShell)

1. Extract `sg_mediator_portable_offline_bundle_x64.zip` to any user-writable folder (e.g., `C:\Users\<username>\sg_mediator\`).
2. Place clinical PDFs, 12-lead ECG XMLs, DICOM files, and CSV/TSV/Parquet tables into an input folder (e.g., `sample_inbox`).
3. Run in **8 GB RAM mode** (default) or **16 GB RAM mode**:
   ```cmd
   run_sg_mediator.bat sample_inbox output_results 8gb
   run_sg_mediator.bat sample_inbox output_results 16gb
   ```
4. Verify local extraction accuracy and generate a PHI-scrubbed discrepancy bundle:
   ```cmd
   benchmark_local.bat sample_inbox sample_inbox\gold_labels.csv output_benchmark
   ```
5. *(Optional, 16 GB workstations)* To enable Tier-5 local LLM arbitration on hedged sentences, run `setup_tier5_ollama.bat` once on an internet-connected PC before copying the bundle, then launch loopback Ollama with `OLLAMA_NUM_PARALLEL=1` prior to running `16gb` mode.
