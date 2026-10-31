# SG-MEDIATOR: Portable Multimodal Clinical Interlingua & Extraction CLI + Antigravity IDE Kit

SG-MEDIATOR is a zero-admin, zero-PowerShell, 100% offline-capable multimodal clinical NLP and ontology harmonization toolkit for **8 GB and 16 GB RAM CPU-only Windows workstations**, paired with an **Antigravity IDE** development and federated clinical context-expansion kit.

## Key Capabilities

1. **Dual-Head Hybrid Semantic Layer (`src/hybrid_semantic_layer.py`, `scripts/train_vq_codebook.py`)**:
   - **Head 1 (Discrete Biomedical Neural Interlingua, `K=16,384`)**: Encodes open-vocabulary clinical phrases via an INT8 quantized **SapBERT** transformer (`Xenova/SapBERT-from-PubMedBERT-fulltext`, UMLS self-aligned `[CLS]` representation), projects into a mean-centered 128-D spherical PCA space, and quantizes against 16,384 unit-norm centroids (`vq_codebook_16384.npz`) with canonical Singapore ontology prototype anchoring.
   - **Head 2 (Singapore Multi-Ontology Crosswalk)**: Maps concepts deterministically to `UMLS CUI`, `SNOMED CT-SG`, `ICD-10-AM`, `ACHI`, and **Singapore MOH TOSP (`1A`–`7C`)** surgical table tiers, including temporal **5th Universal Definition of Myocardial Infarction (2026)** and **4th UDMI (2018)** stratification (`Primary` / `Secondary` / `Procedural MI`, sex-specific hs-cTn URLs, `Primary` vs `Staged PCI`).
2. **Longitudinal Copy-Paste Deduplication & 6-Tier Clinical Sieve (`src/deduplicator.py`, `src/sieve_and_risk.py`)**:
   - 64-bit SimHash 8-word shingle alignment tracks `NEW_AT_VISIT` vs `CARRIED_FORWARD` clinical text chronologically per patient across multiprocess workers.
   - Extracts ConText assertions (`AFFIRMED`, `NEGATED`, `HISTORICAL`, `CONDITIONAL`), SI-unit lab/medication-derived chronic risks (`HbA1c`, `LDL-C`, `eGFR`, `uACR`, Asian BMI thresholds, Agatston CAC), pharmacogenomics (`HLA-B*15:02`, `HLA-B*58:01`, `CYP2C19`), and Singapore SDOH / Nutri-Grade / Healthier SG / CHAS indicators.
3. **Optional Tier-5 Local LLM Assertion Arbitrator (`src/tier5_llm_arbitrator.py`, `16gb` profile only)**:
   - Resolves hedged clinical mentions (`?NSTEMI`, `r/o ACS`, `likely`, `suspect`) via loopback Ollama (`127.0.0.1:11434`, `qwen2.5:3b-instruct-q4_K_M`, `OLLAMA_NUM_PARALLEL=1`) or `llama-cli.exe` with a 6-shot clinical prompt and a hard lexical negation-evidence guard (`0` false negations; automatic deterministic fallback when offline/stopped).
4. **Streaming Multiprocessing + DuckDB Analytics + HIB SHI Enclave (`src/cli.py`, `src/shi_enclave.py`)**:
   - Streams worker outputs into DuckDB (`sg_mediator.duckdb`: `ontology`, `concepts`, `visit_timelines`, and `patient_problem_list` longitudinal view) while isolating Singapore Health Information Bill (HIB) Sensitive Health Information (`HIV`, psychiatric, substance use, TOP, hereditary genetics) behind double-flag authorization and SHA-256 audit logs.

## Validated CPU Scale & Memory Ceilings

| Profile | Workers / Threads | DuckDB Limit | 15,000-Visit Wall Time | Measured Peak RSS | Target Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`8gb`** (Default) | `2` workers / `1` thread | `1536MB` | `8.9s` (~1,685 visits/s) | **`487 MB`** | `< 4,200 MB` |
| **`16gb`** (Offline Rules + SapBERT) | `4` workers / `2` threads | `4096MB` | `7.7s` (~1,948 visits/s) | **`508 MB`** | `< 8,500 MB` |
| **`16gb`** (+ Live Ollama `qwen2.5:3b`) | `4` workers / `2` threads | `4096MB` | `20.5s` (cached per worker) | **`503 MB`** Python (`~2.8 GB` total) | `< 8,500 MB` |

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
├── setup_tier5_ollama.bat              # Optional 16GB Tier-5 Portable Ollama (qwen2.5:3b-instruct-q4_K_M)
├── config/
│   ├── hardware_profiles.yaml          # Memory, worker, and DuckDB caps for 8GB and 16GB profiles
│   ├── sg_clinical_rules.yaml          # Singapore SI lab rules, HSA drugs, Nutri-Grade/SDOH, PGx, 5th UDMI, SHI
│   └── sg_ontology_seed.csv            # Crosswalk seed: VQ ID <-> UMLS, SNOMED-SG, ICD-10-AM, ACHI, MOH TOSP
├── scripts/
│   └── train_vq_codebook.py            # Reproducible SapBERT K=16,384 spherical VQ codebook trainer
├── src/
│   ├── cli.py                          # Multiprocessing + DuckDB streaming orchestrator
│   ├── ingest_multimodal.py            # Folder router for PDF, 12-lead ECG XML, DICOM, and CSV/TSV/Parquet
│   ├── deduplicator.py                 # Longitudinal 64-bit SimHash copy-paste detector & temporal resolver
│   ├── hybrid_semantic_layer.py        # INT8 SapBERT VQ Interlingua + Singapore ontology crosswalk
│   ├── sieve_and_risk.py               # 6-tier clinical NLP sieve + multi-source risk triangulation
│   ├── tier5_llm_arbitrator.py         # Optional loopback LLM assertion arbitrator with negation guard
│   ├── shi_enclave.py                  # Singapore HIB Sensitive Health Information SHA-256 enclave gate
│   └── benchmark_engine.py             # Local evaluator & PHI-redacted reprompt bundle exporter
├── tests/
│   └── test_hybrid_semantic_vq.py      # Zero-leakage unit test suite for SapBERT VQ codebook & fallback
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
