# SG-MEDIATOR: Portable Multimodal Clinical Interlingua & Extraction CLI + Antigravity IDE Kit

This archive contains both:
1. **The Portable Zero-Admin, No-PowerShell CLI (`run_sg_mediator.bat`, `benchmark_local.bat`, `src/`, `config/`)**: Designed to run on locked-down **8 GB RAM or 16 GB RAM CPU-only Windows workstations** without administrator privileges or PowerShell.
2. **The Antigravity IDE Development Kit (`ANTIGRAVITY_MASTER_PROMPT.md`, `antigravity_kit/skills/`, `antigravity_kit/mcp_servers/`)**: Designed for your development workstation running **Antigravity IDE with Gemini 3.8 Flash (High Effort)** to expand clinical context via MCP routing and iteratively patch edge cases from local benchmark logs.

## Directory Layout

```text
sg_mediator_kit/
├── ANTIGRAVITY_MASTER_PROMPT.md        # Copy-paste Master Prompt for Antigravity IDE (Gemini 3.8 Flash High Effort)
├── README.md                           # Quickstart & architecture guide
├── run_sg_mediator.bat                 # Pure cmd.exe CLI launcher (NO PowerShell, NO Admin)
├── benchmark_local.bat                 # Pure cmd.exe local benchmark & PHI-scrubbed reprompt bundle generator
├── build_portable_bundle.bat           # Pure cmd.exe script on Dev PC to bundle Python 3.11 Embeddable + wheels
├── config/
│   ├── hardware_profiles.yaml          # Memory & thread caps for 8GB RAM (<4.2GB peak) and 16GB RAM (<8.5GB peak)
│   ├── sg_clinical_rules.yaml          # Singapore SI lab rules, HSA drugs, Nutri-Grade/SDOH, PGx, ConText, SHI
│   └── sg_ontology_seed.csv            # Crosswalk seed: VQ Codebook ID <-> UMLS, SNOMED-SG, ICD-10-AM, ACHI, MOH TOSP
├── src/
│   ├── cli.py                          # Main CLI entry point & memory-guarded orchestrator
│   ├── ingest_multimodal.py            # Folder router for PDF, ECG XML, DICOM (.dcm), and CSV/TSV/Parquet tables
│   ├── deduplicator.py                 # Longitudinal 64-bit SimHash copy-paste detector & temporal anchor resolver
│   ├── hybrid_semantic_layer.py        # MEDIATOR VQ Interlingua Tokenizer + UMLS/SNOMED/ICD10AM/ACHI/TOSP crosswalk
│   ├── sieve_and_risk.py               # 6-tier clinical NLP sieve + triangulated history/lab/drug/Agatston risk engine
│   ├── shi_enclave.py                  # Singapore Health Information Bill (HIB) Sensitive Health Information gate
│   └── benchmark_engine.py             # Local evaluator & PHI-redacted antigravity_reprompt_bundle.json exporter
├── antigravity_kit/
│   ├── mcp_servers/
│   │   ├── mcp_config.json             # Ready-to-load MCP configuration for Antigravity IDE
│   │   └── sg_clinical_router_mcp.py   # Federated Context-Expansion Router MCP (ACE, MOH TOSP, PubMed, ClinicalTrials, UMLS, PharmGKB)
│   └── skills/
│       ├── sg-mediator-architect/SKILL.md
│       ├── sg-clinical-dedup-and-sieve/SKILL.md
│       ├── sg-multimodal-dicom-ecg/SKILL.md
│       └── sg-local-benchmark-reprompter/SKILL.md
└── sample_inbox/
    ├── visits_notes_and_labs.csv       # Longitudinal polyclinic/SOC/ward table with copy-paste & SI labs
    ├── discharge_summary_note.txt      # Clinical narrative sample
    ├── ecg_12lead_philips_sample.xml   # HL7/Philips 12-lead ECG waveform & interval XML
    ├── ct_calcium_dicom_sr_sample.dcm.json # DICOM Header + TID 3902 Coronary Calcium Agatston Structured Report
    └── gold_labels.csv                 # Ground-truth benchmark labels for local evaluation
```

## Quickstart on Target Machine (Windows `cmd.exe`, Zero Admin, Zero PowerShell)

1. Extract this folder to any user-writable path (e.g., `C:\Users\<username>\sg_mediator\`).
2. Drop your PDFs, ECG XMLs, DICOMs (`.dcm`), and CSV/Parquet tables into an input folder (e.g., `sample_inbox`).
3. Run the CLI in **8 GB RAM mode** (default) or **16 GB RAM mode**:
   ```cmd
   run_sg_mediator.bat sample_inbox output_results --profile 8gb
   ```
4. Run the local benchmark and generate a PHI-scrubbed edge-case bundle for Antigravity IDE:
   ```cmd
   benchmark_local.bat sample_inbox sample_inbox\gold_labels.csv output_benchmark
   ```
5. Copy `output_benchmark\antigravity_reprompt_bundle.json` back to your Antigravity IDE workstation and ask Gemini 3.8 Flash to synthesize a rule/codebook patch!
