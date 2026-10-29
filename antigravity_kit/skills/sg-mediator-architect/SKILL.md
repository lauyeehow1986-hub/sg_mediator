---
name: sg-mediator-architect
description: Core architectural skill for building and maintaining SG-MEDIATOR (Singapore adaptation of MahmoodLab's MEDIATOR Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling) as a zero-admin, no-PowerShell portable CLI for 8GB-16GB RAM CPU workstations.
---
# SG-MEDIATOR Architect Skill

## When to Use
Use this skill whenever designing, extending, or packaging the SG-MEDIATOR CLI, VQ codebook (`[K=16384]`), or Singapore multi-ontology crosswalk (`UMLS`, `SNOMED CT-SG`, `ICD-10-AM`, `ACHI`, `MOH TOSP`).

## Key Rules
1. **Zero PowerShell / Zero Admin Packaging**: Never generate `.ps1` scripts or PyInstaller `--onefile` binaries. Always maintain `run_sg_mediator.bat` and `benchmark_local.bat` using pure `cmd.exe` syntax and the Python 3.11 Windows Embeddable layout (`runtime/python/`).
2. **8 GB vs 16 GB RAM Profiles**:
   - `8gb` profile: `OMP_NUM_THREADS=1`, `ORT_INTRA_OP_NUM_THREADS=1`, 2 workers, DuckDB `1536MB`.
   - `16gb` profile: `OMP_NUM_THREADS=2`, `ORT_INTRA_OP_NUM_THREADS=2`, 4 workers, DuckDB `4096MB`.
3. **Dual-Head Hybrid Semantic Output**: Every extracted clinical event must emit both a discrete VQ token (`vq_token_id`) and deterministic crosswalk fields (`umls_cui`, `snomed_sg_code`, `icd10am_code`, `achi_code`, `moh_tosp_code`, `tosp_table_tier`).
