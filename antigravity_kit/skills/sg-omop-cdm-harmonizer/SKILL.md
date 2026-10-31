---
name: sg-omop-cdm-harmonizer
description: Use when converting SG-MEDIATOR clinical outputs to OHDSI OMOP CDM v5.4, querying OMOPHub API or MCP for standard concept IDs, or mapping Singapore SNOMED-SG, ICD-10-AM, ACHI, MOH TOSP, Nutri-Grade, or Healthier SG codes - enforces assertion-safe domain routing, copy-paste deduplication, HIB SHI enclave gating, and zero-core-mutation skill execution
---

# SG-MEDIATOR OHDSI OMOP CDM v5.4 Harmonizer

## Overview
Harmonizes SG-MEDIATOR CSV or DuckDB outputs into compliant **OHDSI OMOP CDM v5.4** tables and resolves new vocabularies via **OMOPHub REST v1 / MCP** entirely inside this skill folder — **zero changes to `src/` or offline Zone-A runtime bundles**.

**Violating the letter of these rules is violating the spirit of clinical cohort integrity and Singapore HIB/PDPA compliance.**

## When to Use
- Exporting `sg_mediator_extracted_concepts.csv` or `sg_mediator.duckdb` into OHDSI OMOP CDM v5.4 tables (`person`, `condition_occurrence`, `procedure_occurrence`, `drug_exposure`, `measurement`, `observation`, `note_nlp`, `source_to_concept_map`).
- Looking up standard OHDSI `concept_id`s or `Maps to` crosswalks via OMOPHub API (`https://api.omophub.com/v1`) or `@omophub/omophub-mcp` in Zone B (Antigravity IDE).
- Mapping Singapore-specific codes (`MOH TOSP` `1A`–`7C`, `Nutri-Grade`, `Healthier SG`, `CHAS`) into OMOP.

## Non-Negotiable Rules
1. **Zero Core CLI Mutation**: NEVER import `omophub`, add HTTP calls, or hardcode `OMOPHUB_API_KEY` inside `src/` or tracked config files. Read `OMOPHUB_API_KEY` strictly from `os.environ`.
2. **Never Cast Source Codes as `*_concept_id`**: SNOMED SCTIDs (`44054006`, `7181000119102`) are string `concept_code`s, NOT OMOP integer `concept_id`s (`201826`). Always resolve via `resources/sg_omop_cdm_crosswalk.csv` or `scripts/omophub_bridge.py`.
3. **Outbound DLP Guard**: NEVER send `supporting_span`, patient IDs, NRIC/FIN, SMC MCR, or `+65` phone numbers to `api.omophub.com`. Only send de-identified vocabulary codes or `canonical_name`.
4. **Copy-Paste & Planned-Procedure Protection**:
   - Exclude `provenance == 'COPY_PASTE'` from clinical occurrence tables by default (prevents multi-day copy-paste notes from inflating surgical/condition counts); log all mentions in `omop_note_nlp`.
   - Route planned `CONCEPT_PCI_STAGED` (`SD701H`) to `omop_observation` (`observation_concept_id = 4275495` *Planned procedure*, `value_as_concept_id = 4203772`), never as an already-performed `procedure_occurrence`.
5. **Singapore HIB SHI Gate**: Drop `SHI_Restricted` (`CONCEPT_SHI_*`) rows from all OMOP tables by default unless explicitly unlocked with both `--include-shi` and non-empty `--shi-reason`. Blank `omop_note_nlp.snippet` (`""`) by default unless `--include-snippets` is passed inside an authorized enclave.

## Assertion & Schema Routing Reference (`*_type_concept_id = 32858`)

| `assertion_status` | `provenance` | Clinical Domain Table (`condition` / `procedure` / `drug` / `measurement` / `observation`) | `omop_note_nlp` (`note_nlp_concept_id`, `"offset"` quoted in DuckDB) |
|---|---|---|---|
| `AFFIRMED` | `NEW` | Insert into target `target_cdm_table` from `sg_omop_cdm_crosswalk.csv` | `term_exists = 'Y'`, `term_temporal = 'present'` |
| `AFFIRMED` | `COPY_PASTE` | **Suppress** (unless `--include-copy-paste`) | `term_exists = 'Y'`, `term_modifiers` includes `provenance=COPY_PASTE` |
| `HISTORICAL` | `NEW` | Route to `omop_observation` (`observation_concept_id = 4214956`, `value_as_concept_id = target_id`) | `term_exists = 'Y'`, `term_temporal = 'historical'` |
| `FAMILY` | `NEW` | Route to `omop_observation` (`observation_concept_id = 4167217`, `value_as_concept_id = target_id`) | `term_exists = 'Y'`, `term_modifiers` includes `assertion=FAMILY` |
| `NEGATED` | Any | **NEVER insert** (no custom `status` columns!) | `term_exists = 'N'` (`omop_note_nlp` ONLY) |
| `CONDITIONAL` / `UNCERTAIN` | Any | **NEVER insert** | `term_exists = 'U'` (`omop_note_nlp` ONLY) |

## Quick Commands (`scripts/omophub_bridge.py`)

```cmd
REM 1. Offline 128-concept export (CSV or sg_mediator.duckdb -> OMOP CDM v5.4 DuckDB)
runtime\python\python.exe antigravity_kit\skills\sg-omop-cdm-harmonizer\scripts\omophub_bridge.py export --input output\sg_mediator_extracted_concepts.csv --output-duckdb output\omop_cdm.duckdb

REM 2. Zone-B live OMOPHub lookup / mapping (requires set OMOPHUB_API_KEY=oh_...)
runtime\python\python.exe antigravity_kit\skills\sg-omop-cdm-harmonizer\scripts\omophub_bridge.py by-code --vocabulary SNOMED --code 44054006
runtime\python\python.exe antigravity_kit\skills\sg-omop-cdm-harmonizer\scripts\omophub_bridge.py map-code --source-vocabulary ICD10CM --source-code E11.9 --target-vocabulary SNOMED
runtime\python\python.exe antigravity_kit\skills\sg-omop-cdm-harmonizer\scripts\omophub_bridge.py search --query "Type 2 diabetes mellitus" --vocabularies SNOMED,RxNorm,LOINC
```

Optional Zone-B MCP servers (`antigravity_kit/mcp_servers/mcp_config.json` & `~/.gemini/config/mcp_config.json`):
- **`sg-clinical-router-mcp` (`C:/Python314/python.exe`)**: Exposes `resolve_sg_omop_concept` (128-concept offline Singapore TOSP/SNOMED/SDOH crosswalk + DLP-guarded live OMOPHub lookup), `export_sg_mediator_to_omop_cdm` (compliant OMOP CDM v5.4 DuckDB builder), and OMOP-enriched `evaluate_ami_pci_stratification`.
- **`omophub` (`npx -y @omophub/omophub-mcp`)**: Official OMOPHub MCP server for full OHDSI hierarchy/ancestor traversal and FHIR terminology translation when `OMOPHUB_API_KEY` is set in environment (never pass raw clinical spans or PHI to `omophub` MCP tools).

## Rationalization Counter-Table

| Excuse | Reality |
|---|---|
| "SNOMED is standard in OMOP, so cast `snomed_sg_code` to INT" | SNOMED SCTIDs (`7181000119102`) are `concept_code`s, overflow 32-bit INT, and corrupt OHDSI joins. |
| "Add a `status` column to `condition_occurrence` for `NEGATED` rows" | Violates OMOP v5.4 DDL; OHDSI Atlas cohorts ignore custom columns and count negations as false-positive disease. |
| "Query OMOPHub with `supporting_span` from `src/cli.py`" | Exfiltrates clinical text/PHI and breaks Hospital Zone-A air-gapped portability. |
| "Insert all `AFFIRMED` rows regardless of `provenance`" | `COPY_PASTE` rows turn 1 surgery copied across 7 daily progress notes into 7 fake repeat surgeries. |
| "Use `nlp_concept_id` or unquoted `offset` in `NOTE_NLP`" | Exact OMOP v5.4 column is `note_nlp_concept_id` (`term_exists = 'U'` for uncertain); unquoted `offset` fails DuckDB SQL parsing. |

## Red Flags — STOP Immediately
- Editing `src/` or hardcoding `oh_...` API keys anywhere
- Sending `supporting_span` to `api.omophub.com`
- Inserting `NEGATED`, `CONDITIONAL`/`UNCERTAIN`, `COPY_PASTE`, or `CONCEPT_PCI_STAGED` into `condition_occurrence` / `procedure_occurrence`
- Exporting `CONCEPT_SHI_*` without `--include-shi` and `--shi-reason`
