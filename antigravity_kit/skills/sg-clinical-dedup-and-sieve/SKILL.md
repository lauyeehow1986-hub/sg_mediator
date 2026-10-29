---
name: sg-clinical-dedup-and-sieve
description: Longitudinal copy-paste deduplication (SimHash shingle alignment), Singapore clinical shorthand expansion, non-ICU episode stratification, SI-unit lab/medication risk derivation, Nutri-Grade/Healthier SG SDOH extraction, and Health Information Bill (HIB) Sensitive Health Information (SHI) enclave gating.
---
# Singapore Clinical Deduplication & Triangulated Risk Sieve Skill

## When to Use
Use this skill when modifying `src/deduplicator.py`, `src/sieve_and_risk.py`, `src/shi_enclave.py`, or `config/sg_clinical_rules.yaml`.

## Workflow
1. **Longitudinal Copy-Paste Suppression**:
   - Group patient records chronologically by `(visit_date, visit_no)`.
   - Compute 64-bit blake2b shingle hashes over 8-word sliding windows.
   - Mark segments with `>= 85%` overlap with any earlier visit as `provenance="CARRIED_FORWARD"` (recording `origin_visit_no`).
2. **Triangulated Risk Derivation**:
   - Combine explicit text mentions, Singapore SI-unit labs (`mmol/L`, `umol/L`, `mmol/mol`), prior/active HSA-registered medications (*Jardiance, Synjardy, Forxiga, Crestor, Entresto, Plavix*), and Agatston CAC scores.
3. **Singapore Policy & SDOH**:
   - Always capture HPB Nutri-Grade (`A/B` protective `kopi O kosong` vs `C/D` high-sugar beverages), Healthier SG enrollment, Asian BMI (`>=23.0` overweight, `>=27.5` obese), CHAS/PG/MG subsidies, HDB 1-2 room rental housing, and FDW caregiver support.
4. **Sensitive Health Information (SHI) Enclave**:
   - Route HIV/STD, psychiatric, substance abuse, termination of pregnancy, and hereditary genetic test findings through `SHIEnclaveGate` (`src/shi_enclave.py`).
