import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import duckdb

SKILL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_DIR.parent.parent.parent
SCRIPTS_DIR = SKILL_DIR / "scripts"
RESOURCES_DIR = SKILL_DIR / "resources"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class TestOmopSkillBridge(unittest.TestCase):
    def test_all_128_seed_concepts_have_valid_omop_crosswalk(self) -> None:
        import omophub_bridge

        seed_csv = REPO_ROOT / "config" / "sg_ontology_seed.csv"
        crosswalk_csv = RESOURCES_DIR / "sg_omop_cdm_crosswalk.csv"
        self.assertTrue(crosswalk_csv.exists(), "Missing sg_omop_cdm_crosswalk.csv")

        with seed_csv.open("r", encoding="utf-8", newline="") as f:
            seed_rows = {r["concept_code"]: r for r in csv.DictReader(f)}
        self.assertEqual(len(seed_rows), 128)

        crosswalk = omophub_bridge.load_offline_crosswalk(crosswalk_csv)
        self.assertEqual(set(crosswalk.keys()), set(seed_rows.keys()))

        valid_domains = {"Condition", "Procedure", "Drug", "Measurement", "Observation"}
        valid_tables = {
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
            "measurement",
            "observation",
        }

        for code, entry in crosswalk.items():
            omop_id = int(entry["omop_concept_id"])
            self.assertGreater(omop_id, 0, f"Invalid omop_concept_id for {code}")
            snomed_raw = seed_rows[code]["snomed_sg_code"]
            if snomed_raw.isdigit():
                self.assertNotEqual(
                    omop_id,
                    int(snomed_raw),
                    f"{code} used raw SNOMED code {snomed_raw} instead of OMOP concept_id",
                )
            self.assertIn(entry["omop_domain_id"], valid_domains, f"Invalid domain for {code}")
            self.assertIn(entry["target_cdm_table"], valid_tables, f"Invalid table for {code}")
            self.assertEqual(entry["standard_concept"], "S", f"Non-standard concept for {code}")

        # Canonical OHDSI Athena concept_id checks
        self.assertEqual(int(crosswalk["CONCEPT_DM_T2"]["omop_concept_id"]), 201826)
        self.assertEqual(int(crosswalk["CONCEPT_HTN"]["omop_concept_id"]), 320128)
        # Planned staged PCI maps to Observation ('Planned procedure' 4275495) with value_as_concept_id=4203772
        self.assertEqual(crosswalk["CONCEPT_PCI_STAGED"]["target_cdm_table"], "observation")
        self.assertEqual(int(crosswalk["CONCEPT_PCI_STAGED"]["omop_concept_id"]), 4275495)
        self.assertEqual(int(crosswalk["CONCEPT_PCI_STAGED"]["value_as_concept_id"]), 4203772)

    def test_convert_extracted_csv_and_duckdb_enforces_assertion_dedup_and_shi_routing(self) -> None:
        import omophub_bridge

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            extracted_csv = tmp_path / "sg_mediator_extracted_concepts.csv"
            out_db = tmp_path / "omop_cdm.duckdb"

            fields = [
                "patient_id", "visit_no", "visit_date", "modality", "care_setting", "copy_paste_ratio",
                "concept_code", "vq_token_id", "canonical_name", "domain", "assertion_status", "provenance",
                "evidence_source", "umls_cui", "snomed_sg_code", "icd10am_code", "achi_code", "moh_tosp_code",
                "tosp_table_tier", "supporting_span",
            ]
            sample_rows = [
                # 1. AFFIRMED NEW Condition -> omop_condition_occurrence
                {
                    "patient_id": "PT_9f86d081884c",
                    "visit_no": "V001",
                    "visit_date": "2026-05-10",
                    "modality": "clinical_note",
                    "care_setting": "INPATIENT",
                    "copy_paste_ratio": "0.0",
                    "concept_code": "CONCEPT_DM_T2",
                    "vq_token_id": "VQ_01042",
                    "canonical_name": "Type 2 Diabetes Mellitus",
                    "domain": "Condition",
                    "assertion_status": "AFFIRMED",
                    "provenance": "NEW",
                    "evidence_source": "TIER1_AHOCORASICK",
                    "umls_cui": "C0011860",
                    "snomed_sg_code": "44054006",
                    "icd10am_code": "E11.9",
                    "achi_code": "NONE",
                    "moh_tosp_code": "NONE",
                    "tosp_table_tier": "NONE",
                    "supporting_span": "Background T2DM on Jardiance",
                },
                # 2. AFFIRMED NEW Performed Procedure (Primary PCI, SD701H) -> omop_procedure_occurrence
                {
                    "patient_id": "PT_9f86d081884c",
                    "visit_no": "V001",
                    "visit_date": "2026-05-10",
                    "modality": "clinical_note",
                    "care_setting": "INPATIENT",
                    "copy_paste_ratio": "0.0",
                    "concept_code": "CONCEPT_PCI_PRIMARY",
                    "vq_token_id": "VQ_07113",
                    "canonical_name": "Primary Emergent Percutaneous Coronary Intervention",
                    "domain": "Procedure",
                    "assertion_status": "AFFIRMED",
                    "provenance": "NEW",
                    "evidence_source": "TIER1_AHOCORASICK",
                    "umls_cui": "C1532328",
                    "snomed_sg_code": "415070008",
                    "icd10am_code": "Z95.5",
                    "achi_code": "38306-00",
                    "moh_tosp_code": "SD701H",
                    "tosp_table_tier": "5B",
                    "supporting_span": "Successful primary PCI to LAD",
                },
                # 3. COPY_PASTE duplicate of Primary PCI on Day 2 -> suppressed from procedure_occurrence!
                {
                    "patient_id": "PT_9f86d081884c",
                    "visit_no": "V002",
                    "visit_date": "2026-05-11",
                    "modality": "clinical_note",
                    "care_setting": "INPATIENT",
                    "copy_paste_ratio": "0.85",
                    "concept_code": "CONCEPT_PCI_PRIMARY",
                    "vq_token_id": "VQ_07113",
                    "canonical_name": "Primary Emergent Percutaneous Coronary Intervention",
                    "domain": "Procedure",
                    "assertion_status": "AFFIRMED",
                    "provenance": "COPY_PASTE",
                    "evidence_source": "TIER1_AHOCORASICK",
                    "umls_cui": "C1532328",
                    "snomed_sg_code": "415070008",
                    "icd10am_code": "Z95.5",
                    "achi_code": "38306-00",
                    "moh_tosp_code": "SD701H",
                    "tosp_table_tier": "5B",
                    "supporting_span": "Successful primary PCI to LAD",
                },
                # 4. Planned Staged PCI -> omop_observation (4275495 Planned procedure, value_as_concept_id=4203772)
                {
                    "patient_id": "PT_9f86d081884c",
                    "visit_no": "V001",
                    "visit_date": "2026-05-10",
                    "modality": "clinical_note",
                    "care_setting": "INPATIENT",
                    "copy_paste_ratio": "0.0",
                    "concept_code": "CONCEPT_PCI_STAGED",
                    "vq_token_id": "VQ_07114",
                    "canonical_name": "Staged / Deferred Percutaneous Coronary Intervention",
                    "domain": "Procedure",
                    "assertion_status": "AFFIRMED",
                    "provenance": "NEW",
                    "evidence_source": "TIER1_AHOCORASICK",
                    "umls_cui": "C1532328",
                    "snomed_sg_code": "415070008",
                    "icd10am_code": "Z95.5",
                    "achi_code": "38306-00",
                    "moh_tosp_code": "SD701H",
                    "tosp_table_tier": "5B",
                    "supporting_span": "Plan for staged PCI to RCA in 4 weeks",
                },
                # 5. NEGATED Condition -> omop_note_nlp ONLY (term_exists='N')
                {
                    "patient_id": "PT_9f86d081884c",
                    "visit_no": "V001",
                    "visit_date": "2026-05-10",
                    "modality": "clinical_note",
                    "care_setting": "INPATIENT",
                    "copy_paste_ratio": "0.0",
                    "concept_code": "CONCEPT_PULMONARY_EMBOLISM",
                    "vq_token_id": "VQ_03301",
                    "canonical_name": "Acute Pulmonary Thromboembolism",
                    "domain": "Condition",
                    "assertion_status": "NEGATED",
                    "provenance": "NEW",
                    "evidence_source": "TIER3_CONTEXT",
                    "umls_cui": "C0034065",
                    "snomed_sg_code": "59282003",
                    "icd10am_code": "I26.99",
                    "achi_code": "NONE",
                    "moh_tosp_code": "NONE",
                    "tosp_table_tier": "NONE",
                    "supporting_span": "CTPA negative for pulmonary embolism",
                },
                # 6. HISTORICAL Procedure -> omop_observation (4214956 History of clinical finding)
                {
                    "patient_id": "PT_9f86d081884c",
                    "visit_no": "V001",
                    "visit_date": "2026-05-10",
                    "modality": "clinical_note",
                    "care_setting": "INPATIENT",
                    "copy_paste_ratio": "0.0",
                    "concept_code": "CONCEPT_CABG",
                    "vq_token_id": "VQ_07145",
                    "canonical_name": "Coronary Artery Bypass Graft Surgery",
                    "domain": "Procedure",
                    "assertion_status": "HISTORICAL",
                    "provenance": "NEW",
                    "evidence_source": "TIER3_CONTEXT",
                    "umls_cui": "C0010055",
                    "snomed_sg_code": "232717009",
                    "icd10am_code": "Z95.1",
                    "achi_code": "38497-00",
                    "moh_tosp_code": "SD808H",
                    "tosp_table_tier": "6C",
                    "supporting_span": "Past surgical history: CABG in 2018",
                },
                # 7. CONDITIONAL Condition -> omop_note_nlp ONLY (term_exists='U')
                {
                    "patient_id": "PT_9f86d081884c",
                    "visit_no": "V001",
                    "visit_date": "2026-05-10",
                    "modality": "clinical_note",
                    "care_setting": "INPATIENT",
                    "copy_paste_ratio": "0.0",
                    "concept_code": "CONCEPT_AMI_NSTEMI",
                    "vq_token_id": "VQ_03014",
                    "canonical_name": "Non-ST-Elevation Myocardial Infarction",
                    "domain": "Condition",
                    "assertion_status": "CONDITIONAL",
                    "provenance": "NEW",
                    "evidence_source": "TIER5_LLM",
                    "umls_cui": "C0852707",
                    "snomed_sg_code": "401314000",
                    "icd10am_code": "I21.4",
                    "achi_code": "NONE",
                    "moh_tosp_code": "NONE",
                    "tosp_table_tier": "NONE",
                    "supporting_span": "?NSTEMI pending repeat hsTnI",
                },
                # 8. SHI_Restricted -> suppressed unless include_shi=True and shi_reason provided
                {
                    "patient_id": "PT_9f86d081884c",
                    "visit_no": "V001",
                    "visit_date": "2026-05-10",
                    "modality": "clinical_note",
                    "care_setting": "INPATIENT",
                    "copy_paste_ratio": "0.0",
                    "concept_code": "CONCEPT_SHI_HIV",
                    "vq_token_id": "VQ_15001",
                    "canonical_name": "HIV / Retroviral Infection (NEHR Sensitive Health Info)",
                    "domain": "SHI_Restricted",
                    "assertion_status": "AFFIRMED",
                    "provenance": "NEW",
                    "evidence_source": "TIER1_AHOCORASICK",
                    "umls_cui": "C0019693",
                    "snomed_sg_code": "86406008",
                    "icd10am_code": "B20",
                    "achi_code": "NONE",
                    "moh_tosp_code": "NONE",
                    "tosp_table_tier": "NONE",
                    "supporting_span": "[SHI_REDACTED]",
                },
            ]
            with extracted_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(sample_rows)

            summary = omophub_bridge.export_to_omop_duckdb(
                input_path=extracted_csv,
                output_duckdb=out_db,
                include_shi=False,
            )
            self.assertEqual(summary["shi_suppressed_count"], 1)
            self.assertEqual(summary["copy_paste_suppressed_count"], 1)

            con = duckdb.connect(str(out_db), read_only=True)
            try:
                conds = con.execute(
                    "SELECT condition_concept_id, condition_type_concept_id FROM omop_condition_occurrence"
                ).fetchall()
                self.assertEqual(conds, [(201826, 32858)])

                procs = con.execute(
                    "SELECT procedure_concept_id, procedure_source_value, procedure_type_concept_id FROM omop_procedure_occurrence"
                ).fetchall()
                # Only 1 performed Primary PCI (COPY_PASTE duplicate excluded; Staged PCI routed to Observation)
                self.assertEqual(procs, [(4203772, "SD701H", 32858)])

                obs = con.execute(
                    "SELECT observation_concept_id, value_as_concept_id FROM omop_observation ORDER BY observation_concept_id"
                ).fetchall()
                # 4214956 (History of CABG=4336464) + 4275495 (Planned Staged PCI=4203772)
                self.assertEqual(obs, [(4214956, 4336464), (4275495, 4203772)])

                nlp_cols = [
                    r[0]
                    for r in con.execute("DESCRIBE omop_note_nlp").fetchall()
                ]
                self.assertIn("note_nlp_concept_id", nlp_cols)
                self.assertNotIn("nlp_concept_id", nlp_cols)

                nlp_rows = con.execute(
                    "SELECT note_nlp_concept_id, snippet, term_exists FROM omop_note_nlp"
                ).fetchall()
                # All 7 non-SHI rows recorded in note_nlp; snippets blanked by default for DLP!
                self.assertEqual(len(nlp_rows), 7)
                self.assertTrue(all(r[1] == "" for r in nlp_rows))
                self.assertEqual({r[2] for r in nlp_rows}, {"Y", "N", "U"})

                stcm = con.execute(
                    "SELECT DISTINCT source_code, source_vocabulary_id FROM omop_source_to_concept_map WHERE source_code='SD701H'"
                ).fetchall()
                self.assertEqual(stcm, [("SD701H", "SG_MOH_TOSP")])
            finally:
                con.close()

    def test_omophub_api_client_endpoints_and_dlp_phi_guard(self) -> None:
        import omophub_bridge

        captured = []

        def fake_transport(method: str, url: str, headers: dict, body: bytes | None) -> dict:
            captured.append((method, url, headers, json.loads(body.decode("utf-8")) if body else None))
            if "/v1/concepts/by-code/" in url:
                return {"data": {"concept_id": 201826, "concept_name": "Type 2 diabetes mellitus", "domain_id": "Condition", "vocabulary_id": "SNOMED", "standard_concept": "S"}}
            if "/v1/concepts/map" in url:
                return {"data": {"mappings": [{"target_concept_id": 201826, "target_domain_id": "Condition", "target_vocabulary_id": "SNOMED"}]}}
            return {"data": {"concepts": [{"concept_id": 201826, "concept_name": "Type 2 diabetes mellitus", "domain_id": "Condition", "vocabulary_id": "SNOMED", "standard_concept": "S"}]}}

        client = omophub_bridge.OMOPHubClient(api_key="oh_test_ephemeral", transport=fake_transport)
        res_code = client.get_by_code("SNOMED", "44054006")
        self.assertEqual(res_code["data"]["concept_id"], 201826)
        self.assertTrue(captured[0][1].endswith("/v1/concepts/by-code/SNOMED/44054006"))
        self.assertEqual(captured[0][2]["Authorization"], "Bearer oh_test_ephemeral")

        client.map_codes([{"vocabulary_id": "ICD10CM", "concept_code": "E11.9"}], target_vocabulary="SNOMED")
        self.assertTrue(captured[1][1].endswith("/v1/concepts/map"))
        self.assertEqual(captured[1][0], "POST")

        client.search_concepts("Type 2 Diabetes Mellitus", vocabulary_ids=["SNOMED"])
        self.assertIn("/v1/search/concepts?", captured[2][1])

        # DLP guard MUST block Singapore NRIC/FIN, MCR, +65 phone, email, or raw supporting_span
        synthetic_nric = "S" + "8612345" + "D"
        with self.assertRaises(ValueError):
            client.search_concepts(f"Patient {synthetic_nric} has diabetes")
        with self.assertRaises(ValueError):
            client.search_concepts("Call +65 91234567 for follow-up")
        with self.assertRaises(ValueError):
            client.search_concepts("Dr Tan M12345A reviewed CTPA")


if __name__ == "__main__":
    unittest.main()
