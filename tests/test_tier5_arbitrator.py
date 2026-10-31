import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tier5_llm_arbitrator import Tier5Arbitrator, pack_tier5_addon_zip


class TestTier5ArbitratorAndAddonPackager(unittest.TestCase):
    def test_non_loopback_ollama_host_rejected_for_airgap_safety(self):
        with self.assertRaises(ValueError):
            Tier5Arbitrator(enabled=True, ollama_host="10.20.30.40")
        with self.assertRaises(ValueError):
            Tier5Arbitrator(enabled=False, ollama_host="example.com")

    def test_fallback_to_llama_cli_caching_and_negation_evidence_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp)
            gguf = tpath / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
            gguf.write_bytes(b"GGUF_STUB")

            # Real Windows .cmd shim simulating zero-socket llama-cli.exe
            shim_conditional = tpath / "llama_cond.cmd"
            shim_conditional.write_text("@echo off\r\necho CONDITIONAL\r\n", encoding="utf-8")

            arb = Tier5Arbitrator(
                enabled=True,
                ollama_host="127.0.0.1",
                ollama_port=1,  # guaranteed closed loopback port -> falls back to llama-cli
                gguf_path=gguf,
                llama_cli=shim_conditional,
            )
            self.assertEqual(arb.backend, "llama-cli")
            self.assertEqual(arb.arbitrate("?pneumonia, sputum culture sent", "pneumonia"), "CONDITIONAL")
            self.assertEqual(arb.arbitrate("?pneumonia, sputum culture sent", "pneumonia"), "CONDITIONAL")
            self.assertEqual(arb.calls, 1)  # second call served from cache

            # Shim that always emits NEGATED to verify clinical NEGATION_EVIDENCE safety guard
            shim_neg = tpath / "llama_neg.cmd"
            shim_neg.write_text("@echo off\r\necho NEGATED\r\n", encoding="utf-8")
            arb_neg = Tier5Arbitrator(
                enabled=True,
                ollama_host="127.0.0.1",
                ollama_port=1,
                gguf_path=gguf,
                llama_cli=shim_neg,
            )
            # No lexical negation evidence -> hallucinated NEGATED must be rejected (None)
            self.assertIsNone(arb_neg.arbitrate("?nstemi, awaiting repeat hs-troponin", "nstemi"))
            # Explicit negation evidence ("negative") -> NEGATED accepted
            self.assertEqual(
                arb_neg.arbitrate("r/o pe, ctpa negative for pulmonary embolism", "pe"),
                "NEGATED",
            )

    def test_pack_tier5_addon_zip_and_zero_powershell_bat_scripts(self):
        llamacli_bat = ROOT / "setup_tier5_llamacli.bat"
        pack_bat = ROOT / "pack_tier5_addon.bat"
        self.assertTrue(llamacli_bat.exists(), "setup_tier5_llamacli.bat missing")
        self.assertTrue(pack_bat.exists(), "pack_tier5_addon.bat missing")

        for bfile in (llamacli_bat, pack_bat):
            txt = bfile.read_text(encoding="utf-8").lower()
            self.assertNotIn("powershell", txt)
            self.assertNotIn("pwsh", txt)
            self.assertNotIn("invoke-webrequest", txt)

        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp) / "sg_mediator"
            (fake_root / "runtime" / "llama").mkdir(parents=True)
            (fake_root / "models_llm").mkdir(parents=True)
            (fake_root / "runtime" / "llama" / "llama-cli.exe").write_bytes(b"MZ_FAKE_LLAMA")
            (fake_root / "models_llm" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf").write_bytes(b"GGUF_WEIGHTS")

            out_zip = Path(tmp) / "dist" / "sg_mediator_tier5_llm_addon_x64.zip"
            meta = pack_tier5_addon_zip(fake_root, out_zip)
            self.assertTrue(out_zip.exists())
            self.assertEqual(meta["files_packed"], 2)
            self.assertLess(meta["size_bytes"], 2 * 1024 * 1024 * 1024)

            with zipfile.ZipFile(out_zip, "r") as zf:
                names = set(zf.namelist())
            self.assertIn("runtime/llama/llama-cli.exe", names)
            self.assertIn("models_llm/Qwen2.5-3B-Instruct-Q4_K_M.gguf", names)


if __name__ == "__main__":
    unittest.main()
