"""
Tier-5 Local LLM Assertion Arbitrator (optional, 16 GB profile only, 100% offline).

Only invoked for the small fraction of concept mentions whose assertion is genuinely ambiguous
(hedged language such as "?NSTEMI", "r/o ACS", "possible", "query", "likely"). Every other mention
is resolved deterministically by the rule-based ConText engine in sieve_and_risk.py.

Backends (first available wins):
  1. Portable Ollama HTTP server bound to 127.0.0.1 (runtime/ollama/ollama.exe serve)
  2. llama-cli.exe subprocess with a local GGUF file (runtime/llama/llama-cli.exe)

Failure policy: if no backend is reachable or the model returns an unparseable/unsafe answer,
arbitrate() returns None and the caller keeps its deterministic rule-based assertion. The LLM can
never break a run. Only loopback addresses are ever contacted.

Validation (held-out set of 20 hedged sentences, hand-labelled): rules-only 9/20; Qwen2.5-1.5B
zero-shot 10/20 (5 dangerous false negations); Qwen2.5-3B few-shot 18/20 (0 dangerous). Hence the
3B model + few-shot prompt + the negation-evidence guard below. Small synthetic set: treat as a
smoke test, not clinical validation.
"""
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, Optional

VALID_LABELS = ("AFFIRMED", "NEGATED", "HISTORICAL", "CONDITIONAL", "UNCERTAIN")

PROMPT_TEMPLATE = (
    "Classify the concept's status in a clinical sentence. Answer with ONE word:\n"
    "AFFIRMED = the condition is present/confirmed now (hedge words like possible/likely/suspect "
    "do NOT matter if the same sentence gives confirming evidence).\n"
    "NEGATED = the sentence says the condition is absent or ruled out.\n"
    "HISTORICAL = a past event, resolved, or status post.\n"
    "CONDITIONAL = suspected, queried, awaiting tests, or only 'if/consider' - NOT yet confirmed or refuted.\n"
    "A 'r/o', '?', 'query' or 'rule out' marker alone means CONDITIONAL, never NEGATED.\n\n"
    "Sentence: ?pneumonia, sputum culture sent\nConcept: pneumonia\nAnswer: CONDITIONAL\n"
    "Sentence: r/o pe, ctpa shows no embolus, pe excluded\nConcept: pe\nAnswer: NEGATED\n"
    "Sentence: likely uti, urine fullit positive, started on nitrofurantoin\nConcept: uti\nAnswer: AFFIRMED\n"
    "Sentence: s/p appendectomy in 2012\nConcept: appendectomy\nAnswer: HISTORICAL\n"
    "Sentence: rule out gi bleed, hb stable, awaiting oge\nConcept: gi bleed\nAnswer: CONDITIONAL\n"
    "Sentence: suspect hypothyroidism, tsh 48 and ft4 low\nConcept: hypothyroidism\nAnswer: AFFIRMED\n"
    "Sentence: {sentence}\nConcept: {concept}\nAnswer:"
)

# Safety guard: an LLM NEGATED verdict is only accepted if the sentence itself contains negation evidence
NEGATION_EVIDENCE = (
    "no ", "not ", "negative", "exclud", "ruled out", "clear", "normal", "absent", "nil", "without",
    "denies", "free of",
)


class Tier5Arbitrator:
    def __init__(
        self,
        enabled: bool = False,
        root_dir: Optional[Path] = None,
        ollama_host: str = "127.0.0.1",
        ollama_port: int = 11434,
        ollama_model: str = "qwen2.5:3b-instruct-q4_K_M",
        gguf_path: Optional[Path] = None,
        llama_cli: Optional[Path] = None,
        timeout_s: float = 30.0,
    ):
        self.enabled = enabled
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        self.ollama_model = ollama_model
        self.gguf_path = gguf_path or (self.root_dir / "models_llm" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
        self.llama_cli = llama_cli or (self.root_dir / "runtime" / "llama" / "llama-cli.exe")
        self.timeout_s = timeout_s
        self.cache: Dict[str, Optional[str]] = {}
        self.calls = 0
        self.failures = 0
        self._backend: Optional[str] = None
        if enabled:
            self._backend = self._detect_backend()

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def _detect_backend(self) -> Optional[str]:
        try:
            with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=2) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            names = {m.get("name") for m in tags.get("models", [])}
            if self.ollama_model in names:
                return "ollama"
        except Exception:
            pass
        if self.llama_cli.exists() and self.gguf_path.exists():
            return "llama-cli"
        return None

    @staticmethod
    def _parse(text: str) -> Optional[str]:
        m = re.search(r"\b(AFFIRMED|NEGATED|HISTORICAL|CONDITIONAL|UNCERTAIN)\b", text.upper())
        return m.group(1) if m else None

    def _ask_ollama(self, prompt: str) -> Optional[str]:
        body = json.dumps(
            {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 6},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return self._parse(json.loads(resp.read().decode("utf-8")).get("response", ""))

    def _ask_llama_cli(self, prompt: str) -> Optional[str]:
        proc = subprocess.run(
            [
                str(self.llama_cli), "-m", str(self.gguf_path), "-p", prompt,
                "-n", "6", "--temp", "0", "-c", "2048", "-t", "2", "--no-display-prompt",
            ],
            capture_output=True, text=True, timeout=self.timeout_s * 3,
        )
        return self._parse(proc.stdout)

    def arbitrate(self, sentence: str, concept: str) -> Optional[str]:
        if not self.enabled or self._backend is None:
            return None
        key = f"{concept}||{sentence}"
        if key in self.cache:
            return self.cache[key]
        prompt = PROMPT_TEMPLATE.format(sentence=sentence.strip()[:400], concept=concept)
        self.calls += 1
        try:
            label = self._ask_ollama(prompt) if self._backend == "ollama" else self._ask_llama_cli(prompt)
        except Exception:
            self.failures += 1
            label = None
        if label == "NEGATED" and not any(c in sentence.lower() for c in NEGATION_EVIDENCE):
            label = None  # never let the LLM invent a negation; fall back to deterministic rules
        self.cache[key] = label
        return label
