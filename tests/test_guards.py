"""Structural guards.

These are not unit tests of behaviour; they are tripwires on the three
invariants that, if broken, would invalidate results while leaving everything
looking fine. Each corresponds to a risk recorded in ``docs/architecture.md``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "agentic_ir"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


# ---------------------------------------------------------------------------
# Axiom 6: local only
# ---------------------------------------------------------------------------

# llm.py talks to localhost:11434. Data scripts talk to HuggingFace to download
# corpora. Nothing else may reach the network at all.
_NETWORK_MODULES = {"requests", "httpx", "urllib3", "aiohttp", "openai", "anthropic"}
_NETWORK_EXEMPT = {"llm.py"}


def test_no_network_libraries_outside_the_llm_client():
    """A stray HTTP client is how a 'fully local' claim quietly becomes false."""
    offenders: list[str] = []
    for path in _py_files(SRC):
        if path.name in _NETWORK_EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for n in names:
                if n in _NETWORK_MODULES:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} imports {n}")
    assert not offenders, "network libraries outside llm.py:\n  " + "\n  ".join(offenders)


def test_no_hosted_llm_provider_anywhere():
    """The project is local-only by design; a hosted SDK would break the claim
    the report makes about cost and reproducibility."""
    hits: list[str] = []
    pattern = re.compile(r"\b(openai|anthropic|OpenAI\(|ChatCompletion)\b")
    for path in _py_files(SRC) + (_py_files(SCRIPTS) if SCRIPTS.exists() else []):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "test_" in path.name:
                continue
            if pattern.search(line):
                hits.append(f"{path.name}:{i}: {stripped[:70]}")
    assert not hits, "hosted LLM provider referenced:\n  " + "\n  ".join(hits)


# ---------------------------------------------------------------------------
# Section 1.10: agents must never read ground truth
# ---------------------------------------------------------------------------

def test_agents_never_read_gold():
    """``QuestionState.gold`` is carried for the trace only.

    An agent reading it would leak the answer key into the system under
    evaluation, and every number in Chapter 4 would be an artefact -- while
    the scores merely looked good.
    """
    agents_dir = SRC / "agents"
    if not agents_dir.exists():
        pytest.skip("agents package not written yet")
    offenders: list[str] = []
    for path in _py_files(agents_dir):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.gold\b|\bgold_answer\b|supporting_facts", line):
                if not line.strip().startswith("#"):
                    offenders.append(f"{path.name}:{i}: {line.strip()[:70]}")
    assert not offenders, "agent reads ground truth:\n  " + "\n  ".join(offenders)


# ---------------------------------------------------------------------------
# Risk 1: the knowledge graph must not be built from gold triples
# ---------------------------------------------------------------------------

def test_kg_build_never_opens_gold_evidence():
    """2WikiMultihopQA ships gold Wikidata evidence triples.

    Building the graph from those would let the KG Navigator read the answer
    key, making every KG-attributed gain in the report meaningless. The graph
    is built from corpus passage text only; gold triples belong to the scorer.
    """
    build = SRC / "kg" / "build.py"
    if not build.exists():
        pytest.skip("kg/build.py not written yet")
    text = build.read_text(encoding="utf-8")
    banned = re.findall(r'["\'][^"\']*(?:evidence|triple)[^"\']*["\']', text, re.I)
    banned = [b for b in banned if any(x in b.lower() for x in (".json", ".jsonl", "_evidence", "evidences"))]
    assert not banned, f"kg/build.py references gold evidence files: {banned}"


# ---------------------------------------------------------------------------
# The Windows encoding trap
# ---------------------------------------------------------------------------

def test_every_open_specifies_utf8():
    """Windows defaults to cp1252 and both datasets are full of names like
    'Xawery Zulawski'. This crashed once during environment validation; the
    failure would otherwise land mid-run, hours into an evaluation."""
    offenders: list[str] = []
    roots = [SRC] + ([SCRIPTS] if SCRIPTS.exists() else [])
    for path in [p for r in roots for p in _py_files(r)]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            is_open = (isinstance(fn, ast.Name) and fn.id == "open") or (
                isinstance(fn, ast.Attribute) and fn.attr == "open"
            )
            if not is_open:
                continue
            kwargs = {k.arg for k in node.keywords}
            # binary mode needs no encoding
            mode = next(
                (a.value for a in node.args[1:2] if isinstance(a, ast.Constant)),
                next((k.value.value for k in node.keywords
                      if k.arg == "mode" and isinstance(k.value, ast.Constant)), ""),
            )
            if "b" in str(mode):
                continue
            if "encoding" not in kwargs:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "open() without encoding='utf-8':\n  " + "\n  ".join(offenders)
    )
