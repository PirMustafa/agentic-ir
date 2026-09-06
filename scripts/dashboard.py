"""A local dashboard for watching and probing the system.

Four things this makes possible that a terminal does not: watching the sweep
fill in without polling by hand, browsing all 250 questions of a run with the
scores attached and filters for the interesting ones, opening any single
question's full trace as a timeline, and -- when explicitly enabled -- asking
the live pipeline something it has never seen.

Standard library only. There is no framework here on purpose: adding FastAPI to
a coursework repo's dependencies to serve four JSON endpoints is a cost the
project does not need, and installing anything while an evaluation sweep is
using the machine is a risk it does not need either.

    python scripts/dashboard.py                 # read-only, no GPU, no model
    python scripts/dashboard.py --live          # also enables the Ask tab
    python scripts/dashboard.py --port 8123

The Ask tab is off by default, but the cost of turning it on is smaller than
it first looks. ``config/config.yaml`` puts every query-time encoder on the
CPU -- ``dense.query_device``, ``rerank.device`` and ``verifier.nli_device``
are all ``cpu``, so the whole card belongs to Ollama. A live question adds no
VRAM. What it adds is contention for Ollama's generation slot: with
``OLLAMA_NUM_PARALLEL=1`` the requests queue, so asking during an evaluation
run makes both slower rather than putting either at risk.

The default stays off anyway. Those device settings are configuration, not
physics, and a card that is 6.4 GB into 8.1 GB has no room for the version of
this where somebody has set them back to ``cuda``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"
CALIBRATION = ROOT / "results" / "calibration"
PAGE = Path(__file__).resolve().parent / "dashboard.html"

#: How many questions a complete run holds. Progress is shown against this, so
#: a run at 69 reads as 28% rather than as an unqualified number.
TARGET_N = 250

#: Set from --live. Guards the one endpoint that can touch the GPU.
LIVE_ENABLED = False

#: Serialises live questions. Two concurrent pipeline runs on one 8 GB card is
#: not a race worth having, and the browser makes it easy to fire three.
_ASK_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# reading what is on disk
# ---------------------------------------------------------------------------

def run_dirs() -> list[Path]:
    if not RUNS.exists():
        return []
    return sorted(d for d in RUNS.iterdir() if d.is_dir() and not d.name.startswith("_"))


def count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def parse_run_id(run_id: str) -> tuple[str, str, str]:
    """``(config, dataset, stamp)`` from ``agentic_full_hotpotqa_2026...Z``."""
    match = re.match(r"^(.*)_(hotpotqa|twowiki)_(\d{8}T\d{6}Z)$", run_id)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return run_id, "?", ""


def expected_configs() -> tuple[str, ...]:
    """The configurations a complete evaluation grid holds.

    Imported from the harness rather than listed here, so a configuration added
    to ``run_eval`` shows up as a missing cell instead of being invisible.
    """
    try:
        from agentic_ir.eval.run_eval import CONFIGURATIONS

        return tuple(CONFIGURATIONS)
    except Exception:  # noqa: BLE001 - the dashboard still works without it
        return ()


def sweep_alive(pid: int | None) -> bool:
    """Whether the sweep process is still running, without assuming a platform."""
    if pid is None:
        return False
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return str(pid) in out
        return subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001 - a status panel must not take the server down
        return False


def status_payload(sweep_pid: int | None) -> dict:
    """Every run on disk with its progress, newest first within each config."""
    rows = []
    for directory in run_dirs():
        traces = directory / "traces.jsonl"
        if not traces.exists():
            continue
        config, dataset, stamp = parse_run_id(directory.name)
        n = count_lines(traces)
        try:
            modified = traces.stat().st_mtime
        except OSError:
            modified = 0.0
        rows.append({
            "run_id": directory.name,
            "config": config,
            "dataset": dataset,
            "stamp": stamp,
            "n": n,
            "target": TARGET_N,
            "complete": n >= TARGET_N,
            "modified": modified,
            "age_s": max(0.0, time.time() - modified) if modified else None,
        })

    # One row per (config, dataset): the run a table would pick, which is the
    # longest and then the newest. Showing five stale bm25 runs beside the one
    # that counts is noise dressed as completeness.
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["config"], row["dataset"])
        current = best.get(key)
        if current is None or (row["n"], row["stamp"]) > (current["n"], current["stamp"]):
            best[key] = row

    # The grid is every configuration crossed with every dataset, so a cell
    # nobody has started yet appears as an empty row rather than as an absence.
    # A dashboard that only lists what exists cannot show what is missing,
    # which is the one thing a progress view is for.
    for config in expected_configs():
        for dataset in ("hotpotqa", "twowiki"):
            best.setdefault((config, dataset), {
                "run_id": None, "config": config, "dataset": dataset, "stamp": "",
                "n": 0, "target": TARGET_N, "complete": False,
                "modified": 0.0, "age_s": None,
            })

    return {
        "runs": sorted(best.values(), key=lambda r: (r["dataset"], r["config"])),
        "all_runs": sorted(rows, key=lambda r: r["run_id"]),
        "sweep_pid": sweep_pid,
        "sweep_alive": sweep_alive(sweep_pid),
        "live_enabled": LIVE_ENABLED,
        "now": time.time(),
    }


@lru_cache(maxsize=8)
def _load(run_id: str, mtime: float) -> tuple[list[dict], dict[str, dict]]:
    """Traces and per-question scores, cached on the trace file's mtime.

    The mtime is part of the key rather than a thing to check: a run still
    being written gets a new key every time the sweep appends, so the cache
    cannot serve a stale prefix of a growing file.
    """
    directory = RUNS / run_id
    records = []
    with open(directory / "traces.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    scores: dict[str, dict] = {}
    scores_path = directory / "scores.csv"
    if scores_path.exists():
        with open(scores_path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                qid = row.pop("qid", None)
                if qid:
                    scores[qid] = {
                        k: (float(v) if v not in ("", None) else None)
                        for k, v in row.items()
                    }
    return records, scores


def load_run(run_id: str) -> tuple[list[dict], dict[str, dict]]:
    traces = RUNS / run_id / "traces.jsonl"
    if not traces.exists():
        raise FileNotFoundError(run_id)
    return _load(run_id, traces.stat().st_mtime)


def question_rows(run_id: str) -> list[dict]:
    """One compact row per question: enough to filter and sort on."""
    records, scores = load_run(run_id)
    rows = []
    for record in records:
        qid = record["qid"]
        score = scores.get(qid, {})
        metrics = record.get("metrics", {})
        rows.append({
            "qid": qid,
            "question": record.get("question", ""),
            "answer": record.get("final_answer", ""),
            "gold": (record.get("gold") or {}).get("answer", ""),
            "level": (record.get("gold") or {}).get("level"),
            "qtype": (record.get("gold") or {}).get("qtype"),
            "em": score.get("em"),
            "f1": score.get("f1"),
            "sp_f1": score.get("sp_f1"),
            "recall10": score.get("recall@10"),
            "confidence": record.get("confidence"),
            "verdict": record.get("verdict"),
            "terminated_by": record.get("terminated_by"),
            "best_cycle": record.get("best_cycle"),
            "cycles": metrics.get("cycles"),
            "replanned": bool(metrics.get("replanned")),
            "llm_calls": metrics.get("llm_calls"),
            "tool_calls": metrics.get("tool_calls"),
            "latency_s": metrics.get("latency_s"),
            "citation_grounding": metrics.get("citation_grounding"),
            "n_candidates": len(record.get("candidates") or ()),
        })
    return rows


def question_detail(run_id: str, qid: str) -> dict:
    """One question's whole record, plus its scores, for the timeline view."""
    records, scores = load_run(run_id)
    record = next((r for r in records if r["qid"] == qid), None)
    if record is None:
        raise KeyError(qid)
    return {"record": record, "scores": scores.get(qid, {})}


# ---------------------------------------------------------------------------
# the generated tables, as data
# ---------------------------------------------------------------------------

def _detex(cell: str) -> str:
    cell = re.sub(r"\$\^\{?\\downarrow\}?\$", " \u25be", cell)
    cell = re.sub(r"\$?\^?\{?\\(dagger|ddagger|ast)\}?\$?", "", cell)
    cell = re.sub(r"\\text(bf|it|tt)\{(.*?)\}", r"\2", cell)
    cell = cell.replace(r"\Delta", "\u0394").replace(r"\%", "%")
    return re.sub(r"[\\${}^]", "", cell).strip()


def table_payload() -> list[dict]:
    """Every generated table as ``{name, caption, header, rows}``."""
    out = []
    if not TABLES.exists():
        return out
    for path in sorted(TABLES.glob("*.tex")):
        body = path.read_text(encoding="utf-8")
        sources = re.findall(r"^%\s+(\S+/\S+):\s+(\S+)\s+\((\d+) questions\)", body, re.M)
        for block in re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", body, re.S):
            caption = re.search(r"\\caption\{(.*?)\}\n", block, re.S)
            inner = re.search(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", block, re.S)
            if inner is None:
                continue
            grid = []
            for line in inner.group(1).splitlines():
                if "&" not in line:
                    continue
                cells = [
                    _detex(c) for c in line.strip().removesuffix(r"\\").split("&")
                ]
                grid.append(cells)
            if not grid:
                continue
            label = re.search(r"\\label\{(.*?)\}", block)
            out.append({
                "file": path.name,
                "label": label.group(1) if label else path.stem,
                "caption": _detex(caption.group(1)) if caption else "",
                "header": grid[0],
                "rows": grid[1:],
                "sources": [
                    {"key": key, "run_id": run_id, "n": int(n)}
                    for key, run_id, n in sources
                ],
            })
    return out


def calibration_payload() -> dict:
    if not CALIBRATION.exists():
        return {}
    out = {}
    for path in sorted(CALIBRATION.glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


# ---------------------------------------------------------------------------
# the live pipeline
# ---------------------------------------------------------------------------

def ask_live(question: str, dataset: str, config_name: str) -> dict:
    """Run one question through the real system and return its trace record.

    Imports are local and the pipeline is built per request. Loading a corpus
    and three models at import time would make a read-only dashboard pay the
    whole cost of the live one.
    """
    from agentic_ir.config import load_config
    from agentic_ir.eval.run_eval import build_system, config_for, load_pipeline
    from agentic_ir.llm import get_client
    from agentic_ir.state import QuestionState
    from agentic_ir.trace import build_trace_record

    cfg = config_for(config_name, load_config())
    pipeline = load_pipeline(
        dataset, cfg, with_kg=bool(cfg.get("agents.kg.enabled", True))
    )
    system = build_system(
        config_name, dataset, cfg=cfg, pipeline=pipeline, client=get_client()
    )
    qid = f"live_{int(time.time())}"
    state = QuestionState(
        qid=qid, question=question, dataset=dataset, config_name=config_name
    )
    started = time.perf_counter()
    state = system.run(qid, question, gold=None, state=state)
    # The transition path lives on the orchestrator, not on the state, and
    # ``build_trace_record`` defaults it to empty. During an evaluation the
    # TraceWriter supplies it; here nothing does unless it is passed, and a
    # live question would render with a blank state-machine panel -- the one
    # panel that shows the loop.
    record = build_trace_record(
        state, run_id="dashboard", seed=int(cfg.get("project.seed", 42)),
        model=str(cfg.get("llm.default_model", "")),
        transitions=tuple(getattr(system, "transitions", ()) or ()),
    )
    record["_wall_s"] = time.perf_counter() - started
    record["_notes"] = list(getattr(pipeline, "notes", ()) or ())
    return {"record": record, "scores": {}}


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    sweep_pid: int | None = None

    def log_message(self, fmt, *args):  # noqa: A003 - quieten the default access log
        pass

    def _send(self, payload, status: int = 200, content_type: str = "application/json"):
        if content_type == "application/json":
            body = json.dumps(payload).encode("utf-8")
        else:
            body = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        url = urlparse(self.path)
        query = parse_qs(url.query)
        route = url.path

        try:
            if route in ("/", "/index.html"):
                if not PAGE.exists():
                    return self._send("dashboard.html is missing", 500, "text/plain")
                return self._send(PAGE.read_bytes(), 200, "text/html")

            if route == "/api/status":
                return self._send(status_payload(self.sweep_pid))

            if route == "/api/questions":
                run_id = (query.get("run") or [""])[0]
                return self._send({"run_id": run_id, "rows": question_rows(run_id)})

            if route == "/api/question":
                run_id = (query.get("run") or [""])[0]
                qid = (query.get("qid") or [""])[0]
                return self._send(question_detail(run_id, qid))

            if route == "/api/tables":
                return self._send({
                    "tables": table_payload(),
                    "calibration": calibration_payload(),
                })

            return self._send({"error": "not found"}, 404)

        except FileNotFoundError as exc:
            return self._send({"error": f"no such run: {exc}"}, 404)
        except KeyError as exc:
            return self._send({"error": f"no such question: {exc}"}, 404)
        except Exception as exc:  # noqa: BLE001 - a bad request must not kill the server
            return self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/api/ask":
            return self._send({"error": "not found"}, 404)
        if not LIVE_ENABLED:
            return self._send({
                "error": "live answering is disabled. Restart with --live. "
                         "It queues behind a running evaluation rather than "
                         "competing with it, so long as the encoders are on CPU."
            }, 403)

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send({"error": "malformed request"}, 400)

        question = (payload.get("question") or "").strip()
        if not question:
            return self._send({"error": "the question is empty"}, 400)

        if not _ASK_LOCK.acquire(blocking=False):
            return self._send({
                "error": "a question is already running; one at a time on this card"
            }, 429)
        try:
            return self._send(ask_live(
                question,
                payload.get("dataset") or "hotpotqa",
                payload.get("config") or "agentic_full",
            ))
        except Exception as exc:  # noqa: BLE001
            return self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)
        finally:
            _ASK_LOCK.release()


def main(argv: list[str] | None = None) -> int:
    global LIVE_ENABLED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--live", action="store_true",
        help="enable the Ask tab. Needs Ollama. Queues behind a running sweep "
             "rather than competing with it, as long as the encoders stay on CPU.",
    )
    parser.add_argument(
        "--sweep-pid", type=int, default=None,
        help="pid of a running sweep, so the status panel can say whether it is alive",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    LIVE_ENABLED = args.live
    Handler.sweep_pid = args.sweep_pid

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"dashboard on {url}")
    print(f"  runs      {len(run_dirs())} on disk")
    print(f"  live ask  {'ENABLED -- this will use the GPU' if args.live else 'disabled'}")
    if args.sweep_pid:
        alive = "alive" if sweep_alive(args.sweep_pid) else "not running"
        print(f"  sweep     pid {args.sweep_pid} ({alive})")
    print("  ctrl-c to stop")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
