"""
Append-only JSONL trace for CEO / worker orchestration (ablations, interpretability).

Enable:
  export HASHIRU_ORCHESTRATION_TRACE_JSONL=/path/to/run.jsonl
  export HASHIRU_ORCHESTRATION_TRACE_JSONL=/path/to/logs_dir

Path rules:

- Ends with ``.jsonl`` → **single file**; every process appends to that path (can grow large).
- Anything else → treated as a **directory**; each HASHIRU process gets its own file:
  ``trace_<YYYYMMDD_HHMMSS>_<pid>.jsonl``. Start a new file by restarting HASHIRU (new process).

If HASHIRU runs under **WSL**, set variables in the **same WSL shell** before ``python app.py``,
with Linux paths. ``export VAR="~/x"`` does not expand ``~`` in bash — use ``VAR=~/x`` or absolute paths.

Each line is one JSON object with at least: ts (unix), event (str), plus fields. Event lines also
include ``trace_file`` (basename) in directory mode.

Call ``init_orchestration_trace_session()`` once at app startup (optional but recommended) to emit
``trace_session_start`` and create the per-session file immediately.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any

_logger = logging.getLogger(__name__)

_session_lock = threading.Lock()
# Resolved absolute path to the active JSONL file; None = not yet resolved; "" = tracing disabled
_trace_abs_path: str | None = None
_trace_basename: str = ""
_session_header_written: bool = False


def trace_path() -> str | None:
    raw = os.environ.get("HASHIRU_ORCHESTRATION_TRACE_JSONL", "").strip()
    if not raw:
        return None
    p = os.path.normpath(os.path.expandvars(os.path.expanduser(raw)))
    return p or None


def _explicit_jsonl_file(configured: str) -> bool:
    return configured.lower().endswith(".jsonl")


def _session_filename() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"trace_{stamp}_{os.getpid()}.jsonl"


def _ensure_trace_target_unlocked(configured: str) -> str | None:
    """
    Set global trace file path on first use. Caller must hold _session_lock.
    Returns absolute path or None if tracing unavailable.
    """
    global _trace_abs_path, _trace_basename

    if _trace_abs_path is not None:
        if _trace_abs_path == "":
            return None
        return _trace_abs_path

    try:
        if _explicit_jsonl_file(configured):
            abs_path = os.path.abspath(configured)
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            _trace_basename = os.path.basename(abs_path)
            _trace_abs_path = abs_path
            return abs_path

        dir_abs = os.path.abspath(configured)
        os.makedirs(dir_abs, exist_ok=True)
        fname = _session_filename()
        abs_path = os.path.join(dir_abs, fname)
        _trace_basename = fname
        _trace_abs_path = abs_path
        return abs_path
    except OSError as e:
        _logger.warning(
            "HASHIRU_ORCHESTRATION_TRACE_JSONL: could not init (%s): %s",
            configured,
            e,
        )
        _trace_abs_path = ""
        return None


def init_orchestration_trace_session() -> str | None:
    """
    Create the session trace file and write a single ``trace_session_start`` line.
    Safe to call once at app startup; no-op if tracing is disabled or header already written.
    """
    global _session_header_written

    cfg = trace_path()
    if not cfg:
        with _session_lock:
            global _trace_abs_path
            if _trace_abs_path is None:
                _trace_abs_path = ""
        return None

    with _session_lock:
        if _session_header_written:
            return _trace_abs_path if _trace_abs_path not in (None, "") else None
        abs_path = _ensure_trace_target_unlocked(cfg)
        if not abs_path:
            return None
        try:
            meta = {
                "ts": time.time(),
                "event": "trace_session_start",
                "trace_file": _trace_basename,
                "trace_path": abs_path,
                "pid": os.getpid(),
            }
            line = json.dumps(meta, default=str, ensure_ascii=False) + "\n"
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(line)
            _session_header_written = True
        except OSError as e:
            _logger.warning(
                "HASHIRU_ORCHESTRATION_TRACE_JSONL: session start write failed (%s): %s",
                abs_path,
                e,
            )
            return None
    return abs_path


def _truncate(val: Any, limit: int = 4000) -> Any:
    if isinstance(val, str) and len(val) > limit:
        return val[:limit] + f"...<truncated {len(val) - limit} chars>"
    if isinstance(val, dict):
        return {k: _truncate(v, limit=800) for k, v in list(val.items())[:80]}
    if isinstance(val, (list, tuple)) and len(val) > 64:
        return [_truncate(x, limit=400) for x in val[:10]] + [f"...<{len(val)-10} more>"]
    return val


def log_orchestration_event(event: str, **fields: Any) -> None:
    global _session_header_written

    cfg = trace_path()
    if not cfg:
        return

    with _session_lock:
        abs_path = _ensure_trace_target_unlocked(cfg)
        if not abs_path:
            return

        if not _session_header_written:
            try:
                if (not os.path.isfile(abs_path)) or os.path.getsize(abs_path) == 0:
                    meta = {
                        "ts": time.time(),
                        "event": "trace_session_start",
                        "trace_file": _trace_basename,
                        "trace_path": abs_path,
                        "pid": os.getpid(),
                    }
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(meta, default=str, ensure_ascii=False) + "\n")
            except OSError as e:
                _logger.warning(
                    "HASHIRU_ORCHESTRATION_TRACE_JSONL: lazy session header failed: %s", e
                )
            _session_header_written = True

        rec: dict[str, Any] = {"ts": time.time(), "event": event}
        if _trace_basename:
            rec["trace_file"] = _trace_basename
        for k, v in fields.items():
            rec[k] = _truncate(v)
        line = json.dumps(rec, default=str, ensure_ascii=False) + "\n"
        try:
            with open(abs_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            _logger.warning(
                "HASHIRU_ORCHESTRATION_TRACE_JSONL: could not append (%s): %s",
                abs_path,
                e,
            )
