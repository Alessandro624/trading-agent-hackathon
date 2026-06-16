from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copyfile


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    journal_path: Path
    dashboard_path: Path
    resumed_from: Path | None = None


def create_run_context(base_dir: Path, ticker: str, resume_from: Path | None = None) -> RunContext:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}_{ticker.upper()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    context = RunContext(run_dir, run_dir / "journal.jsonl", run_dir / "dashboard.html", _resume_journal_path(resume_from))
    if context.resumed_from:
        copyfile(context.resumed_from, context.journal_path)
    return context


def latest_run(base_dir: Path) -> RunContext:
    runs = sorted([path for path in base_dir.glob("run_*") if path.is_dir()])
    if not runs:
        raise RuntimeError(f"No journal runs found in {base_dir}")
    run_dir = runs[-1]
    return RunContext(run_dir, run_dir / "journal.jsonl", run_dir / "dashboard.html")


def _resume_journal_path(resume_from: Path | None) -> Path | None:
    if resume_from is None:
        return None
    path = resume_from
    if path.is_dir():
        path = path / "journal.jsonl"
    if not path.exists():
        raise RuntimeError(f"Resume journal not found: {path}")
    if path.name != "journal.jsonl":
        raise RuntimeError(f"Resume path must be a journal.jsonl file or run directory: {resume_from}")
    return path
