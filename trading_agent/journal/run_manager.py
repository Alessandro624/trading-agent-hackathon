from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copyfile

RESUMABLE_SIDECARS = (
    "human_input.md",
    "human_input.cursor",
    "constraints.json",
    "scheduled_actions.json",
    "conditional_orders.json",
    "pending_confirmations.json",
    "instruction_ledger.json",
    "agent_replies.md",
    "agent_replies.jsonl",
    "resolver_cache.json",
    "sector_cache.json",
)


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    journal_path: Path
    dashboard_path: Path
    human_input_path: Path
    human_input_cursor_path: Path
    resumed_from: Path | None = None


def create_run_context(base_dir: Path, ticker: str | None = None, resume_from: Path | None = None) -> RunContext:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag = _folder_tag(ticker)
    folder_name = f"run_{timestamp}{f'_{tag}' if tag else ''}"
    run_dir = _unique_run_dir(base_dir / folder_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    context = RunContext(
        run_dir,
        run_dir / "journal.jsonl",
        run_dir / "dashboard.html",
        run_dir / "human_input.md",
        run_dir / "human_input.cursor",
        _resume_journal_path(resume_from),
    )
    if context.resumed_from:
        copyfile(context.resumed_from, context.journal_path)
        for sidecar in RESUMABLE_SIDECARS:
            _copy_resume_sidecar(context.resumed_from.parent, context.run_dir, sidecar)
    _ensure_human_input_sidecars(context)
    return context


def _folder_tag(ticker: str | None) -> str:
    if not ticker:
        return ""
    cleaned = ticker.strip().upper()
    if not cleaned or cleaned == "MULTI":
        return ""
    return cleaned


def latest_run(base_dir: Path) -> RunContext:
    runs = sorted([path for path in base_dir.glob("run_*") if path.is_dir()])
    if not runs:
        raise RuntimeError(f"No journal runs found in {base_dir}")
    run_dir = runs[-1]
    return RunContext(
        run_dir,
        run_dir / "journal.jsonl",
        run_dir / "dashboard.html",
        run_dir / "human_input.md",
        run_dir / "human_input.cursor",
    )


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


def _copy_resume_sidecar(previous_run_dir: Path, run_dir: Path, name: str) -> None:
    source = previous_run_dir / name
    if source.exists():
        copyfile(source, run_dir / name)


def _ensure_human_input_sidecars(context: RunContext) -> None:
    if not context.human_input_path.exists():
        context.human_input_path.write_text("", encoding="utf-8")
    if not context.human_input_cursor_path.exists():
        context.human_input_cursor_path.write_text("0", encoding="utf-8")


def _unique_run_dir(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        next_candidate = candidate.with_name(f"{candidate.name}_{index}")
        if not next_candidate.exists():
            return next_candidate
        index += 1
