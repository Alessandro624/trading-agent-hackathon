from __future__ import annotations

import html
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("trading_agent.dashboard")


class DashboardProjectionWatcher:
    _SIDECARS = (
        "constraints.json",
        "scheduled_actions.json",
        "conditional_orders.json",
        "pending_confirmations.json",
        "instruction_ledger.json",
        "agent_replies.md",
    )

    def __init__(
        self,
        journal_path: Path,
        projection_path: Path,
        *,
        poll_seconds: float = 0.5,
    ) -> None:
        self.journal_path = Path(journal_path)
        self.projection_path = Path(projection_path)
        self.poll_seconds = poll_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_snapshot: tuple[tuple[str, int, int] | None, ...] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        write_dashboard_projection(self.journal_path, self.projection_path)
        self._last_snapshot = self._snapshot()
        self._thread = threading.Thread(
            target=self._watch,
            name="dashboard-projection-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds * 3))
        self._thread = None

    def _watched_paths(self) -> tuple[Path, ...]:
        run_dir = self.journal_path.parent
        return (self.journal_path, *(run_dir / name for name in self._SIDECARS))

    def _snapshot(self) -> tuple[tuple[str, int, int] | None, ...]:
        snapshot: list[tuple[str, int, int] | None] = []
        for path in self._watched_paths():
            try:
                stat = path.stat()
                snapshot.append((path.name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                snapshot.append(None)
        return tuple(snapshot)

    def _watch(self) -> None:
        while not self._stop_event.wait(self.poll_seconds):
            snapshot = self._snapshot()
            if snapshot == self._last_snapshot:
                continue
            try:
                write_dashboard_projection(self.journal_path, self.projection_path)
                self._last_snapshot = snapshot
            except Exception:
                logger.exception("dashboard.projection.refresh_failed")


def render_dashboard(journal_path: Path, output_path: Path) -> Path:
    shell_path = output_path
    projection_path = output_path.parent / "dashboard_data.js"
    shell_path.write_text(_SHELL_HTML, encoding="utf-8")
    write_dashboard_projection(journal_path, projection_path)
    return shell_path


def write_dashboard_projection(journal_path: Path, projection_path: Path) -> Path:
    run_dir = journal_path.parent
    data = _build_projection_data(journal_path, run_dir)
    payload = "window.DASHBOARD_DATA = " + json.dumps(data, ensure_ascii=False, default=str) + ";\n"
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".dashboard_data.", suffix=".js", dir=str(projection_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_path, projection_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return projection_path


def _build_projection_data(journal_path: Path, run_dir: Path) -> dict[str, Any]:
    rows = _read_rows(journal_path)
    cycle_rows = [r for r in rows if r.get("entry_type", "cycle") == "cycle"]
    stage_rows = [r for r in rows if r.get("entry_type") == "stage"]
    human_event_rows = [r for r in rows if r.get("entry_type") == "human_event"]
    ledger = _load_json_sidecar(run_dir / "instruction_ledger.json", "entries")
    confirmations = _confirmation_workflows(
        _load_json_sidecar(run_dir / "pending_confirmations.json", "confirmations"),
        ledger,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": run_dir.name,
        "stats": _stats(cycle_rows, stage_rows, human_event_rows, rows),
        "cycles": [_cycle_summary(row) for row in cycle_rows],
        "human_events": _grouped_human_events(human_event_rows),
        "stages": [_stage_summary(row) for row in stage_rows[-200:]],
        "constraints": _load_json_sidecar(run_dir / "constraints.json", "constraints"),
        "scheduled_actions": _load_json_sidecar(run_dir / "scheduled_actions.json", "actions"),
        "conditional_orders": _load_json_sidecar(run_dir / "conditional_orders.json", "orders"),
        "pending_confirmations": confirmations,
        "instruction_ledger": ledger,
        "agent_replies": _load_agent_replies(run_dir / "agent_replies.md"),
        "diversification_series": _diversification_series(cycle_rows),
        "evidence": _collect_evidence(cycle_rows, stage_rows, human_event_rows),
    }


def _confirmation_workflows(
    confirmations: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for confirmation in confirmations:
        confirmation_id = str(confirmation.get("confirmation_id") or "")
        children = [entry for entry in ledger if str(entry.get("originating_confirmation_id") or "") == confirmation_id]
        if not children:
            continue
        statuses = [str(entry.get("status") or "queued").lower() for entry in children]
        completed = statuses.count("completed")
        cancelled = statuses.count("cancelled") + statuses.count("abandoned")
        active = statuses.count("active")
        queued = statuses.count("queued")
        if completed == len(children):
            workflow_status = "completed"
        elif cancelled == len(children):
            workflow_status = "cancelled"
        elif completed and cancelled and completed + cancelled == len(children):
            workflow_status = "partially_cancelled"
        elif completed:
            workflow_status = "partially_completed"
        elif active:
            workflow_status = "active"
        else:
            workflow_status = "queued"
        confirmation["workflow_id"] = next(
            (entry.get("workflow_id") for entry in children if entry.get("workflow_id")),
            f"workflow-{confirmation_id}",
        )
        confirmation["workflow_status"] = workflow_status
        confirmation["operations_progress"] = {
            "completed": completed,
            "cancelled": cancelled,
            "total": len(children),
        }
        confirmation["child_instructions"] = children
        confirmation["queued_operations"] = queued
    return confirmations


def _read_rows(journal_path: Path) -> list[dict[str, Any]]:
    if not journal_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_json_sidecar(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        logger.warning("dashboard.sidecar.malformed path=%s reason=%s", path, error)
        return []
    items = data.get(key) if isinstance(data, dict) else None
    return items if isinstance(items, list) else []


def _load_agent_replies(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        from trading_agent.core.agent_reply import read_agent_replies

        return read_agent_replies(path)
    except Exception as error:
        logger.warning("dashboard.agent_replies.malformed path=%s reason=%s", path, error)
        return []


def _stats(cycle_rows, stage_rows, human_event_rows, all_rows) -> dict[str, Any]:
    from collections import Counter

    actions = Counter(row.get("action", "UNKNOWN") for row in cycle_rows)
    tickers = sorted({row.get("ticker", "-") for row in all_rows})
    timestamps = [row.get("timestamp") for row in (cycle_rows or all_rows) if row.get("timestamp")]
    guardrails = sum(len(row.get("guardrails_triggered") or []) for row in cycle_rows)
    failures = sum(len(row.get("failures") or []) for row in all_rows)
    failures += sum(1 for row in stage_rows if row.get("status") == "failed")
    retries = sum(_retry_count(row) for row in cycle_rows)
    fallbacks = sum(1 for row in cycle_rows if row.get("llm_fallback_used"))
    return {
        "actions": dict(actions),
        "cycle_count": len(cycle_rows),
        "stage_count": len(stage_rows),
        "human_event_count": len(human_event_rows),
        "tickers": ", ".join(tickers) if tickers else "-",
        "guardrails": guardrails,
        "failures": failures,
        "retries": retries,
        "fallbacks": fallbacks,
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }


def _retry_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("retry_count") or 0)
    except (TypeError, ValueError):
        return 0


def _cycle_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": f"cycle:{row.get('timestamp')}:{row.get('ticker')}",
        "timestamp": row.get("timestamp"),
        "ticker": row.get("ticker"),
        "action": row.get("action"),
        "outcome": row.get("outcome"),
        "confidence": row.get("confidence"),
        "cycle_summary": row.get("cycle_summary"),
        "rationale": (row.get("rationale") or "")[:500],
        "instruction_id": row.get("instruction_id"),
        "cancelled_by": row.get("cancelled_by"),
        "reversal_of": row.get("reversal_of"),
        "retry_count": row.get("retry_count"),
        "failure_type": row.get("failure_type"),
        "quantity": (row.get("decision") or {}).get("quantity"),
        "guardrails_triggered": row.get("guardrails_triggered") or [],
        "failures": row.get("failures") or [],
        "portfolio_after": _portfolio_after(row),
        "details": row,
    }


def _stage_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": row.get("timestamp"),
        "ticker": row.get("ticker"),
        "stage": row.get("stage"),
        "status": row.get("status"),
        "message": (row.get("message") or "")[:500],
        "details": row.get("details") or {},
    }


def _grouped_human_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in events:
        key = str(row.get("note_id") or row.get("instruction_id") or row.get("timestamp") or len(order))
        if key not in groups:
            groups[key] = {
                "entity_id": f"human:{key}",
                "note_id": key,
                "original_note": "",
                "events": [],
                "tickers": [],
                "instruction_ids": [],
                "details": {},
                "statuses": [],
            }
            order.append(key)
        group = groups[key]
        if not group["original_note"] and str(row.get("status") or "").lower() == "received":
            group["original_note"] = str(row.get("note") or "")
        group["events"].append(
            {
                "timestamp": row.get("timestamp"),
                "status": row.get("status"),
                "instruction_id": row.get("instruction_id"),
                "ticker": row.get("ticker"),
                "note": (row.get("note") or "")[:500],
                "details": row.get("details") or {},
            }
        )
        ticker = str(row.get("ticker") or "")
        if ticker and ticker != "HUMAN" and ticker not in group["tickers"]:
            group["tickers"].append(ticker)
        instruction_id = str(row.get("instruction_id") or "")
        if instruction_id and instruction_id not in group["instruction_ids"]:
            group["instruction_ids"].append(instruction_id)
        details = row.get("details")
        if isinstance(details, dict):
            group["details"].update({k: v for k, v in details.items() if v is not None})
        group["statuses"].append(str(row.get("status") or ""))
        if not group["original_note"]:
            group["original_note"] = str(group["events"][0].get("note") or "")
    return [groups[k] for k in order]


def _portfolio_after(row: dict[str, Any]) -> dict[str, Any]:
    execution = row.get("execution_result") or {}
    portfolio = execution.get("portfolio_after") or {}
    if portfolio:
        return portfolio
    return row.get("market_snapshot", {}).get("portfolio_after") or {}


def _diversification_series(cycle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from trading_agent.core.sector_classifier import SectorClassifier
        from trading_agent.core.portfolio import normalize_positions

        classifier = SectorClassifier()
    except ImportError:
        return []
    series: list[dict[str, Any]] = []
    for row in cycle_rows:
        portfolio = _portfolio_after(row) or {}
        positions = portfolio.get("positions")
        normalized = normalize_positions(positions) if positions else {}
        if not normalized:
            continue
        totals: dict[str, float] = {}
        total_value = 0.0
        for symbol, data in normalized.items():
            sector = classifier.classify(symbol, llm_client=None)
            try:
                mv = float(data.get("market_value") or 0.0)
            except (TypeError, ValueError):
                mv = 0.0
            totals[sector] = totals.get(sector, 0.0) + mv
            total_value += mv
        if total_value <= 0:
            continue
        breakdown = {s: round(v / total_value * 100, 1) for s, v in sorted(totals.items())}
        series.append(
            {
                "timestamp": row.get("timestamp"),
                "ticker": row.get("ticker"),
                "action": row.get("action"),
                "breakdown": breakdown,
            }
        )
    return series


def _collect_evidence(cycle_rows, stage_rows, human_event_rows) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in cycle_rows + stage_rows + human_event_rows:
        details = row.get("details") if isinstance(row, dict) else None
        if not isinstance(details, dict):
            continue
        evidence_list = details.get("evidence") or []
        if not isinstance(evidence_list, list):
            continue
        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue
            source_id = str(ev.get("source_id") or "")
            if not source_id:
                continue
            if source_id not in seen:
                seen[source_id] = {
                    "source_id": source_id,
                    "title": ev.get("title"),
                    "url": ev.get("url"),
                    "publisher": ev.get("publisher"),
                    "published_at": ev.get("published_at"),
                    "provider": ev.get("provider"),
                    "is_inference": bool(ev.get("is_inference", False)),
                    "clickable": bool(ev.get("url") and str(ev.get("url")).startswith(("http://", "https://"))),
                    "cited_in": [],
                }
            seen[source_id]["cited_in"].append(
                {
                    "timestamp": row.get("timestamp"),
                    "entry_type": row.get("entry_type"),
                    "stage": row.get("stage"),
                    "ticker": row.get("ticker"),
                }
            )
    return list(seen.values())


_SHELL_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trading Agent Journal</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5d6978;
      --line: #d9dee7;
      --soft-line: #edf0f5;
      --buy: #147d4f;
      --sell: #b42318;
      --hold: #667085;
      --wait: #1769aa;
      --warn: #b54708;
      --fail: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }
    main.shell { width: min(1680px, calc(100% - 48px)); margin: 22px auto 42px; }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }
    h1 { font-size: 24px; margin: 0 0 4px; }
    h2 { font-size: 18px; margin: 0 0 10px; }
    h3 { font-size: 14px; margin: 16px 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
    .meta { color: var(--muted); font-size: 13px; }
    .source { max-width: 460px; text-align: right; overflow-wrap: anywhere; }
    .generated { font-size: 12px; color: var(--muted); }
    nav.tabs {
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
      margin-bottom: 16px;
    }
    .tab-button {
      background: transparent;
      border: 1px solid transparent;
      border-bottom: none;
      border-radius: 6px 6px 0 0;
      padding: 8px 14px;
      font-size: 13px;
      color: var(--muted);
      cursor: pointer;
      font-family: inherit;
    }
    .tab-button:hover { background: #eef1f5; color: var(--text); }
    .tab-button.active {
      background: var(--panel);
      color: var(--text);
      border-color: var(--line);
      border-bottom-color: var(--panel);
      margin-bottom: -1px;
      font-weight: 600;
    }
    .tab-button.has-new::after {
      content: ""; display: inline-block; width: 7px; height: 7px;
      margin-left: 6px; border-radius: 50%; background: #1769aa;
      vertical-align: 1px;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .stat-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      min-height: 70px;
      position: relative;
      overflow: hidden;
    }
    .stat-card::before {
      content: "";
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 4px;
      background: #d9dee7;
    }
    .stat-card.info::before { background: #1769aa; }
    .stat-card.good::before { background: #147d4f; }
    .stat-card.warn::before { background: #b54708; }
    .stat-card span { display: block; color: var(--muted); font-size: 12px; }
    .stat-card strong { display: block; font-size: 16px; margin-top: 4px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { font-size: 18px; }
    .metric .split { display: flex; gap: 8px; }
    .metric .split b { font-size: 13px; }
    .metric .split b.buy { color: var(--buy); }
    .metric .split b.sell { color: var(--sell); }
    .metric .split b.hold { color: var(--hold); }
    .metric .split b.wait { color: var(--wait); }
    table.store {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    table.store th, table.store td {
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--soft-line);
      font-size: 13px;
      vertical-align: top;
    }
    table.store th { background: #f1f4f9; color: var(--muted); font-weight: 600; }
    table.store tr:last-child td { border-bottom: none; }
    table.store tr.entity-row { cursor: pointer; }
    table.store tr.entity-row:hover { background: #f8fafc; }
    table.store tr.entity-row td { user-select: none; -webkit-user-select: none; }
    .lifecycle-card.entity-row .lc-head,
    .reply-card.entity-row .rc-meta { cursor: pointer; }
    .table-scroll { max-height: 520px; overflow: auto; border-radius: 8px; margin-bottom: 14px; }
    .table-scroll table.store { min-width: 720px; }
    .table-scroll table.store th { position: sticky; top: 0; z-index: 1; }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
    }
    .badge.active, .badge.completed, .badge.confirmed { background: #dcfce7; color: #166534; }
    .badge.inactive, .badge.abandoned, .badge.rejected, .badge.expired, .badge.cancelled { background: #fee2e2; color: #991b1b; }
    .badge.pending, .badge.queued, .badge.modified { background: #fef3c7; color: #92400e; }
    .badge.triggered, .badge.filled, .badge.submitted { background: #dbeafe; color: #1e40af; }
    .badge.skipped, .badge.waiting { background: #f3f4f6; color: #6b7280; }
    .lifecycle-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 10px;
    }
    .lifecycle-card .lc-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }
    .lifecycle-card .lc-note { font-weight: 600; }
    .lifecycle-card .lc-meta { color: var(--muted); font-size: 12px; }
    .lifecycle-card .lc-steps {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 6px;
    }
    .lifecycle-card .lc-step {
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      background: #eef1f5;
      color: var(--muted);
    }
    .lifecycle-card .lc-summary {
      margin-top: 6px;
      font-size: 13px;
      color: var(--text);
    }
    .lifecycle-card .lc-next-step {
      margin-top: 4px;
      font-size: 12px;
      color: var(--muted);
      font-style: italic;
    }
    .lifecycle-card .lc-evidence {
      margin-top: 6px;
      font-size: 12px;
    }
    .lifecycle-card .lc-evidence a { color: var(--wait); text-decoration: none; }
    .lifecycle-card .lc-evidence a:hover { text-decoration: underline; }
    .lifecycle-card .lc-evidence .inference { color: var(--warn); font-style: italic; }
    .reply-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 10px;
    }
    .reply-card .rc-meta { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .reply-card .rc-note { font-style: italic; color: var(--muted); margin-bottom: 6px; }
    .reply-card .rc-body { white-space: pre-wrap; }
    .empty { color: var(--muted); padding: 24px; text-align: center; background: var(--panel); border: 1px dashed var(--line); border-radius: 8px; }
    .bar {
      display: flex;
      height: 8px;
      border-radius: 999px;
      overflow: hidden;
      background: #eef1f5;
      margin-bottom: 3px;
    }
    .bar > div { height: 100%; }
    .bar-labels { display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px; }
    .refresh-indicator {
      position: fixed;
      bottom: 12px;
      right: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 11px;
      color: var(--muted);
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .refresh-indicator.stale { color: var(--warn); border-color: var(--warn); }
    .modal-backdrop {
      display: none; position: fixed; inset: 0; z-index: 100;
      background: rgba(15, 23, 42, 0.45); padding: 4vh 18px;
    }
    .modal-backdrop.open { display: flex; align-items: flex-start; justify-content: center; }
    .modal-card {
      width: min(920px, 100%); max-height: 92vh; overflow: auto;
      background: var(--panel); border-radius: 10px; box-shadow: 0 20px 60px rgba(0,0,0,.25);
    }
    .modal-head {
      position: sticky; top: 0; display: flex; justify-content: space-between;
      align-items: center; gap: 12px; padding: 14px 18px; background: var(--panel);
      border-bottom: 1px solid var(--line); z-index: 1;
    }
    .modal-head h2 { margin: 0; }
    .modal-close { border: 0; background: #eef1f5; border-radius: 6px; padding: 6px 10px; cursor: pointer; font-size: 13px; }
    .modal-body { padding: 16px 18px 24px; }
    .modal-user-section { background: #f8fafc; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }
    .modal-user-section .mu-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }
    .modal-user-section .mu-value { font-size: 14px; color: var(--text); }
    .modal-user-section .mu-value.big { font-size: 20px; font-weight: 700; }
    .modal-kpi-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
    .modal-kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; flex: 1; min-width: 120px; }
    .modal-kpi .mk-label { font-size: 11px; color: var(--muted); text-transform: uppercase; }
    .modal-kpi .mk-value { font-size: 16px; font-weight: 600; margin-top: 2px; }
    .modal-evidence { margin-bottom: 14px; }
    .modal-evidence a { color: var(--wait); font-size: 13px; }
    .modal-section { margin-top: 12px; }
    .modal-section h3 { margin-top: 0; }
    .modal-section-content { background: #f8fafc; border-left: 3px solid #d9dee7; padding: 10px 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
    .modal-section pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }
    .modal-tech { margin-top: 14px; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .modal-tech .tech-inner { padding: 12px 14px; }
    .detail-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 8px; }
    .detail-item { background: #f8fafc; border: 1px solid var(--soft-line); border-radius: 6px; padding: 8px 10px; overflow-wrap: anywhere; }
    .detail-item span { display: block; font-size: 11px; color: var(--muted); text-transform: uppercase; }
    .modal-rationale { font-size: 13px; line-height: 1.6; color: var(--text); background: #f8fafc; border-left: 3px solid var(--wait); padding: 10px 12px; border-radius: 0 6px 6px 0; margin-bottom: 10px; }
    .modal-risks { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .modal-risk-tag { background: #fef3c7; color: #92400e; border-radius: 4px; padding: 2px 8px; font-size: 12px; }
    .modal-guardrail-tag { background: #fee2e2; color: #991b1b; border-radius: 4px; padding: 2px 8px; font-size: 12px; }
    .copy-button { margin-left: 7px; border: 1px solid var(--line); background: #fff; border-radius: 5px; padding: 2px 7px; color: var(--wait); cursor: pointer; font-size: 11px; transition: background 0.15s, color 0.15s; }
    .copy-button.copied { background: #dcfce7; color: #166534; border-color: #86efac; }
    @media (max-width: 760px) {
      main.shell { width: min(100% - 20px, 1180px); margin-top: 14px; }
      .summary, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>Trading Agent Journal</h1>
        <div class="meta" id="run-meta">Loading…</div>
      </div>
      <div class="meta source" id="source-meta"></div>
    </header>
    <nav class="tabs" id="tab-nav">
      <button class="tab-button active" data-tab="overview" onclick="showTab('overview', this)">Overview</button>
      <button class="tab-button" data-tab="trades" onclick="showTab('trades', this)">Trades</button>
      <button class="tab-button" data-tab="human" onclick="showTab('human', this)">Human Messages</button>
      <button class="tab-button" data-tab="confirmations" onclick="showTab('confirmations', this)">Confirmations</button>
      <button class="tab-button" data-tab="scheduled" onclick="showTab('scheduled', this)">Scheduled &amp; Conditional</button>
      <button class="tab-button" data-tab="constraints" onclick="showTab('constraints', this)">Constraints</button>
      <button class="tab-button" data-tab="replies" onclick="showTab('replies', this)">Agent Replies</button>
      <button class="tab-button" data-tab="diversification" onclick="showTab('diversification', this)">Diversification</button>
      <button class="tab-button" data-tab="evidence" onclick="showTab('evidence', this)">Evidence</button>
      <button class="tab-button" data-tab="audit" onclick="showTab('audit', this)">Technical Audit</button>
    </nav>
    <section id="tab-overview" class="tab-panel active"></section>
    <section id="tab-trades" class="tab-panel"></section>
    <section id="tab-human" class="tab-panel"></section>
    <section id="tab-confirmations" class="tab-panel"></section>
    <section id="tab-scheduled" class="tab-panel"></section>
    <section id="tab-constraints" class="tab-panel"></section>
    <section id="tab-replies" class="tab-panel"></section>
    <section id="tab-diversification" class="tab-panel"></section>
    <section id="tab-evidence" class="tab-panel"></section>
    <section id="tab-audit" class="tab-panel"></section>
  </main>
  <div class="modal-backdrop" id="entity-modal" onclick="closeModal(event)">
    <article class="modal-card" onclick="event.stopPropagation()">
      <div class="modal-head"><h2 id="modal-title">Details</h2><button class="modal-close" onclick="closeModal()">Close</button></div>
      <div class="modal-body" id="modal-body"></div>
    </article>
  </div>
  <div class="refresh-indicator" id="refresh-indicator">—</div>
  <script>
    let DASHBOARD_DATA = null;
    let activeTab = "overview";
    let modalEntityId = null;
    let entityIndex = {};

    function showTab(name, button) {
      document.querySelectorAll(".tab-button").forEach((b) => b.classList.remove("active"));
      if (button) button.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach((p) => {
        p.classList.toggle("active", p.id === "tab-" + name);
      });
      activeTab = name;
      renderTab(name);
      markTabSeen(name);
    }

    function esc(value) {
      return String(value == null ? "-" : value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function escAttr(value) {
      return esc(value).replaceAll('"', "&quot;").replaceAll("'", "&#39;");
    }

    function entityRow(entityId) {
      return ' class="entity-row" data-entity-id="' + escAttr(entityId) + '" onclick="openEntity(this.dataset.entityId)"';
    }

    function buildEntityIndex(data) {
      const index = {};
      const add = (item, id, title) => {
        if (!item) return;
        item.entity_id = item.entity_id || id;
        item.entity_title = title;
        index[item.entity_id] = item;
      };
      (data.cycles || []).forEach((x, i) => add(x, "cycle:" + i, (x.action || "Decision") + " " + (x.ticker || "")));
      (data.human_events || []).forEach((x, i) => add(x, "human:" + i, "Human message"));
      (data.pending_confirmations || []).forEach((x, i) => add(x, "confirmation:" + (x.confirmation_id || i), "Confirmation " + (x.confirmation_id || "")));
      (data.scheduled_actions || []).forEach((x, i) => add(x, "scheduled:" + (x.action_id || i), "Scheduled action"));
      (data.conditional_orders || []).forEach((x, i) => add(x, "conditional:" + (x.order_id || i), "Conditional order"));
      (data.constraints || []).forEach((x, i) => add(x, "constraint:" + (x.constraint_id || i), "Constraint"));
      (data.agent_replies || []).forEach((x, i) => add(x, "reply:" + i, "Agent reply"));
      (data.evidence || []).forEach((x, i) => add(x, "evidence:" + (x.source_id || i), "Evidence source"));
      (data.instruction_ledger || []).forEach((x, i) => add(x, "ledger:" + (x.instruction_id || x.request_id || i), "Instruction"));
      (data.stages || []).forEach((x, i) => add(x, "stage:" + i, "Stage " + (x.stage || "")));
      entityIndex = index;
    }

    function detailValue(value) {
      if (value == null || value === "") return "-";
      if (typeof value === "object") return esc(JSON.stringify(value));
      return esc(value);
    }

    function isShortScalar(value) {
      if (value == null || typeof value === "object") return false;
      const text = String(value);
      return text.length <= 80 && !text.includes("\\n");
    }

    function modalTextSection(title, value, raw) {
      if (value == null || value === "" || (Array.isArray(value) && !value.length)) return "";
      const body = typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
      const content = raw ? '<pre>' + esc(body) + '</pre>' : esc(body);
      return '<section class="modal-section"><h3>' + esc(title) + '</h3><div class="modal-section-content">' + content + '</div></section>';
    }

    function renderEntityModal() {
      if (!modalEntityId) return;
      const item = entityIndex[modalEntityId];
      if (!item) return;
      document.getElementById("modal-title").textContent = item.entity_title || "Details";
      const details = item.details || {};

      let html = "";

      const action = item.action || item.status || item.outcome || details.outcome_type || "";
      const ticker = item.ticker && item.ticker !== "HUMAN" ? item.ticker : "";
      const confidence = item.confidence != null ? Math.round(item.confidence * 100) + "%" : null;
      if (action || ticker) {
        html += '<div class="modal-kpi-row">';
        if (ticker) html += '<div class="modal-kpi"><div class="mk-label">Ticker</div><div class="mk-value">' + esc(ticker) + '</div></div>';
        if (action) html += '<div class="modal-kpi"><div class="mk-label">Decision</div><div class="mk-value">' + badge(action) + '</div></div>';
        if (confidence) html += '<div class="modal-kpi"><div class="mk-label">Confidence</div><div class="mk-value">' + esc(confidence) + '</div></div>';
        if (item.quantity != null && item.quantity > 0) html += '<div class="modal-kpi"><div class="mk-label">Quantity</div><div class="mk-value">' + esc(item.quantity) + '</div></div>';
        html += '</div>';
      }

      const userRequest = item.original_note || details.original_note;
      const agentReply = item.reply || details.agent_reply;
      const rawSummary = details.user_summary || item.cycle_summary || details.user_facing_summary;
      const userSummary = rawSummary && rawSummary !== userRequest && rawSummary !== agentReply ? rawSummary : null;
      if (userRequest) {
        html += '<div class="modal-user-section"><div class="mu-label">Your request</div><div class="mu-value">' + esc(userRequest) + '</div></div>';
      }
      if (agentReply && agentReply !== userRequest) {
        html += '<div class="modal-user-section"><div class="mu-label">What happened</div><div class="mu-value">' + esc(agentReply) + '</div></div>';
      } else if (userSummary) {
        html += '<div class="modal-user-section"><div class="mu-label">What happened</div><div class="mu-value">' + esc(userSummary) + '</div></div>';
      }

      const rationale = item.rationale || details.rationale;
      if (rationale) {
        html += '<div class="modal-rationale">' + esc(rationale) + '</div>';
      }

      const risks = (details.risks || []);
      const guardrails = (item.guardrails_triggered || details.guardrails_triggered || []);
      if (risks.length || guardrails.length) {
        html += '<div class="modal-risks">';
        for (const r of risks) html += '<span class="modal-risk-tag">⚠ ' + esc(r) + '</span>';
        for (const g of guardrails) html += '<span class="modal-guardrail-tag">🔒 ' + esc(g.replace("guardrail:", "")) + '</span>';
        html += '</div>';
      }

      const evidence = (details.evidence || item.evidence || []);
      if (evidence.length) {
        html += '<div class="modal-evidence">' + evidenceLinks(evidence) + '</div>';
      }

      const ops = item.proposed_operations || details.proposed_operations || details.operations || [];
      if (ops.length) {
        html += '<div class="modal-user-section"><div class="mu-label">Proposed operations</div>';
        for (const op of ops) {
          const side = ((op.action || op.side || op.intent_type || "").replace("forced_","").toUpperCase());
          const opTicker = op.ticker || op.symbol || "-";
          const qty = op.quantity || op.requested_quantity ? " × " + (op.quantity || op.requested_quantity) : "";
          const notional = op.notional || op.requested_notional_usd ? " ($" + (op.notional || op.requested_notional_usd) + ")" : "";
          html += '<div style="margin-top:6px"><span class="badge ' + side.toLowerCase() + '">' + esc(side) + '</span> ' + esc(opTicker) + esc(qty) + '<span style="color:var(--muted)">' + esc(notional) + '</span></div>';
        }
        html += '</div>';
      }

      const nextStep = details.next_step;
      if (nextStep) {
        html += '<div style="font-size:13px;color:var(--muted);margin-top:6px;font-style:italic">Next: ' + esc(nextStep) + '</div>';
      }

      const timeline = item.events || details.timeline || details.lifecycle;
      if (timeline && timeline.length) {
        html += '<div class="modal-user-section" style="margin-top:12px"><div class="mu-label">Timeline</div>';
        for (const ev of timeline) {
          html += '<div style="margin-top:6px;font-size:13px"><span class="badge ' + esc((ev.status||"").toLowerCase()) + '">' + esc(ev.status||"-") + '</span> ' + fmtTime(ev.timestamp) + (ev.note ? ' — ' + esc((ev.note||"").slice(0,120)) : "") + '</div>';
        }
        html += '</div>';
      }

      const preferred = ["timestamp","ticker","action","outcome","status","confidence","quantity","confirmation_id","workflow_status","instruction_id","request_id"];
      let cards = "";
      for (const key of preferred) {
        if (!isShortScalar(item[key])) continue;
        cards += '<div class="detail-item"><span>' + esc(key.replaceAll("_"," ")) + '</span>' + detailValue(item[key]) + '</div>';
      }

      const technical = { decision: details.decision, technical_opinion: details.technical_opinion, risk_assessment: details.risk_assessment, market_snapshot: details.market_snapshot, technical_details: details.technical_details, constraints_checked: details.constraints_checked };
      Object.keys(technical).forEach(k => technical[k] == null && delete technical[k]);
      const execution = details.execution_result || item.execution_result;
      const portfolioRisk = { portfolio_after: item.portfolio_after || details.portfolio_after, risk_assessment: details.risk_assessment, risk_limits: details.risk_limits };
      Object.keys(portfolioRisk).forEach(k => portfolioRisk[k] == null && delete portfolioRisk[k]);

      let techHtml = "";
      if (cards) techHtml += '<div class="detail-grid" style="margin-bottom:10px">' + cards + '</div>';
      if (execution) techHtml += modalTextSection("Execution result", execution, true);
      if (Object.keys(portfolioRisk).length) techHtml += modalTextSection("Portfolio / Risk", portfolioRisk, true);
      if (Object.keys(technical).length) techHtml += modalTextSection("Market / Technical", technical, true);
      techHtml += modalTextSection("Complete raw details", item.details || item, true);

      if (techHtml) {
        html += '<div class="modal-tech" id="modal-tech-section">';
        html += '<h3 style="padding:10px 14px;margin:0;background:#f1f4f9;border-bottom:1px solid var(--line)">Technical details</h3>';
        html += '<div class="tech-inner" id="modal-tech-inner">' + techHtml + '</div>';
        html += '</div>';
      }

      document.getElementById("modal-body").innerHTML = html;
    }

    async function copyConfirmationId(value, event) {
      if (event) event.stopPropagation();
      const btn = event && event.target;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(value);
        } else {
          const textarea = document.createElement("textarea");
          textarea.value = value;
          textarea.style.position = "fixed";
          textarea.style.opacity = "0";
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          textarea.remove();
        }
      } catch (error) { /* silent fail */ }
      if (btn) {
        const original = btn.textContent;
        btn.textContent = "✓ Copied";
        btn.classList.add("copied");
        setTimeout(() => { btn.textContent = original; btn.classList.remove("copied"); }, 1500);
      }
    }

    function openEntity(entityId) {
      modalEntityId = entityId;
      renderEntityModal();
      document.getElementById("entity-modal").classList.add("open");
    }

    function closeModal(event) {
      if (event && event.target !== document.getElementById("entity-modal")) return;
      modalEntityId = null;
      document.getElementById("entity-modal").classList.remove("open");
    }

    function tabSignature(name, data) {
      const sources = {
        overview: [data.generated_at, (data.cycles || []).length],
        trades: (data.cycles || []).map(x => [x.entity_id, x.action, x.outcome]),
        human: (data.human_events || []).map(x => [x.note_id, x.statuses]),
        confirmations: (data.pending_confirmations || []).map(x => [x.confirmation_id, x.status]),
        scheduled: [(data.scheduled_actions || []).map(x => [x.action_id, x.status]), (data.conditional_orders || []).map(x => [x.order_id, x.status])],
        constraints: (data.constraints || []).map(x => [x.constraint_id, x.active]),
        replies: (data.agent_replies || []).map(x => x.timestamp),
        diversification: data.diversification_series || [], evidence: (data.evidence || []).map(x => x.source_id),
        audit: [(data.instruction_ledger || []).map(x => [x.request_id, x.status, x.updated_at]), (data.stages || []).length],
      };
      return JSON.stringify(sources[name] || []);
    }

    function seenKey(name) { return "dashboardSeen:" + (DASHBOARD_DATA ? DASHBOARD_DATA.run_dir : "run") + ":" + name; }

    function markTabSeen(name) {
      if (!DASHBOARD_DATA) return;
      localStorage.setItem(seenKey(name), tabSignature(name, DASHBOARD_DATA));
      const button = document.querySelector('.tab-button[data-tab="' + name + '"]');
      if (button) button.classList.remove("has-new");
    }

    function updateNoveltyDots(data) {
      document.querySelectorAll(".tab-button").forEach(button => {
        const name = button.dataset.tab;
        const changed = tabHasContent(name, data) && localStorage.getItem(seenKey(name)) !== tabSignature(name, data);
        button.classList.toggle("has-new", changed && name !== activeTab);
      });
      markTabSeen(activeTab);
    }

    function tabHasContent(name, data) {
      const counts = {
        overview: (data.cycles || []).length + (data.human_events || []).length + (data.pending_confirmations || []).length,
        trades: (data.cycles || []).length,
        human: (data.human_events || []).length,
        confirmations: (data.pending_confirmations || []).length,
        scheduled: (data.scheduled_actions || []).length + (data.conditional_orders || []).length,
        constraints: (data.constraints || []).length,
        replies: (data.agent_replies || []).length,
        diversification: (data.diversification_series || []).length,
        evidence: (data.evidence || []).length,
        audit: (data.instruction_ledger || []).length + (data.stages || []).length,
      };
      return (counts[name] || 0) > 0;
    }

    function itemTimestamp(item) {
      const events = item && item.events;
      return item && (item.updated_at || item.timestamp || item.created_at || item.published_at || item.triggered_at) || (events && events.length ? events[events.length - 1].timestamp : null) || "";
    }

    function newestFirst(items) {
      return [...(items || [])].sort((a, b) => String(itemTimestamp(b)).localeCompare(String(itemTimestamp(a))));
    }

    function normalizeNewestFirst(data) {
      for (const key of ["cycles", "human_events", "stages", "constraints", "scheduled_actions", "conditional_orders", "pending_confirmations", "instruction_ledger", "agent_replies", "diversification_series", "evidence"]) {
        data[key] = newestFirst(data[key]);
      }
    }

    function wrapScrollableTables(panel) {
      panel.querySelectorAll("table.store").forEach(table => {
        if (table.parentElement && table.parentElement.classList.contains("table-scroll")) return;
        const wrapper = document.createElement("div");
        wrapper.className = "table-scroll";
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      });
    }

    function saveScrollPositions() {
      const positions = {};
      document.querySelectorAll(".table-scroll").forEach((el, i) => { positions[i] = el.scrollTop; });
      return positions;
    }

    function restoreScrollPositions(positions) {
      document.querySelectorAll(".table-scroll").forEach((el, i) => { if (positions[i] != null) el.scrollTop = positions[i]; });
    }

    function fmtTime(ts) {
      if (!ts) return "-";
      try {
        const d = new Date(ts);
        return d.toLocaleString();
      } catch (e) { return ts; }
    }

    function badge(status) {
      const cls = (status || "").toLowerCase();
      return '<span class="badge ' + cls + '">' + esc(status || "-") + '</span>';
    }

    function sectorBar(breakdown) {
      if (!breakdown || !Object.keys(breakdown).length) return '<span class="empty">—</span>';
      const COLORS = {
        technology: "#1769aa", automotive: "#147d4f", manufacturing: "#b54708",
        energy: "#b42318", healthcare: "#5b21b6", finance: "#0e7490",
        consumer: "#92400e", materials: "#0f766e", utilities: "#a16207",
        real_estate: "#be185d", communication: "#1d4ed8", industrials: "#7c2d12",
        bonds: "#475569", cash: "#94a3b8", crypto: "#db2777", other: "#667085",
      };
      let bars = "";
      let labels = "";
      for (const [sector, pct] of Object.entries(breakdown)) {
        const color = COLORS[sector] || "#667085";
        bars += '<div style="width:' + pct + '%;background:' + color + '" title="' + esc(sector) + ' ' + pct + '%"></div>';
        labels += '<span style="color:' + color + '">' + esc(sector.slice(0, 4)) + ' ' + pct + '%</span>';
      }
      return '<div class="bar">' + bars + '</div><div class="bar-labels">' + labels + '</div>';
    }

    function evidenceLinks(evidenceList) {
      if (!evidenceList || !evidenceList.length) return "";
      let html = '<div class="lc-evidence"><strong>Evidence:</strong><ul style="margin:2px 0 0 16px;padding:0">';
      for (const ev of evidenceList) {
        if (ev.is_inference) {
          html += '<li><span class="inference">[inference]</span> ' + esc(ev.title || ev.excerpt || "") + '</li>';
        } else if (ev.url && (ev.url.startsWith("http://") || ev.url.startsWith("https://"))) {
          html += '<li><a href="' + esc(ev.url) + '" target="_blank" rel="noopener">' + esc(ev.title || ev.url) + '</a> <span style="color:#5d6978">(' + esc(ev.publisher || ev.provider || "") + ')</span></li>';
        } else {
          html += '<li>' + esc(ev.title || ev.excerpt || "") + ' <span style="color:#5d6978">(' + esc(ev.provider || "") + ')</span></li>';
        }
      }
      html += '</ul></div>';
      return html;
    }

    let _lastTabFingerprint = {};

    function renderTab(name) {
      if (!DASHBOARD_DATA) return;
      const panel = document.getElementById("tab-" + name);
      if (!panel) return;
      const fingerprint = tabSignature(name, DASHBOARD_DATA);
      if (_lastTabFingerprint[name] === fingerprint && panel.children.length > 0) return;
      _lastTabFingerprint[name] = fingerprint;
      const renderers = {
        overview: renderOverview,
        trades: renderTrades,
        human: renderHuman,
        confirmations: renderConfirmations,
        scheduled: renderScheduled,
        constraints: renderConstraints,
        replies: renderReplies,
        diversification: renderDiversification,
        evidence: renderEvidence,
        audit: renderAudit,
      };
      panel.innerHTML = (renderers[name] || renderOverview)(DASHBOARD_DATA);
      wrapScrollableTables(panel);
    }

    function renderOverview(data) {
      const s = data.stats;
      const actions = s.actions || {};
      let html = '<h2>Run Summary</h2>';
      html += '<div class="summary">';
      html += '<div class="stat-card info"><span>Cycles</span><strong>' + s.cycle_count + '</strong></div>';
      html += '<div class="stat-card info"><span>Human messages</span><strong>' + s.human_event_count + '</strong></div>';
      html += '<div class="stat-card ' + (s.failures ? "warn" : "good") + '"><span>Failures</span><strong>' + s.failures + '</strong></div>';
      html += '<div class="stat-card ' + (s.guardrails || s.fallbacks ? "warn" : "good") + '"><span>Guardrails / Fallbacks</span><strong>' + s.guardrails + ' / ' + s.fallbacks + '</strong></div>';
      html += '<div class="stat-card info"><span>Retries</span><strong>' + s.retries + '</strong></div>';
      html += '</div>';
      html += '<div class="grid">';
      html += '<div class="metric"><span>BUY</span><strong style="color:#147d4f">' + (actions.BUY || 0) + '</strong></div>';
      html += '<div class="metric"><span>SELL</span><strong style="color:#b42318">' + (actions.SELL || 0) + '</strong></div>';
      html += '<div class="metric"><span>HOLD</span><strong style="color:#667085">' + (actions.HOLD || 0) + '</strong></div>';
      html += '<div class="metric"><span>WAIT</span><strong style="color:#1769aa">' + (actions.WAIT || 0) + '</strong></div>';
      html += '<div class="metric"><span>Tickers</span><strong>' + esc(s.tickers) + '</strong></div>';
      html += '<div class="metric"><span>Pending confirmations</span><strong>' + (data.pending_confirmations || []).filter(c => c.status === "pending").length + '</strong></div>';
      html += '</div>';
      html += '<h3>Last 5 cycles</h3>';
      const recent = (data.cycles || []).slice(0, 5);
      if (!recent.length) {
        html += '<div class="empty">No cycles recorded yet.</div>';
      } else {
        html += '<table class="store"><thead><tr><th>Timestamp</th><th>Ticker</th><th>Action</th><th>Outcome</th><th>Summary</th></tr></thead><tbody>';
        for (const c of recent) {
          html += '<tr' + entityRow(c.entity_id) + '><td>' + fmtTime(c.timestamp) + '</td><td>' + esc(c.ticker) + '</td><td>' + esc(c.action) + '</td><td>' + badge(c.outcome) + '</td><td>' + esc((c.cycle_summary || c.rationale || "").slice(0, 160)) + '</td></tr>';
        }
        html += '</tbody></table>';
      }
      return html;
    }

    function renderTrades(data) {
      const cycles = data.cycles || [];
      if (!cycles.length) return '<h2>Decisions</h2><div class="empty">No decisions recorded.</div>';
      let html = '<h2>Decisions</h2><p class="meta">BUY, SELL, HOLD and WAIT are all retained. Select a row for the complete rationale and technical context.</p>';
      html += '<table class="store"><thead><tr><th>Timestamp</th><th>Ticker</th><th>Action</th><th>Outcome</th><th>Summary</th></tr></thead><tbody>';
      for (const c of cycles) {
        html += '<tr' + entityRow(c.entity_id) + '><td>' + fmtTime(c.timestamp) + '</td><td>' + esc(c.ticker) + '</td><td>' + esc(c.action) + '</td><td>' + badge(c.outcome) + '</td><td>' + esc((c.cycle_summary || c.rationale || "").slice(0, 180)) + '</td></tr>';
      }
      html += '</tbody></table>';
      return html;
    }

    function renderHuman(data) {
      const groups = data.human_events || [];
      if (!groups.length) return '<h2>Human Messages</h2><div class="empty">No human messages in this run.</div>';
      let html = '<h2>Human Messages</h2>';
      html += '<p class="meta">Each card is one request lifecycle: received → resolved → queued → active → completed/cancelled. Intermediate states are grouped, never duplicated.</p>';
      for (const g of groups.slice(0, 30)) {
        const first = g.events[0] || {};
        const last = g.events[g.events.length - 1] || {};
        const details = g.details || {};
        const userSummary = details.user_summary || first.note || "";
        const nextStep = details.next_step || "";
        const evidence = details.evidence || [];
        const outcomeType = details.outcome_type || "";
        html += '<div class="lifecycle-card entity-row" data-entity-id="' + escAttr(g.entity_id) + '" onclick="openEntity(this.dataset.entityId)">';
        html += '<div class="lc-head"><span class="lc-note">' + esc((g.original_note || first.note || "").slice(0, 160)) + '</span><span class="lc-meta">' + fmtTime(last.timestamp) + '</span></div>';
        html += '<div class="lc-meta">Tickers: ' + esc(g.tickers.join(", ") || "-") + ' &middot; Outcome: <code>' + esc(outcomeType) + '</code></div>';
        if (userSummary && userSummary !== first.note) {
          html += '<div class="lc-summary">' + esc(userSummary) + '</div>';
        }
        if (nextStep) {
          html += '<div class="lc-next-step">Next: ' + esc(nextStep) + '</div>';
        }
        html += '<div class="lc-steps">';
        for (const ev of g.events) {
          html += '<span class="lc-step">' + esc(ev.status) + ' &middot; ' + fmtTime(ev.timestamp) + '</span>';
        }
        html += '</div>';
        html += evidenceLinks(evidence);
        html += '</div>';
      }
      return html;
    }

    function renderConfirmations(data) {
      const items = data.pending_confirmations || [];
      if (!items.length) return '<h2>Confirmations</h2><div class="empty">No pending confirmations in this run.</div>';
      let html = '<h2>Confirmations</h2>';
      html += '<p class="meta">Risky or ambiguous proposals awaiting user response. Reply with "confirm CF-XXXX", "reject CF-XXXX", or "confirm CF-XXXX but &lt;modification&gt;".</p>';
      for (const c of items) {
        const ops = c.proposed_operations || [];
        const details = c.details || {};
        const categories = (details.categories || details.sector_categories || []);
        const actions = ops.map(o => ((o.action || o.side || o.intent_type || "").replace("forced_","").toUpperCase() + (o.ticker ? " " + o.ticker : "") + (o.quantity ? " ×" + o.quantity : "") + (o.notional || o.requested_notional_usd ? " $" + (o.notional || o.requested_notional_usd) : "")).trim()).filter(Boolean);
        html += '<div class="lifecycle-card entity-row" data-entity-id="' + escAttr(c.entity_id) + '" onclick="openEntity(this.dataset.entityId)">';
        html += '<div class="lc-head">';
        html += '<span class="lc-note"><code>' + esc(c.confirmation_id) + '</code><button class="copy-button" data-confirmation-id="' + escAttr(c.confirmation_id) + '" onclick="copyConfirmationId(this.dataset.confirmationId, event)">Copy</button></span>';
        html += '<span>' + badge(c.status) + '</span>';
        html += '</div>';
        html += '<div class="lc-meta">' + fmtTime(c.created_at) + (c.expires_at ? ' · Expires: ' + fmtTime(c.expires_at) : '') + '</div>';
        if (c.proposal_text) {
          html += '<div class="lc-summary" style="margin-top:6px">' + esc(c.proposal_text) + '</div>';
        }
        if (actions.length) {
          html += '<div class="lc-steps" style="margin-top:8px">';
          for (const a of actions) {
            html += '<span class="lc-step" style="background:#dbeafe;color:#1e40af">' + esc(a) + '</span>';
          }
          html += '</div>';
        }
        if (categories.length) {
          html += '<div class="lc-meta" style="margin-top:4px">Sectors: ' + esc(categories.join(", ")) + '</div>';
        }
        html += '</div>';
      }
      return html;
    }

    function renderScheduled(data) {
      const sched = data.scheduled_actions || [];
      const cond = data.conditional_orders || [];
      if (!sched.length && !cond.length) return '<h2>Scheduled &amp; Conditional</h2><div class="empty">No scheduled actions or conditional orders in this run.</div>';
      let html = '<h2>Scheduled &amp; Conditional</h2>';
      if (sched.length) {
        html += '<h3>Scheduled Actions</h3>';
        html += '<table class="store"><thead><tr><th>ID</th><th>Intent / Ticker</th><th>Trigger</th><th>Value</th><th>Status</th></tr></thead><tbody>';
        for (const a of sched) {
          html += '<tr' + entityRow(a.entity_id) + '><td><code>' + esc(a.action_id) + '</code></td><td>' + esc(a.wrapped_intent_type) + ' / ' + esc(a.target_ticker || "-") + '</td><td>' + esc(a.trigger_type) + '</td><td>' + esc(a.trigger_value) + '</td><td>' + badge(a.status) + '</td></tr>';
        }
        html += '</tbody></table>';
      }
      if (cond.length) {
        html += '<h3>Conditional Orders</h3>';
        html += '<table class="store"><thead><tr><th>Order ID</th><th>Ticker</th><th>Trigger / Price</th><th>Fraction</th><th>Status</th></tr></thead><tbody>';
        for (const o of cond) {
          html += '<tr' + entityRow(o.entity_id) + '><td><code>' + esc(o.order_id) + '</code></td><td>' + esc(o.ticker) + '</td><td>' + esc(o.trigger_type) + ' / ' + esc(o.trigger_price) + '</td><td>' + esc(o.trigger_fraction) + '</td><td>' + badge(o.status) + '</td></tr>';
        }
        html += '</tbody></table>';
      }
      return html;
    }

    function renderConstraints(data) {
      const items = data.constraints || [];
      if (!items.length) return '<h2>Constraints</h2><div class="empty">No constraints set in this run.</div>';
      let html = '<h2>Constraints</h2>';
      html += '<p class="meta">Persistent rules the user has set. Active constraints block matching buys unless explicitly overridden for a single instruction.</p>';
      html += '<table class="store"><thead><tr><th>ID</th><th>Type</th><th>Value</th><th>Status</th><th>Rationale</th></tr></thead><tbody>';
      for (const c of items) {
        html += '<tr' + entityRow(c.entity_id) + '><td><code>' + esc(c.constraint_id) + '</code></td><td>' + esc(c.type) + '</td><td>' + esc(c.value) + '</td><td>' + badge(c.active ? "active" : "inactive") + '</td><td>' + esc((c.rationale || "").slice(0, 160)) + '</td></tr>';
      }
      html += '</tbody></table>';
      return html;
    }

    function renderReplies(data) {
      const items = data.agent_replies || [];
      if (!items.length) return '<h2>Agent Replies</h2><div class="empty">No agent replies in this run.</div>';
      let html = '<h2>Agent Replies</h2>';
      html += '<p class="meta">Textual responses to information requests and advisories. Most recent first.</p>';
      for (const r of items) {
        html += '<div class="reply-card entity-row" data-entity-id="' + escAttr(r.entity_id) + '" onclick="openEntity(this.dataset.entityId)">';
        html += '<div class="rc-meta">' + fmtTime(r.timestamp) + ' &middot; ' + badge(r.question_type || "general") + '</div>';
        html += '<div class="rc-note">User note: ' + esc(r.note) + '</div>';
        html += '<div class="rc-body">' + esc(r.reply) + '</div>';
        html += '</div>';
      }
      return html;
    }

    function renderDiversification(data) {
      const series = data.diversification_series || [];
      if (!series.length) return '<h2>Diversification</h2><div class="empty">No portfolio snapshots with open positions yet.</div>';
      let html = '<h2>Diversification</h2>';
      html += '<p class="meta">Sector allocation over time. Uses the dynamic SectorClassifier so external buys are coloured correctly.</p>';
      const latest = series[0];
      html += '<h3>Latest snapshot</h3>';
      html += '<div style="margin-bottom:18px">' + sectorBar(latest.breakdown) + '</div>';
      html += '<h3>Recent history (last 15 cycles with positions)</h3>';
      html += '<table class="store"><thead><tr><th>Timestamp</th><th>Action</th><th>Ticker</th><th>Sector breakdown</th></tr></thead><tbody>';
      for (const item of series.slice(0, 15)) {
        html += '<tr><td>' + fmtTime(item.timestamp) + '</td><td>' + esc(item.action) + '</td><td>' + esc(item.ticker) + '</td><td>' + sectorBar(item.breakdown) + '</td></tr>';
      }
      html += '</tbody></table>';
      return html;
    }

    function renderEvidence(data) {
      const items = data.evidence || [];
      if (!items.length) return '<h2>Evidence</h2><div class="empty">No external evidence collected in this run.</div>';
      let html = '<h2>Evidence</h2>';
      html += '<p class="meta">Every external source cited by the resolver, with the locations where it was used. Real URLs are clickable; LLM inferences are labelled.</p>';
      html += '<table class="store"><thead><tr><th>Source ID</th><th>Title</th><th>Provider</th><th>Published</th><th>Cited in</th></tr></thead><tbody>';
      for (const ev of items) {
        const title = ev.clickable ? '<a href="' + esc(ev.url) + '" target="_blank" rel="noopener">' + esc(ev.title || ev.url) + '</a>' : (ev.is_inference ? '<span class="inference">[inference]</span> ' + esc(ev.title) : esc(ev.title));
        const citedIn = (ev.cited_in || []).length + ' location(s)';
        html += '<tr' + entityRow(ev.entity_id) + '><td><code>' + esc(ev.source_id) + '</code></td><td>' + title + '</td><td>' + esc(ev.provider) + (ev.publisher ? ' / ' + esc(ev.publisher) : '') + '</td><td>' + fmtTime(ev.published_at) + '</td><td>' + citedIn + '</td></tr>';
      }
      html += '</tbody></table>';
      return html;
    }

    function renderAudit(data) {
      let html = '<h2>Technical Audit</h2>';
      html += '<p class="meta">Confidence, provider, schema, retries, limits, and diagnostics. Not shown in main tabs.</p>';
      html += '<h3>Stats</h3>';
      html += renderStatsCards(data.stats || {});
      html += '<h3>Instruction Ledger</h3>';
      const ledger = data.instruction_ledger || [];
      if (!ledger.length) {
        html += '<div class="empty">No instructions recorded.</div>';
      } else {
        html += '<table class="store"><thead><tr><th>Request</th><th>Intent</th><th>Ticker</th><th>Status</th><th>Updated / Retries</th></tr></thead><tbody>';
        for (const e of ledger) {
          html += '<tr' + entityRow(e.entity_id) + '><td><code>' + esc(e.request_id) + '</code></td><td>' + esc(e.intent_type) + '</td><td>' + esc(e.target_ticker || "-") + '</td><td>' + badge(e.status) + '</td><td>' + fmtTime(e.updated_at) + ' / ' + esc(e.retry_count) + '</td></tr>';
        }
        html += '</tbody></table>';
      }
      html += '<h3>Recent Stages</h3>';
      const stages = data.stages || [];
      if (!stages.length) {
        html += '<div class="empty">No stages recorded.</div>';
      } else {
        html += '<table class="store"><thead><tr><th>Timestamp</th><th>Stage</th><th>Status</th><th>Ticker</th><th>Message</th></tr></thead><tbody>';
        for (const s of stages.slice(0, 50)) {
          html += '<tr' + entityRow(s.entity_id) + '><td>' + fmtTime(s.timestamp) + '</td><td>' + esc(s.stage) + '</td><td>' + badge(s.status) + '</td><td>' + esc(s.ticker) + '</td><td>' + esc((s.message || "").slice(0, 200)) + '</td></tr>';
        }
        html += '</tbody></table>';
      }
      return html;
    }

    function renderStatsCards(stats) {
      const labels = {
        cycle_count: "Cycles", stage_count: "Stages", human_event_count: "Human events",
        guardrails: "Guardrails", failures: "Failures", retries: "Retries", fallbacks: "Fallbacks",
        tickers: "Tickers", first_timestamp: "First event", last_timestamp: "Last event",
      };
      let html = '<div class="summary">';
      for (const [key, label] of Object.entries(labels)) {
        let value = stats[key];
        if (key.endsWith("timestamp")) value = fmtTime(value);
        html += '<div class="stat-card ' + ((key === "failures" || key === "fallbacks") && value ? "warn" : "info") + '"><span>' + label + '</span><strong>' + esc(value) + '</strong></div>';
      }
      html += '</div>';
      return html;
    }

    function applyData(data) {
      const scrollY = window.scrollY;
      const tableScrollPositions = saveScrollPositions();
      DASHBOARD_DATA = data;
      normalizeNewestFirst(data);
      buildEntityIndex(data);
      document.getElementById("run-meta").textContent = "Run: " + data.run_dir + " · Generated: " + fmtTime(data.generated_at);
      document.getElementById("source-meta").textContent = "Projection: dashboard_data.js";
      renderTab(activeTab);
      updateNoveltyDots(data);
      if (modalEntityId) renderEntityModal();
      requestAnimationFrame(() => { window.scrollTo(0, scrollY); restoreScrollPositions(tableScrollPositions); });
      const indicator = document.getElementById("refresh-indicator");
      indicator.textContent = "Updated " + fmtTime(data.generated_at);
      indicator.classList.remove("stale");
    }

    function loadProjection() {
      const script = document.createElement("script");
      script.src = "dashboard_data.js?t=" + Date.now();
      script.onload = function () {
        if (window.DASHBOARD_DATA) {
          applyData(window.DASHBOARD_DATA);
        }
      };
      script.onerror = function () {
        const indicator = document.getElementById("refresh-indicator");
        indicator.textContent = "Projection unavailable";
        indicator.classList.add("stale");
      };
      document.body.appendChild(script);
    }

    loadProjection();
    setInterval(loadProjection, 1000);
  </script>
</body>
</html>
"""
