from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_agent.core.portfolio import float_or_none, normalize_positions


def render_dashboard(journal_path: Path, output_path: Path) -> Path:
    rows = _read_rows(journal_path)
    display_rows = _cycle_rows(rows)
    stats = _stats(rows)
    details_json = json.dumps([_details_payload(row) for row in display_rows], ensure_ascii=False)
    table_rows = "\n".join(_render_row(index, row) for index, row in enumerate(display_rows))
    human_events_html = _render_human_events(rows)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="10">
  <title>Trading Agent Journal</title>
  <style>
    :root {{
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
      --shadow: 0 10px 24px rgba(16, 24, 40, .06);
      --shadow-soft: 0 4px 14px rgba(16, 24, 40, .05);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    main.dashboard-shell {{ width: min(1680px, calc(100% - 48px)); margin: 22px auto 42px; }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 18px;
    }}
    h1 {{ font-size: 26px; line-height: 1.15; margin: 0 0 6px; letter-spacing: 0; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .source {{ max-width: 460px; text-align: right; overflow-wrap: anywhere; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .summary-item, .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow-soft);
    }}
    .summary-item {{ padding: 13px 14px; min-height: 76px; }}
    .stat-card {{ position: relative; overflow: hidden; }}
    .stat-card::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 4px;
      background: #d9dee7;
    }}
    .stat-card.good::before {{ background: #147d4f; }}
    .stat-card.warn::before {{ background: #b54708; }}
    .stat-card.info::before {{ background: #1769aa; }}
    .summary-item span, .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }}
    .summary-item strong {{ font-size: 14px; font-weight: 700; overflow-wrap: anywhere; }}
    .stat-value {{ display: block; font-size: 18px; line-height: 1.25; margin-top: 4px; }}
    .stat-sub {{ display: block; color: var(--muted); font-size: 12px; margin-top: 5px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      padding: 13px 14px;
    }}
    .metric strong {{ font-size: 22px; line-height: 1.15; }}
    .metric .split {{ display: flex; gap: 8px; flex-wrap: wrap; font-size: 16px; font-weight: 700; }}
    .human-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow-soft);
      margin: 0 0 16px;
      overflow: hidden;
    }}
    .human-panel h2 {{
      font-size: 14px;
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--soft-line);
      color: #344054;
    }}
    .human-list {{
      max-height: 330px;
      overflow: auto;
    }}
    .human-card {{
      border-bottom: 1px solid var(--soft-line);
      font-size: 13px;
      background: #fff;
    }}
    .human-card:last-child {{ border-bottom: 0; }}
    .human-card summary {{
      list-style: none;
      cursor: pointer;
      padding: 10px 12px;
    }}
    .human-card summary::-webkit-details-marker {{ display: none; }}
    .human-card summary::before {{
      content: "+";
      display: inline-grid;
      place-items: center;
      width: 18px;
      height: 18px;
      border: 1px solid var(--line);
      border-radius: 50%;
      margin-right: 8px;
      color: #344054;
      font-weight: 700;
      font-size: 12px;
    }}
    .human-card[open] summary::before {{ content: "-"; }}
    .human-summary-main {{
      display: inline;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .human-summary-sub {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 5px 0 0 28px;
      overflow-wrap: anywhere;
    }}
    .human-card-body {{
      padding: 0 12px 12px 40px;
    }}
    .human-steps {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 0 0 6px; }}
    .human-step-chip {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      background: #f7f8fa;
      color: #344054;
      font-size: 12px;
      font-weight: 700;
    }}
    .human-meta {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 6px; }}
    .human-chip {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      background: #fff;
      color: #344054;
      font-size: 12px;
      font-weight: 700;
    }}
    .human-chip.intent {{ border-color: #cfe2ff; background: #f5f9ff; color: #1769aa; }}
    .human-chip.target {{ border-color: #b7e4c7; background: #f0f9f3; color: #147d4f; }}
    .human-chip.status {{ border-color: #d9dee7; background: #f7f8fa; color: #344054; }}
    .human-detail {{ color: #344054; margin-top: 3px; overflow-wrap: anywhere; }}
    .human-detail span {{ color: var(--muted); font-size: 12px; font-weight: 700; margin-right: 4px; }}
    .human-target-list {{ display: grid; gap: 3px; margin-top: 3px; }}
    .human-target-row {{
      display: grid;
      grid-template-columns: minmax(72px, auto) 1fr;
      gap: 8px;
      align-items: start;
      color: #344054;
    }}
    .human-target-row b {{ color: #17202a; }}
    .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow-x: hidden;
      overflow-y: auto;
      box-shadow: var(--shadow);
      max-height: min(720px, calc(100vh - 260px));
    }}
    .table-wrap::-webkit-scrollbar {{ width: 10px; }}
    .table-wrap::-webkit-scrollbar-track {{ background: #f1f4f8; }}
    .table-wrap::-webkit-scrollbar-thumb {{ background: #c7ced8; border-radius: 999px; border: 2px solid #f1f4f8; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{
      border-bottom: 1px solid var(--soft-line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    tbody tr:hover {{ background: #fbfcfe; }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f1f4f8;
      color: #344054;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .03em;
      white-space: nowrap;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .cell-clamp {{
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 3;
      overflow: hidden;
    }}
    .summary-cell {{ line-height: 1.35; }}
    .outcome-cell {{ line-height: 1.35; }}
    .toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin: 16px 0 10px;
      flex-wrap: wrap;
    }}
    .filter-group {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .filter-button {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 6px 10px;
      cursor: pointer;
      font-weight: 700;
      color: #344054;
    }}
    .filter-button.active {{ background: #17202a; color: #fff; border-color: #17202a; }}
    .journal-row {{ cursor: pointer; }}
    .journal-row:focus-visible {{ outline: 2px solid #147d4f; outline-offset: -2px; }}
    .money, .numeric {{ font-family: Consolas, Monaco, monospace; font-variant-numeric: tabular-nums; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      color: #fff;
      font-weight: 700;
      font-size: 12px;
    }}
    .BUY {{ background: var(--buy); }}
    .SELL {{ background: var(--sell); }}
    .HOLD {{ background: var(--hold); }}
    .WAIT {{ background: var(--wait); }}
    .guardrail {{ color: var(--warn); font-weight: 700; font-size: 12px; }}
    .failed {{ color: var(--fail); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .time-main {{ font-weight: 700; white-space: nowrap; }}
    .time-sub {{ color: var(--muted); font-size: 12px; white-space: nowrap; }}
    .confidence-chip, .llm-chip {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 3px 7px;
      background: #fff;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .llm-chip.fallback {{ border-color: #f2c94c; background: #fff8df; color: #8a5a00; }}
    .source-list {{ display: flex; flex-wrap: wrap; gap: 4px; max-width: 100%; }}
    .source-pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      background: #fff;
      color: #344054;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .portfolio-mini {{ min-width: 150px; }}
    .portfolio-line {{ display: flex; justify-content: space-between; gap: 10px; font-size: 12px; margin-bottom: 4px; }}
    .portfolio-track {{ height: 7px; border-radius: 999px; background: #eef1f5; overflow: hidden; margin-bottom: 7px; }}
    .portfolio-bar {{ height: 100%; border-radius: inherit; background: linear-gradient(90deg, #147d4f, #6cbf8a); }}
    .portfolio-pos {{ color: var(--muted); font-size: 12px; }}
    .portfolio-delta {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      background: #eef6f1;
      color: #147d4f;
      font-size: 12px;
      font-weight: 700;
    }}
    .portfolio-delta.negative {{ background: #fff1f0; color: #b42318; }}
    .outcome-badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
    }}
    .outcome-badge.success {{ background: #eef6f1; color: #147d4f; }}
    .outcome-badge.failed {{ background: #fff1f0; color: #b42318; }}
    .outcome-badge.neutral {{ background: #f1f4f8; color: #344054; }}
    .outcome-badge.cancelled {{ background: #fef3c7; color: #92400e; text-decoration: line-through; }}
    .signal-stack {{ display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }}
    .signal-pill {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      background: #fff;
      color: #344054;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .signal-pill.good {{ border-color: #b7e4c7; background: #f0f9f3; color: #147d4f; }}
    .signal-pill.warn {{ border-color: #fed7aa; background: #fff7ed; color: #b54708; }}
    .signal-pill.muted {{ color: var(--muted); }}
    .signal-pill.cancelled {{ border-color: #fbbf24; background: #fffbeb; color: #92400e; }}
    .signal-pill.retry {{ border-color: #93c5fd; background: #eff6ff; color: #1e40af; }}
    .signal-pill.failed {{ border-color: #fca5a5; background: #fef2f2; color: #991b1b; }}
    .signal-pill.reversal {{ border-color: #c4b5fd; background: #f5f3ff; color: #5b21b6; }}
    .summary-signals {{ margin-top: 8px; }}
    dialog {{
      width: min(820px, calc(100% - 28px));
      max-height: min(860px, 92vh);
      overflow: auto;
      border: 0;
      border-radius: 8px;
      box-shadow: 0 24px 64px rgba(16, 24, 40, .22);
      padding: 0;
    }}
    dialog::backdrop {{ background: rgba(23, 32, 42, .42); }}
    .modal-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 3;
      background: var(--panel);
    }}
    .modal-head h2 {{ font-size: 18px; margin: 0; }}
    .modal-body {{ padding: 16px 18px 18px; }}
    .modal-body h3 {{ font-size: 13px; margin: 14px 0 6px; color: #344054; }}
    .modal-body pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f7f8fa;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font-family: Consolas, Monaco, monospace;
      font-size: 13px;
      margin: 0;
    }}
    .close-button {{ border: 0; background: transparent; font-size: 22px; cursor: pointer; line-height: 1; }}
    .alert-panel {{
      border: 1px solid #fed7aa;
      background: #fff7ed;
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 12px;
      color: #9a3412;
      font-weight: 700;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .detail-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }}
    .detail-card.decision-card {{ border-color: #b7e4c7; background: #f0f9f3; }}
    .detail-card.portfolio-card {{ border-color: #cfe2ff; background: #f5f9ff; }}
    .detail-card.execution-card {{ border-color: #d9dee7; background: #fbfcfe; }}
    .detail-card span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 3px; }}
    .detail-card strong {{ font-size: 14px; }}
    .empty-state {{
      display: none;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--muted);
      padding: 18px;
      text-align: center;
      margin-top: 12px;
    }}
    .empty-state strong {{ display: block; color: var(--text); margin-bottom: 4px; }}
    .empty-state.visible {{ display: block; }}
    @media (max-width: 920px) {{
      header {{ display: block; }}
      .source {{ max-width: none; text-align: left; margin-top: 8px; }}
      .table-wrap {{ overflow: visible; max-height: none; border: 0; box-shadow: none; background: transparent; }}
      table, tbody, tr, td {{ display: block; width: 100%; }}
      colgroup, thead {{ display: none; }}
      table {{ min-width: 0; border-collapse: separate; }}
      tbody {{ display: grid; gap: 12px; }}
      tr {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow-soft);
        padding: 10px 12px;
      }}
      td {{
        border-bottom: 1px solid var(--soft-line);
        padding: 9px 0;
        display: grid;
        grid-template-columns: 116px 1fr;
        gap: 10px;
        align-items: start;
      }}
      td:last-child {{ border-bottom: 0; }}
      td::before {{
        content: attr(data-label);
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .03em;
      }}
      .cell-clamp {{ -webkit-line-clamp: 4; }}
      .portfolio-mini {{ min-width: 0; }}
      .toolbar {{ display: grid; }}
    }}
    @media (max-width: 760px) {{
      main.dashboard-shell {{ width: min(100% - 20px, 1180px); margin-top: 14px; }}
      .summary, .grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      h1 {{ font-size: 22px; }}
    }}
  </style>
</head>
<body>
  <main class="dashboard-shell">
    <header>
      <div>
        <h1>Trading Agent Journal</h1>
        <div class="meta">Run: {html.escape(output_path.parent.name)} | Tickers: {html.escape(stats["tickers"])} | Generated: {html.escape(_format_datetime(datetime.now().isoformat()))}</div>
      </div>
      <div class="meta source">Source: {html.escape(str(journal_path))}</div>
    </header>
    <section class="summary" aria-label="Run summary">
      <div class="summary-item stat-card info"><span>Run summary</span><strong class="stat-value">{html.escape(stats["run_summary_short"])}</strong><small class="stat-sub">{html.escape(stats["run_summary_detail"])}</small></div>
      <div class="summary-item stat-card {stats["recovery_class"]}"><span>Recovery signal</span><strong class="stat-value">{stats["guardrails"]} guardrails, {stats["fallbacks"]} fallbacks</strong><small class="stat-sub">{stats["retries"]} retries recorded</small></div>
      <div class="summary-item stat-card {stats["health_class"]}"><span>Execution health</span><strong class="stat-value">{stats["failures"]} failures</strong><small class="stat-sub">Across {stats["cycle_count"]} cycles</small></div>
      <div class="summary-item stat-card info"><span>Portfolio outcome</span><strong class="stat-value">{html.escape(stats["portfolio_outcome"])}</strong><small class="stat-sub">Latest snapshot</small></div>
      <div class="summary-item stat-card {stats["portfolio_class"]}"><span>Portfolio path</span><strong class="stat-value">{html.escape(stats["portfolio_path_short"])}</strong><small class="stat-sub">{html.escape(stats["portfolio_path_detail"])}</small></div>
    </section>
    <section class="grid" aria-label="Run metrics">
      <div class="metric"><span>Total cycles</span><strong>{stats["cycle_count"]}</strong></div>
      <div class="metric"><span>Progress events</span><strong>{stats["stage_count"]}</strong></div>
      <div class="metric"><span>Actions</span><strong class="split"><b>BUY {stats["actions"].get("BUY", 0)}</b><b>SELL {stats["actions"].get("SELL", 0)}</b><b>HOLD {stats["actions"].get("HOLD", 0)}</b><b>WAIT {stats["actions"].get("WAIT", 0)}</b></strong></div>
      <div class="metric"><span>Guardrails</span><strong>{stats["guardrails"]}</strong></div>
      <div class="metric"><span>Failures / retries</span><strong>{stats["failures"]} / {stats["retries"]}</strong></div>
      <div class="metric"><span>Human messages</span><strong>{stats["human_event_count"]}</strong></div>
    </section>
    {human_events_html}
    <section class="toolbar" aria-label="Journal controls">
      <div class="filter-group">
        <button class="filter-button active" type="button" data-filter="all" onclick="filterRows('all', this)">All</button>
        <button class="filter-button" type="button" data-filter="trades" onclick="filterRows('trades', this)">Trades</button>
        <button class="filter-button" type="button" data-filter="failed" onclick="filterRows('failed', this)">Failed</button>
        <button class="filter-button" type="button" data-filter="guardrails" onclick="filterRows('guardrails', this)">Guardrails</button>
      </div>
    </section>
    <div class="table-wrap">
      <table>
        <colgroup>
          <col style="width:11%">
          <col style="width:11%">
          <col style="width:42%">
          <col style="width:13%">
          <col style="width:13%">
          <col style="width:10%">
        </colgroup>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Action</th>
            <th>Summary</th>
            <th>Cash movement</th>
            <th>Diversification</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>{table_rows or _empty_row()}</tbody>
      </table>
    </div>
    <div id="emptyState" class="empty-state" aria-live="polite">
      <strong>No rows to show</strong>
      <span>Select another filter to inspect the journal.</span>
    </div>
    <dialog id="detailDialog">
      <div class="modal-head">
        <h2 id="detailTitle">Cycle details</h2>
        <button class="close-button" type="button" onclick="document.getElementById('detailDialog').close()" aria-label="Close">&times;</button>
      </div>
      <div class="modal-body" id="detailBody"></div>
    </dialog>
  </main>
  <script>
    const journalDetails = {details_json};
    function escapeHtml(value) {{
      return String(value || "-")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}
    function block(title, text) {{
      return `<h3>${{title}}</h3><pre>${{escapeHtml(text)}}</pre>`;
    }}
    function card(label, value, cssClass = "") {{
      return `<div class="detail-card ${{cssClass}}"><span>${{label}}</span><strong>${{escapeHtml(value)}}</strong></div>`;
    }}
    function showDetails(index) {{
      const item = journalDetails[index];
      document.getElementById("detailTitle").textContent = `${{item.timestamp}} | ${{item.ticker}} ${{item.action}}`;
      document.getElementById("detailBody").innerHTML =
        item.alert_html +
        `<div class="detail-grid">
          ${{card("Decision", item.action + " " + item.ticker, "decision-card")}}
          ${{card("Draft", item.draft_short)}}
          ${{card("Quantity", item.quantity_short)}}
          ${{card("Confidence", item.confidence)}}
          ${{card("Portfolio", item.portfolio_short, "portfolio-card")}}
          ${{card("Execution", item.outcome_short, "execution-card")}}
          ${{card("Order price", item.order_price_short)}}
          ${{card("Fill price", item.fill_price_short)}}
          ${{card("LLM", item.llm_short)}}
          ${{card("Data readiness", item.data_short)}}
        </div>` +
        block("Summary", item.cycle_summary) +
        block("Rationale", item.rationale) +
        block("Rationale details", item.rationale_markdown) +
        block("Decision audit", item.decision_audit_markdown) +
        block("Technical signals", item.technical_markdown) +
        block("Guardrails", item.guardrails_markdown) +
        block("Failures", item.failures_markdown) +
        block("LLM provider", item.llm_markdown) +
        block("Portfolio after", item.portfolio_markdown) +
        block("Sector diversification", item.diversification_text) +
        block("Outcome", item.outcome) +
        block("Structured details", item.structured_details);
      document.getElementById("detailDialog").showModal();
    }}
    function handleRowKey(event, index) {{
      if (event.key === "Enter" || event.key === " ") {{
        event.preventDefault();
        showDetails(index);
      }}
    }}
    function filterRows(filter, button) {{
      document.querySelectorAll(".filter-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      let visibleCount = 0;
      document.querySelectorAll(".journal-row").forEach((row) => {{
        const matchesFilter =
          filter === "all" ||
          row.dataset.kind === filter ||
          (filter === "guardrails" && row.dataset.guardrails === "1");
        row.style.display = matchesFilter ? "" : "none";
        if (matchesFilter) visibleCount += 1;
      }});
      updateEmptyState(filter, visibleCount);
    }}
    function updateEmptyState(filter, visibleCount) {{
      const emptyState = document.getElementById("emptyState");
      if (!emptyState) return;
      const messages = {{
        all: ["No journal entries", "Run the agent to create the first cycle."],
        trades: ["No BUY or SELL cycles", "The agent only held, waited, or no trade was attempted."],
        failed: ["No failed cycles", "All visible cycles completed without a failed execution."],
        guardrails: ["No guardrails triggered", "No data quality or safety guardrails fired in this run."]
      }};
      const message = messages[filter] || messages.all;
      emptyState.querySelector("strong").textContent = message[0];
      emptyState.querySelector("span").textContent = message[1];
      emptyState.classList.toggle("visible", visibleCount === 0);
    }}
  </script>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def _read_rows(journal_path: Path) -> list[dict[str, Any]]:
    if not journal_path.exists():
        return []
    return [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cycle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("entry_type", "cycle") == "cycle"]


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cycle_rows = _cycle_rows(rows)
    stage_rows = [row for row in rows if row.get("entry_type") == "stage"]
    human_event_rows = [row for row in rows if row.get("entry_type") == "human_event"]
    timeline_rows = cycle_rows or rows
    actions = Counter(row.get("action", "UNKNOWN") for row in cycle_rows)
    tickers = sorted({row.get("ticker", "-") for row in rows})
    timestamps = [row.get("timestamp") for row in timeline_rows if row.get("timestamp")]
    guardrails = sum(len(row.get("guardrails_triggered") or []) for row in cycle_rows)
    failures = sum(len(row.get("failures") or []) for row in rows)
    failures += sum(1 for row in stage_rows if row.get("status") == "failed")
    retries = sum(_retry_count(row) for row in cycle_rows)
    fallbacks = sum(1 for row in cycle_rows if row.get("llm_fallback_used"))
    portfolio_outcome = _portfolio_outcome(cycle_rows)
    portfolio_path = _portfolio_path(cycle_rows)
    if timestamps:
        run_summary = f"{len(cycle_rows)} cycles from {_format_datetime(min(timestamps))} to {_format_datetime(max(timestamps))}"
        run_summary_short = f"{len(cycle_rows)} cycles"
        run_summary_detail = f"{_format_datetime(min(timestamps))} to {_format_datetime(max(timestamps))}"
    else:
        run_summary = "No cycles recorded"
        run_summary_short = "No cycles"
        run_summary_detail = "-"
    if not cycle_rows and stage_rows:
        run_summary = f"No completed cycles; {len(stage_rows)} progress events recorded"
        run_summary_short = "0 cycles"
        run_summary_detail = f"{len(stage_rows)} progress events"
    initial = _initial_portfolio(cycle_rows)
    current = _current_portfolio(cycle_rows)
    initial_cash = _safe_float((initial or {}).get("cash"))
    current_cash = _safe_float((current or {}).get("cash"))
    initial_value = _safe_float((initial or {}).get("portfolio_value"))
    current_value = _safe_float((current or {}).get("portfolio_value"))
    cash_delta = None if initial_cash is None or current_cash is None else current_cash - initial_cash
    value_delta = None if initial_value is None or current_value is None else current_value - initial_value
    return {
        "actions": actions,
        "cycle_count": len(cycle_rows),
        "stage_count": len(stage_rows),
        "human_event_count": len(human_event_rows),
        "tickers": ", ".join(tickers) if tickers else "-",
        "run_summary": run_summary,
        "run_summary_short": run_summary_short,
        "run_summary_detail": run_summary_detail,
        "guardrails": guardrails,
        "failures": failures,
        "retries": retries,
        "fallbacks": fallbacks,
        "portfolio_outcome": portfolio_outcome,
        "portfolio_path": portfolio_path,
        "portfolio_path_short": _portfolio_path_short(initial_cash, current_cash, initial_value, current_value),
        "portfolio_path_detail": _portfolio_path_detail(cash_delta, value_delta),
        "portfolio_class": "warn" if value_delta is not None and value_delta < 0 else "good",
        "recovery_class": "warn" if guardrails or fallbacks or retries else "good",
        "health_class": "warn" if failures else "good",
    }


def _render_human_events(rows: list[dict[str, Any]]) -> str:
    events = [row for row in rows if row.get("entry_type") == "human_event"]
    if not events:
        return ""
    groups = _group_human_events(events)
    items = "\n".join(_render_human_event_group(group, is_latest=index == len(groups[-8:]) - 1) for index, group in enumerate(groups[-8:]))
    return f"""
    <section class="human-panel" aria-label="Human message timeline">
      <h2>Human message timeline</h2>
      <div class="human-list">{items}</div>
    </section>"""


def _group_human_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in events:
        key = str(row.get("note_id") or row.get("instruction_id") or row.get("timestamp") or len(order))
        if key not in groups:
            groups[key] = {"events": [], "details": {}, "tickers": [], "instruction_ids": []}
            order.append(key)
        group = groups[key]
        group["events"].append(row)
        ticker = str(row.get("ticker") or "")
        if ticker and ticker != "HUMAN" and ticker not in group["tickers"]:
            group["tickers"].append(ticker)
        instruction_id = str(row.get("instruction_id") or "")
        if instruction_id and instruction_id not in group["instruction_ids"]:
            group["instruction_ids"].append(instruction_id)
        details = row.get("details")
        if isinstance(details, dict):
            group["details"].update({key: value for key, value in details.items() if value is not None})
    return [groups[key] for key in order]


def _render_human_event_group(group: dict[str, Any], *, is_latest: bool) -> str:
    events = group.get("events") or []
    first = events[0] if events else {}
    latest = events[-1] if events else {}
    timestamp = _format_datetime(str(latest.get("timestamp", "-")))
    note = str(first.get("note", "-"))
    details = group.get("details") or {}
    statuses = [str(row.get("status", "-")) for row in events]
    status_summary = _human_status_summary(statuses)
    summary_row = {
        **latest,
        "status": statuses[-1] if statuses else latest.get("status", "-"),
        "ticker": ", ".join(group.get("tickers") or []) or latest.get("ticker", "HUMAN"),
    }
    instruction_id = ", ".join(group.get("instruction_ids") or []) or "-"
    meta = _human_event_meta(summary_row, details)
    detail_lines = _human_event_detail_lines(details)
    open_attr = " open" if is_latest else ""
    return f"""
      <details class="human-card"{open_attr}>
        <summary>
          <span class="human-summary-main">{_e(note)}</span>
          <span class="human-summary-sub">{_e(status_summary)} | {_e(timestamp)} | instruction {_e(instruction_id)}</span>
        </summary>
        <div class="human-card-body">
          <div class="human-meta">{meta}</div>
          <div class="human-steps">{_human_status_steps(statuses)}</div>
          {detail_lines}
        </div>
      </details>"""


def _human_status_summary(statuses: list[str]) -> str:
    cleaned = [status for status in statuses if status and status != "-"]
    if not cleaned:
        return "No status"
    if len(cleaned) == 1:
        return f"Latest {cleaned[-1]}"
    return f"{len(cleaned)} updates; latest {cleaned[-1]}"


def _human_status_steps(statuses: list[str]) -> str:
    cleaned = [s for s in statuses if s]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return f'<span class="human-step-chip">{_e(cleaned[0])}</span>'
    return f'<span class="human-step-chip">{_e(cleaned[0])}</span>' f'<span class="human-step-chip">···</span>' f'<span class="human-step-chip">{_e(cleaned[-1])}</span>'


def _human_event_meta(row: dict[str, Any], details: Any) -> str:
    detail = details if isinstance(details, dict) else {}
    chips = [f'<span class="human-chip status">{_e(row.get("status", "-"))}</span>']
    if detail.get("intent_type"):
        chips.append(f'<span class="human-chip intent">{_e(detail.get("intent_type"))}</span>')
    ticker = str(row.get("ticker") or "")
    if ticker and ticker != "HUMAN":
        chips.append(f'<span class="human-chip target">target {_e(ticker)}</span>')
    if detail.get("confidence") is not None:
        chips.append(f'<span class="human-chip">confidence {_e(_format_confidence(detail.get("confidence")))}</span>')
    if detail.get("topic"):
        chips.append(f'<span class="human-chip">topic {_e(detail.get("topic"))}</span>')
    if detail.get("reason"):
        chips.append(f'<span class="human-chip">{_e(detail.get("reason"))}</span>')
    return "".join(chips)


def _human_event_detail_lines(details: Any) -> str:
    detail = details if isinstance(details, dict) else {}
    lines: list[str] = []
    rationale = detail.get("rationale")
    if rationale:
        lines.append(_human_detail("Rationale", str(rationale)))
    risk_profile = detail.get("human_risk_profile")
    if isinstance(risk_profile, dict):
        lines.append(
            _human_detail(
                "Risk profile",
                (
                    f"{risk_profile.get('risk_preference', 'neutral')}; "
                    f"buy {_format_confidence(risk_profile.get('buy_aggressiveness'))}; "
                    f"sell {_format_confidence(risk_profile.get('sell_aggressiveness'))}; "
                    f"{risk_profile.get('rationale', '')}"
                ).strip(),
            )
        )
    risk_limits = detail.get("risk_limits")
    if isinstance(risk_limits, dict):
        lines.append(
            _human_detail(
                "Risk limits",
                f"max buy {risk_limits.get('max_buy_quantity', '-')}; max sell {risk_limits.get('max_sell_quantity', '-')}",
            )
        )
    planned = detail.get("planned_targets")
    if isinstance(planned, list) and planned:
        if _has_human_target_rationale(planned):
            lines.append(_human_target_rationale_detail("Target rationale", planned))
        else:
            lines.append(_human_detail("Planned positions", _format_human_target_list(planned)))
    excluded = detail.get("excluded_tickers")
    if isinstance(excluded, list) and excluded:
        if _has_human_target_rationale(excluded):
            lines.append(_human_target_rationale_detail("Excluded rationale", excluded))
        else:
            lines.append(_human_detail("Excluded positions", _format_human_target_list(excluded)))
    buyable = detail.get("buyable_universe")
    if buyable:
        lines.append(_human_detail("Buy universe", ", ".join(str(item) for item in buyable)))
    sellable = detail.get("sellable_open_positions")
    if sellable:
        lines.append(_human_detail("Sellable positions", ", ".join(str(item) for item in sellable)))
    return "".join(lines)


def _human_detail(label: str, value: str) -> str:
    return f'<div class="human-detail"><span>{_e(label)}:</span>{_e(value)}</div>'


def _human_target_rationale_detail(label: str, items: list[Any]) -> str:
    rows: list[str] = []
    rationales = [str(item.get("rationale") or "").strip() for item in items if isinstance(item, dict)]
    all_same = len(set(rationales)) == 1 and rationales[0]

    for item in items:
        if not isinstance(item, dict):
            rows.append(f'<div class="human-target-row"><b>{_e(str(item))}</b><div>-</div></div>')
            continue
        ticker = str(item.get("ticker") or "-").upper()
        seq_i = item.get("sequence_index")
        seq_t = item.get("sequence_total")
        seq = f" {seq_i}/{seq_t}" if seq_i and seq_t else ""
        rationale = str(item.get("rationale") or "").strip() or "-"
        display = "-" if all_same else rationale
        rows.append(f'<div class="human-target-row"><b>{_e(ticker + seq)}</b><div>{_e(display)}</div></div>')

    common = f'<div class="human-detail"><span>Common rationale:</span>{_e(rationales[0])}</div>' if all_same else ""
    return f'<div class="human-detail"><span>{_e(label)}:</span>' f'<div class="human-target-list">{"".join(rows)}</div></div>' + common


def _has_human_target_rationale(items: list[Any]) -> bool:
    return any(isinstance(item, dict) and str(item.get("rationale") or "").strip() for item in items)


def _format_human_target_list(items: list[Any]) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            ticker = str(item.get("ticker") or item.get("symbol") or "-").upper()
            sequence_index = item.get("sequence_index")
            sequence_total = item.get("sequence_total")
            sequence = f" {sequence_index}/{sequence_total}" if sequence_index and sequence_total else ""
            rationale = str(item.get("rationale") or "").strip()
            parts.append(f"{ticker}{sequence}: {rationale}" if rationale else f"{ticker}{sequence}")
        else:
            parts.append(str(item))
    return "; ".join(parts)


def _format_confidence(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _render_row(index: int, row: dict[str, Any]) -> str:
    action = row.get("action", "-")
    ticker = row.get("ticker", "-")
    signals = _signals_html(row)
    portfolio = _portfolio_delta_html(row)
    after_positions = (_portfolio_after(row) or {}).get("positions")
    breakdown = _sector_breakdown(after_positions)
    diversification_html = _sector_bar_html(breakdown)
    outcome = _outcome_text(row)
    outcome_badge = _outcome_badge_html(row)
    timestamp = str(row.get("timestamp", "-"))
    filter_kind = _row_filter_kind(row)
    has_guardrails = "1" if row.get("guardrails_triggered") else "0"
    return f"""
        <tr class="journal-row" tabindex="0" onclick="showDetails({index})" onkeydown="handleRowKey(event, {index})" data-ticker="{_e(ticker)}" data-kind="{filter_kind}" data-guardrails="{has_guardrails}" title="Open details">
          <td data-label="Timestamp" data-raw-timestamp="{_e(timestamp)}">{_timestamp_html(timestamp)}</td>
          <td data-label="Action"><span class="badge {_e(action)}">{_e(action)} {_e(ticker)}</span></td>
          <td data-label="Summary"><div class="cell-clamp summary-cell">{_e(row.get("cycle_summary", "-"))}</div><div class="summary-signals">{signals}</div></td>
          <td data-label="Cash movement">{portfolio}</td>
          <td data-label="Diversification">{diversification_html}</td>
          <td data-label="Outcome">{outcome_badge}</td>
        </tr>"""


def _details_payload(row: dict[str, Any]) -> dict[str, str]:
    return {
        "timestamp": str(row.get("timestamp", "-")),
        "ticker": str(row.get("ticker", "-")),
        "action": str(row.get("action", "-")),
        "draft_short": _draft_short(row),
        "confidence": f"{float(row.get('confidence', 0)):.2f}",
        "quantity_short": _quantity_short(row),
        "portfolio_short": _portfolio_short(row),
        "order_price_short": _order_price_short(row),
        "fill_price_short": _fill_price_short(row),
        "outcome_short": _outcome_short(row),
        "llm_short": _llm_short(row),
        "data_short": _data_readiness(row),
        "alert_html": _alert_html(row),
        "cycle_summary": str(row.get("cycle_summary", "-")),
        "rationale": str(row.get("rationale", "-")),
        "rationale_markdown": _rationale_markdown(row),
        "decision_audit_markdown": _decision_audit_markdown(row),
        "technical_markdown": _technical_markdown(row),
        "guardrails_markdown": _markdown_list(row.get("guardrails_triggered") or [], "No guardrails triggered."),
        "failures_markdown": _markdown_list(row.get("failures") or [], "No failures recorded."),
        "llm_markdown": _llm_markdown(row),
        "portfolio_markdown": _portfolio_markdown(row),
        "diversification_text": _diversification_text(_portfolio_after(row)),
        "outcome": _outcome_text(row),
        "structured_details": json.dumps(_format_dashboard_value(row), indent=2, ensure_ascii=False),
    }


def _quantity_short(row: dict[str, Any]) -> str:
    decision = row.get("decision") or {}
    execution = row.get("execution_result") or {}
    requested = execution.get("requested_quantity", decision.get("quantity"))
    allowed = execution.get("allowed_quantity")
    if allowed is None:
        return f"requested {requested}"
    return f"requested {requested}, max {allowed}"


def _order_price_short(row: dict[str, Any]) -> str:
    execution = row.get("execution_result") or {}
    return _format_money(_safe_float(execution.get("current_price_at_order")))


def _fill_price_short(row: dict[str, Any]) -> str:
    execution = row.get("execution_result") or {}
    return _format_money(_safe_float(execution.get("filled_avg_price")))


def _draft_short(row: dict[str, Any]) -> str:
    draft = row.get("draft_decision") or {}
    if not draft:
        return "same as final"
    return f"{draft.get('action', '-')} qty {draft.get('quantity', '-')}"


def _decision_audit_markdown(row: dict[str, Any]) -> str:
    decision = row.get("decision") or {}
    draft = row.get("draft_decision") or {}
    execution = row.get("execution_result") or {}
    lines = [
        f"- draft_action: {draft.get('action', 'n/a')}",
        f"- draft_quantity: {draft.get('quantity', 'n/a')}",
        f"- draft_confidence: {_format_dashboard_value(draft.get('confidence', 'n/a'))}",
        f"- llm_action: {decision.get('action', row.get('action', '-'))}",
        f"- llm_quantity: {decision.get('quantity', '-')}",
        f"- llm_confidence: {_format_dashboard_value(decision.get('confidence', row.get('confidence', '-')))}",
        f"- executor_attempted_action: {execution.get('attempted_action', '-')}",
        f"- requested_quantity: {execution.get('requested_quantity', decision.get('quantity', '-'))}",
        f"- allowed_quantity: {execution.get('allowed_quantity', 'n/a')}",
        f"- current_price_at_order: {_format_money(_safe_float(execution.get('current_price_at_order')))}",
        f"- filled_avg_price: {_format_money(_safe_float(execution.get('filled_avg_price')))}",
        f"- execution_status: {execution.get('status', row.get('outcome', '-'))}",
    ]
    if execution.get("message"):
        lines.append(f"- execution_message: {execution.get('message')}")
    if execution.get("risk_explanation"):
        lines.append(f"- risk_explanation: {execution.get('risk_explanation')}")
    return "\n".join(lines)


def _rationale_markdown(row: dict[str, Any]) -> str:
    decision = row.get("decision") or {}
    details = decision.get("rationale_details") or {}
    if not isinstance(details, dict) or not details:
        return "No structured rationale details recorded."
    lines: list[str] = []
    if details.get("summary"):
        lines.append(f"- summary: {details.get('summary')}")
    for item in details.get("evidence") or []:
        lines.append(f"- evidence: {item}")
    for item in details.get("risks") or []:
        lines.append(f"- risk: {item}")
    if details.get("data_quality"):
        lines.append(f"- data_quality: {details.get('data_quality')}")
    return "\n".join(lines) if lines else "No structured rationale details recorded."


def _technical_markdown(row: dict[str, Any]) -> str:
    snapshot = row.get("market_snapshot") or {}
    indicators = snapshot.get("technical_indicators") or {}
    lines = [
        f"- confidence: {indicators.get('confidence', 'none')}",
        f"- summary: {_technical_summary(indicators)}",
    ]
    for key in ["sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_histogram"]:
        if indicators.get(key) is not None:
            lines.append(f"- {key}: {_format_dashboard_value(indicators.get(key))}")
    for note in indicators.get("notes") or []:
        lines.append(f"- note: {note}")
    return "\n".join(lines)


def _technical_summary(indicators: dict[str, Any]) -> str:
    parts: list[str] = []
    rsi = indicators.get("rsi_14")
    if isinstance(rsi, (int, float)):
        if rsi >= 70:
            parts.append("RSI overbought")
        elif rsi <= 30:
            parts.append("RSI oversold")
        else:
            parts.append("RSI neutral")
    macd = indicators.get("macd")
    signal = indicators.get("macd_signal")
    if isinstance(macd, (int, float)) and isinstance(signal, (int, float)):
        parts.append("MACD bullish" if macd > signal else "MACD bearish")
    sma_20 = indicators.get("sma_20")
    sma_50 = indicators.get("sma_50")
    if isinstance(sma_20, (int, float)) and isinstance(sma_50, (int, float)):
        parts.append("SMA20 above SMA50" if sma_20 > sma_50 else "SMA20 below SMA50")
    return "; ".join(parts) if parts else "No technical signal"


def _format_dashboard_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, dict):
        return {key: _format_dashboard_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_dashboard_value(item) for item in value]
    return value


def _data_quality_pill(row: dict[str, Any]) -> str:
    snapshot = row.get("market_snapshot") or {}
    values = [
        snapshot.get("price_confidence", "none"),
        snapshot.get("news_confidence", "none"),
        (snapshot.get("technical_indicators") or {}).get("confidence", "none"),
    ]
    if "low" in values or "none" in values:
        return '<span class="signal-pill warn">Data quality: partial</span>'
    return '<span class="signal-pill good">Data quality: good</span>'


def _technical_pill(row: dict[str, Any]) -> str:
    indicators = (row.get("market_snapshot") or {}).get("technical_indicators") or {}
    confidence = indicators.get("confidence", "none")
    if confidence in {"high", "medium"}:
        return '<span class="signal-pill good">Technicals: ready</span>'
    return '<span class="signal-pill muted">Technicals: limited</span>'


def _llm_pill(row: dict[str, Any]) -> str:
    provider = row.get("llm_provider") or "none"
    if row.get("llm_fallback_used"):
        return f'<span class="signal-pill warn">AI: fallback {_e(provider)}</span>'
    return f'<span class="signal-pill muted">AI: {_e(provider)}</span>'


def _llm_markdown(row: dict[str, Any]) -> str:
    lines = [f"- provider: {row.get('llm_provider') or 'none'}"]
    lines.append(f"- fallback_used: {bool(row.get('llm_fallback_used'))}")
    if row.get("llm_fallback_provider"):
        lines.append(f"- fallback_provider: {row.get('llm_fallback_provider')}")
    if row.get("llm_fallback_reason"):
        lines.append(f"- fallback_reason: {row.get('llm_fallback_reason')}")
    return "\n".join(lines)


def _signals_html(row: dict[str, Any]) -> str:
    pieces = [
        _data_quality_pill(row),
        _technical_pill(row),
        _llm_pill(row),
    ]
    if row.get("guardrails_triggered"):
        pieces.append('<span class="signal-pill warn">Guardrail</span>')
    if row.get("cancelled_by"):
        pieces.append(f'<span class="signal-pill cancelled" title="Cancelled by instruction {row["cancelled_by"]}">CANCELLATO</span>')
    retry_count = int(row.get("retry_count") or 0)
    if retry_count > 0:
        pieces.append(f'<span class="signal-pill retry" title="Retry attempt {retry_count}">RETRY {retry_count}</span>')
    failure_type = row.get("failure_type")
    if failure_type:
        pieces.append(f'<span class="signal-pill failed" title="Classified failure type">{failure_type}</span>')
    reversal_of = row.get("reversal_of")
    if reversal_of:
        pieces.append(f'<span class="signal-pill reversal" title="Reversal of instruction {reversal_of}">REVERSAL</span>')
    return '<div class="signal-stack">' + "".join(pieces) + "</div>"


def _outcome_badge_html(row: dict[str, Any]) -> str:
    if row.get("cancelled_by"):
        return '<span class="outcome-badge cancelled">Cancelled</span>'
    execution = row.get("execution_result") or {}
    status = str(execution.get("status", row.get("outcome", ""))).lower()
    if status in {"filled", "skipped"}:
        return '<span class="outcome-badge success">Success</span>'
    if status == "waiting":
        return '<span class="outcome-badge neutral">Waiting</span>'
    if status == "submitted":
        return '<span class="outcome-badge neutral">Submitted</span>'
    if status in {"blocked", "rejected", "failed"}:
        return '<span class="outcome-badge failed">Failed</span>'
    return '<span class="outcome-badge neutral">Pending</span>'


def _outcome_short(row: dict[str, Any]) -> str:
    execution = row.get("execution_result") or {}
    status = execution.get("status", row.get("outcome", "-"))
    return str(status)


def _llm_short(row: dict[str, Any]) -> str:
    provider = row.get("llm_provider") or "none"
    if row.get("llm_fallback_used"):
        return f"{provider} fallback"
    return str(provider)


def _portfolio_short(row: dict[str, Any]) -> str:
    before = _portfolio_before(row)
    after = _portfolio_after(row)
    cash = _safe_float((after or {}).get("cash"))
    cash_before = _safe_float((before or {}).get("cash"))
    delta = None if cash is None or cash_before is None else cash - cash_before
    value = _safe_float((after or {}).get("portfolio_value"))
    value_before = _safe_float((before or {}).get("portfolio_value"))
    value_delta = None if value is None or value_before is None else value - value_before
    positions = _positions_count((after or {}).get("positions"))
    if value is not None:
        return f"value {_format_money(value)} ({_format_signed_money(value_delta)}), cash {_format_money(cash)}, {positions} positions"
    return f"cash {_format_money(cash)} ({_format_signed_money(delta)}), {positions} positions"


def _alert_html(row: dict[str, Any]) -> str:
    guardrails = row.get("guardrails_triggered") or []
    failures = row.get("failures") or []
    if not guardrails and not failures:
        return ""
    parts = []
    if guardrails:
        parts.append(f"{len(guardrails)} guardrail(s)")
    if failures:
        parts.append(f"{len(failures)} failure(s)")
    return f'<div class="alert-panel">Attention: {_e(", ".join(parts))}</div>'


def _row_filter_kind(row: dict[str, Any]) -> str:
    action = row.get("action")
    execution = row.get("execution_result") or {}
    status = str(execution.get("status", row.get("outcome", ""))).lower()
    if status in {"blocked", "rejected", "failed"}:
        return "failed"
    if action in {"BUY", "SELL"}:
        return "trades"
    return "other"


def _portfolio_outcome(rows: list[dict[str, Any]]) -> str:
    for row in reversed(rows):
        portfolio = _portfolio_after(row)
        if portfolio:
            cash = _safe_float(portfolio.get("cash"))
            value = _safe_float(portfolio.get("portfolio_value"))
            positions = _positions_count(portfolio.get("positions"))
            if value is not None:
                return f"value {_format_money(value)}, cash {_format_money(cash)}, {positions} open positions"
            cash_text = f"cash {_format_money(cash)}" if cash is not None else "cash unknown"
            return f"{cash_text}, {positions} open positions"
    return "No portfolio snapshot"


def _portfolio_path(rows: list[dict[str, Any]]) -> str:
    initial = _initial_portfolio(rows)
    current = _current_portfolio(rows)
    if not initial and not current:
        return "No portfolio path"
    initial_cash = _safe_float((initial or {}).get("cash"))
    current_cash = _safe_float((current or {}).get("cash"))
    initial_value = _safe_float((initial or {}).get("portfolio_value"))
    current_value = _safe_float((current or {}).get("portfolio_value"))
    if initial_value is not None or current_value is not None:
        delta = None if initial_value is None or current_value is None else current_value - initial_value
        return f"Initial value {_format_money(initial_value)} to Current value {_format_money(current_value)} ({_format_signed_money(delta)})"
    delta = None if initial_cash is None or current_cash is None else current_cash - initial_cash
    return f"Initial cash {_format_money(initial_cash)} to Current cash {_format_money(current_cash)} ({_format_signed_money(delta)})"


def _portfolio_path_short(
    initial_cash: float | None,
    current_cash: float | None,
    initial_value: float | None,
    current_value: float | None,
) -> str:
    if initial_value is not None or current_value is not None:
        return f"{_format_money(initial_value)} to {_format_money(current_value)}"
    return f"{_format_money(initial_cash)} to {_format_money(current_cash)}"


def _portfolio_path_detail(cash_delta: float | None, value_delta: float | None) -> str:
    if value_delta is not None:
        return f"Portfolio value movement {_format_signed_money(value_delta)}, Cash movement {_format_signed_money(cash_delta)}"
    return f"Cash movement {_format_signed_money(cash_delta)}"


def _portfolio_delta_html(row: dict[str, Any]) -> str:
    before = _portfolio_before(row)
    after = _portfolio_after(row)
    if not before and not after:
        return '<span class="muted">No snapshot</span>'
    cash = _safe_float((after or {}).get("cash"))
    cash_before = _safe_float((before or {}).get("cash"))
    delta = None if cash is None or cash_before is None else cash - cash_before
    delta_class = " negative" if delta is not None and delta < 0 else ""
    delta_label = _cash_movement_label(row, delta)
    positions = _positions_count((after or {}).get("positions"))
    value = _safe_float((after or {}).get("portfolio_value"))
    value_line = f"Portfolio {_e(_format_money(value))}<br>" if value is not None else ""
    return (
        '<div class="portfolio-mini">'
        f'<span class="portfolio-delta{delta_class} money">{_e(delta_label)}</span>'
        f'<div class="portfolio-pos">{value_line}Cash {_e(_format_money(cash))}, Pos {positions}</div>'
        "</div>"
    )


def _cash_movement_label(row: dict[str, Any], delta: float | None) -> str:
    if delta is None:
        return "n/a"
    action = row.get("action")
    if action == "BUY" and delta < 0:
        return f"Spent {_format_money(abs(delta))}"
    if action == "SELL" and delta > 0:
        return f"Received {_format_money(delta)}"
    if delta == 0:
        return "No cash change"
    return _format_signed_money(delta)


def _portfolio_markdown(row: dict[str, Any]) -> str:
    portfolio = _portfolio_after(row)
    before = _portfolio_before(row)
    if not portfolio:
        return "No portfolio snapshot recorded."
    cash = _safe_float(portfolio.get("cash"))
    cash_before = _safe_float((before or {}).get("cash"))
    delta = None if cash is None or cash_before is None else cash - cash_before
    value = _safe_float(portfolio.get("portfolio_value"))
    value_before = _safe_float((before or {}).get("portfolio_value"))
    value_delta = None if value is None or value_before is None else value - value_before
    positions = portfolio.get("positions")
    lines = [
        f"- initial_cash: {_format_money(cash_before) if cash_before is not None else 'unknown'}",
        f"- cash_after: {_format_money(cash) if cash is not None else 'unknown'}",
        f"- cash_delta: {_format_signed_money(delta) if delta is not None else 'unknown'}",
        f"- initial_portfolio_value: {_format_money(value_before) if value_before is not None else 'unknown'}",
        f"- portfolio_value_after: {_format_money(value) if value is not None else 'unknown'}",
        f"- portfolio_value_delta: {_format_signed_money(value_delta) if value_delta is not None else 'unknown'}",
    ]
    normalized = normalize_positions(positions)
    if not normalized:
        lines.append("- positions: none")
    else:
        lines.append(f"- positions: {len(normalized)}")
        for symbol, payload in normalized.items():
            qty = _safe_float(payload.get("qty"))
            market_value = _safe_float(payload.get("market_value"))
            qty_text = f"{qty:.2f}" if qty is not None else "unknown"
            value_text = _format_money(market_value) if market_value is not None else "unknown"
            lines.append(f"- {symbol}: qty {qty_text}, market_value {value_text}")
    return "\n".join(lines)


def _portfolio_after(row: dict[str, Any]) -> dict[str, Any] | None:
    execution = row.get("execution_result") or {}
    portfolio = execution.get("portfolio_after")
    return portfolio if isinstance(portfolio, dict) else None


def _portfolio_before(row: dict[str, Any]) -> dict[str, Any] | None:
    execution = row.get("execution_result") or {}
    portfolio = execution.get("portfolio_before")
    return portfolio if isinstance(portfolio, dict) else None


def _initial_portfolio(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        portfolio = _portfolio_before(row) or _portfolio_after(row)
        if portfolio:
            return portfolio
    return None


def _current_portfolio(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        portfolio = _portfolio_after(row) or _portfolio_before(row)
        if portfolio:
            return portfolio
    return None


def _positions_count(positions: Any) -> int:
    return len(normalize_positions(positions))


def _safe_float(value: Any) -> float | None:
    return float_or_none(value)


def _format_money(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"${value:,.2f}"


def _format_signed_money(value: float | None) -> str:
    if value is None:
        return "unknown"
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _data_readiness(row: dict[str, Any]) -> str:
    snapshot = row.get("market_snapshot") or {}
    price = snapshot.get("price_confidence", "none")
    news = snapshot.get("news_confidence", "none")
    technical = (snapshot.get("technical_indicators") or {}).get("confidence", "none")
    if price in {"high", "medium"} and news in {"high", "medium"} and technical in {"high", "medium"}:
        return "Market data ready, news available, technical signals ready"
    missing = []
    if price in {"low", "none"}:
        missing.append("market price")
    if news in {"low", "none"}:
        missing.append("news")
    if technical in {"low", "none"}:
        missing.append("technical signals")
    return "Partial data: " + ", ".join(missing)


def _outcome_text(row: dict[str, Any]) -> str:
    execution = row.get("execution_result") or {}
    if execution:
        status = execution.get("status", row.get("outcome", "-"))
        message = execution.get("message", "")
        return f"{status}: {message}".strip()
    return str(row.get("outcome", "-"))


def _retry_count(row: dict[str, Any]) -> int:
    snapshot = row.get("market_snapshot") or {}
    execution = row.get("execution_result") or {}
    return int(snapshot.get("retry_count") or 0) + int(execution.get("retry_count") or 0)


def _markdown_list(items: list[Any], empty: str) -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def _empty_row() -> str:
    return '<tr><td colspan="6" class="muted">No entries yet.</td></tr>'


def _timestamp_html(value: str) -> str:
    parsed = _parse_datetime(value)
    if not parsed:
        return _e(value)
    return f'<div class="time-main">{parsed.strftime("%d/%m %H:%M:%S")}</div>' f'<div class="time-sub">{parsed.strftime("%Y")} UTC</div>'


def _format_datetime(value: str) -> str:
    parsed = _parse_datetime(value)
    if not parsed:
        return value
    return parsed.strftime("%d/%m/%Y %H:%M:%S")


def _parse_datetime(value: str) -> datetime | None:
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone().replace(tzinfo=None)
    return parsed


def _sector_breakdown(positions: Any) -> dict[str, float]:
    try:
        from trading_agent.core.rebalance import TICKER_SECTOR
    except ImportError:
        return {}
    normalized = normalize_positions(positions)
    totals: dict[str, float] = {}
    for symbol, data in normalized.items():
        sector = TICKER_SECTOR.get(symbol.upper(), "other")
        mv = _safe_float(data.get("market_value")) or 0.0
        totals[sector] = totals.get(sector, 0) + mv
    total = sum(totals.values())
    if total <= 0:
        return {}
    return {s: round(v / total * 100, 1) for s, v in sorted(totals.items())}


def _sector_bar_html(breakdown: dict[str, float]) -> str:
    if not breakdown:
        return '<span class="muted">—</span>'
    COLORS = {
        "technology": "#1769aa",
        "automotive": "#147d4f",
        "manufacturing": "#b54708",
        "energy": "#b42318",
        "healthcare": "#5b21b6",
        "finance": "#0e7490",
        "consumer": "#92400e",
        "other": "#667085",
    }
    bars = "".join(f'<div style="height:100%;width:{pct}%;background:{COLORS.get(sector,"#667085")}" ' f'title="{sector} {pct}%"></div>' for sector, pct in breakdown.items())
    labels = " ".join(f'<span style="font-size:11px;color:{COLORS.get(s,"#667085")}">' f"{s[:4]} {p}%</span>" for s, p in breakdown.items())
    return (
        f'<div style="display:flex;height:6px;border-radius:999px;overflow:hidden;'
        f'background:#eef1f5;margin-bottom:3px">{bars}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:4px">{labels}</div>'
    )


def _diversification_text(portfolio: dict | None) -> str:
    if not portfolio:
        return "No portfolio snapshot."
    breakdown = _sector_breakdown(portfolio.get("positions"))
    if not breakdown:
        return "No open positions."
    return "\n".join(f"- {sector}: {pct}%" for sector, pct in breakdown.items())


def _e(value: Any) -> str:
    return html.escape(str(value))
