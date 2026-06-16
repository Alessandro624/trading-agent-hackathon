from __future__ import annotations

import argparse
from pathlib import Path

from trading_agent.core import HumanInputStore
from trading_agent.journal import latest_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Append live human notes for the trading agent.")
    parser.add_argument("--file", type=Path, default=None, help="Path to the human_input.md file used by the running agent.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory to target. Defaults to the latest run.")
    parser.add_argument("--base-dir", type=Path, default=Path("journal_runs"), help="Base directory used to find the latest run.")
    parser.add_argument(
        "--cursor",
        type=Path,
        default=None,
        help="Optional cursor path. The writer does not consume notes, but the store requires a cursor path.",
    )
    args = parser.parse_args()
    input_path, cursor_path = resolve_human_input_paths(args.file, args.run_dir, args.base_dir, args.cursor)
    store = HumanInputStore(input_path, cursor_path)
    if not input_path.exists():
        input_path.write_text("", encoding="utf-8")
    if not cursor_path.exists():
        cursor_path.write_text("0", encoding="utf-8")
    print(f"writing human notes to: {input_path}")
    print("Press Ctrl+C or submit an empty line to stop.")
    while True:
        try:
            note = input("human> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not note:
            break
        store.append_note(note)
        print("saved")


def resolve_human_input_paths(
    file_path: Path | None,
    run_dir: Path | None,
    base_dir: Path,
    cursor_path: Path | None,
) -> tuple[Path, Path]:
    if file_path is not None:
        return file_path, cursor_path or file_path.with_suffix(".cursor")
    if run_dir is not None:
        input_path = run_dir / "human_input.md"
        return input_path, cursor_path or run_dir / "human_input.cursor"
    context = latest_run(base_dir)
    return context.human_input_path, cursor_path or context.human_input_cursor_path


if __name__ == "__main__":
    main()
