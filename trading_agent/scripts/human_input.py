from __future__ import annotations

import argparse
from pathlib import Path

from trading_agent.core import HumanInputStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Append live human notes for the trading agent.")
    parser.add_argument("--file", type=Path, required=True, help="Path to the human_input.md file used by the running agent.")
    parser.add_argument(
        "--cursor",
        type=Path,
        default=None,
        help="Optional cursor path. The writer does not consume notes, but the store requires a cursor path.",
    )
    args = parser.parse_args()
    store = HumanInputStore(args.file, args.cursor or args.file.with_suffix(".cursor"))
    print(f"writing human notes to: {args.file}")
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


if __name__ == "__main__":
    main()
