from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.models import utc_now_iso


@dataclass(frozen=True)
class HumanInputBatch:
    notes: list[str]
    text: str
    cursor_before: int
    cursor_after: int


class HumanInputStore:
    def __init__(self, input_path: Path, cursor_path: Path) -> None:
        self.input_path = input_path
        self.cursor_path = cursor_path
        self.input_path.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)

    def append_note(self, note: str, timestamp: str | None = None) -> None:
        cleaned = clean_text(note, max_chars=2000)
        if not cleaned:
            return
        with self.input_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"\n## {timestamp or utc_now_iso()}\n{cleaned}\n")

    def read_new_notes(self) -> HumanInputBatch:
        if not self.input_path.exists():
            self._write_cursor(0)
            return HumanInputBatch([], "", 0, 0)
        content = self.input_path.read_bytes()
        file_size = len(content)
        cursor_before = self._read_cursor()
        if cursor_before > file_size:
            cursor_before = 0
        new_text = content[cursor_before:].decode("utf-8", errors="ignore")
        notes = _parse_notes(new_text)
        self._write_cursor(file_size)
        return HumanInputBatch(notes, "\n".join(notes), cursor_before, file_size)

    def _read_cursor(self) -> int:
        if not self.cursor_path.exists():
            return 0
        try:
            return max(0, int(self.cursor_path.read_text(encoding="utf-8").strip() or "0"))
        except ValueError:
            return 0

    def _write_cursor(self, value: int) -> None:
        self.cursor_path.write_text(str(max(0, value)), encoding="utf-8")


def _parse_notes(raw: str) -> list[str]:
    notes: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            if current:
                notes.append(clean_text(" ".join(current), max_chars=2000))
                current = []
            continue
        current.append(stripped)
    if current:
        notes.append(clean_text(" ".join(current), max_chars=2000))
    return [note for note in notes if note]
