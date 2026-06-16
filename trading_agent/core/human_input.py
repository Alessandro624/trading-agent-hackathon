from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Callable, TypeVar

from trading_agent.core.data_hygiene import clean_text
from trading_agent.core.models import utc_now_iso

T = TypeVar("T")


@dataclass(frozen=True)
class HumanInputBatch:
    notes: list[str]
    text: str
    cursor_before: int
    cursor_after: int


class HumanInputStore:
    def __init__(self, input_path: Path, cursor_path: Path, io_retries: int = 5, retry_seconds: float = 0.05) -> None:
        self.input_path = input_path
        self.cursor_path = cursor_path
        self.io_retries = max(1, io_retries)
        self.retry_seconds = max(0.0, retry_seconds)
        self.input_path.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)

    def append_note(self, note: str, timestamp: str | None = None) -> None:
        cleaned = clean_text(note, max_chars=2000)
        if not cleaned:
            return
        self._with_io_retry(lambda: self._append_input_text(f"\n## {timestamp or utc_now_iso()}\n{cleaned}\n"))

    def read_new_notes(self) -> HumanInputBatch:
        if not self.input_path.exists():
            self._write_cursor(0)
            return HumanInputBatch([], "", 0, 0)
        content = self._with_io_retry(self._read_input_bytes)
        file_size = len(content)
        cursor_before = self._read_cursor()
        if cursor_before > file_size:
            cursor_before = 0
        new_text = content[cursor_before:].decode("utf-8", errors="ignore")
        notes = _parse_notes(new_text)
        self._write_cursor(file_size)
        return HumanInputBatch(notes, "\n".join(notes), cursor_before, file_size)

    def read_next_note(self) -> HumanInputBatch:
        if not self.input_path.exists():
            self._write_cursor(0)
            return HumanInputBatch([], "", 0, 0)
        content = self._with_io_retry(self._read_input_bytes)
        file_size = len(content)
        cursor_before = self._read_cursor()
        if cursor_before > file_size:
            cursor_before = 0
        note, cursor_after = _parse_next_note(content, cursor_before)
        self._write_cursor(cursor_after)
        notes = [note] if note else []
        return HumanInputBatch(notes, note or "", cursor_before, cursor_after)

    def _append_input_text(self, text: str) -> None:
        with self.input_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def _read_input_bytes(self) -> bytes:
        return self.input_path.read_bytes()

    def _read_cursor(self) -> int:
        if not self.cursor_path.exists():
            return 0
        try:
            return max(0, int(self.cursor_path.read_text(encoding="utf-8").strip() or "0"))
        except ValueError:
            return 0

    def _write_cursor(self, value: int) -> None:
        self._with_io_retry(lambda: self.cursor_path.write_text(str(max(0, value)), encoding="utf-8"))

    def _with_io_retry(self, action: Callable[[], T]) -> T:
        last_error: PermissionError | None = None
        for attempt in range(self.io_retries):
            try:
                return action()
            except PermissionError as error:
                last_error = error
                if attempt < self.io_retries - 1 and self.retry_seconds > 0:
                    sleep(self.retry_seconds)
        if last_error is not None:
            raise last_error
        return action()


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


def _parse_next_note(content: bytes, cursor: int) -> tuple[str | None, int]:
    file_size = len(content)
    offset = max(0, min(cursor, file_size))
    current: list[str] = []
    for raw_line in content[offset:].splitlines(keepends=True):
        line_start = offset
        offset += len(raw_line)
        stripped = raw_line.decode("utf-8", errors="ignore").strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            if current:
                return clean_text(" ".join(current), max_chars=2000), line_start
            continue
        current.append(stripped)
    if current:
        return clean_text(" ".join(current), max_chars=2000), file_size
    return None, file_size
