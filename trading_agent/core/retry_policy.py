from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    delay_seconds: float = 0.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)

    def run(self, operation: Callable[[], T], on_failure: Callable[[int, Exception], None] | None = None) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except self.retryable_exceptions as error:
                last_error = error
                if on_failure:
                    on_failure(attempt, error)
                if attempt < self.max_attempts and self.delay_seconds > 0:
                    sleep(self.delay_seconds)
        assert last_error is not None
        raise last_error
