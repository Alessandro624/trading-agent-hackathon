from __future__ import annotations

from typing import Any
import json


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 20,
) -> dict[str, Any]:
    import requests

    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    if not response.content:
        return {}
    return json.loads(response.text)
