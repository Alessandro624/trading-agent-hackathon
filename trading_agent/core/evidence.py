from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class Evidence:
    source_id: str
    title: str
    url: str
    publisher: str
    published_at: str | None
    provider: str
    query: str
    excerpt: str
    confidence: float = 0.0
    is_inference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls(
            source_id=str(data.get("source_id") or _hash_parts(data.get("provider", ""), data.get("url", ""), data.get("title", ""))),
            title=str(data.get("title") or ""),
            url=str(data.get("url") or ""),
            publisher=str(data.get("publisher") or ""),
            published_at=str(data.get("published_at") or "") or None,
            provider=str(data.get("provider") or "inference"),
            query=str(data.get("query") or ""),
            excerpt=str(data.get("excerpt") or "")[:500],
            confidence=_clamp_float(data.get("confidence"), 0.0, 1.0, 0.0),
            is_inference=bool(data.get("is_inference", False)),
        )


def make_evidence(
    *,
    title: str,
    url: str = "",
    published_at: str | None = None,
    provider: str,
    query: str,
    excerpt: str = "",
    confidence: float = 0.0,
    is_inference: bool = False,
) -> Evidence:
    publisher = _publisher_from_url(url) if url else (provider if is_inference else "")
    source_id = _hash_parts(provider, url, title)
    return Evidence(
        source_id=source_id,
        title=title.strip()[:300],
        url=url.strip(),
        publisher=publisher,
        published_at=published_at,
        provider=provider,
        query=query.strip()[:200],
        excerpt=(excerpt or "").strip()[:500],
        confidence=_clamp_float(confidence, 0.0, 1.0, 0.0),
        is_inference=is_inference,
    )


def make_inference(
    *,
    title: str,
    query: str,
    excerpt: str,
    confidence: float = 0.0,
) -> Evidence:
    return make_evidence(
        title=title,
        url="",
        published_at=None,
        provider="inference",
        query=query,
        excerpt=excerpt,
        confidence=confidence,
        is_inference=True,
    )


def _publisher_from_url(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def _hash_parts(*parts: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return max(low, min(high, result))


def dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    result: list[Evidence] = []
    for item in items:
        if item.source_id in seen:
            continue
        seen.add(item.source_id)
        result.append(item)
    return result


def evidence_to_citations(items: list[Evidence]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": item.source_id,
            "title": item.title,
            "url": item.url,
            "publisher": item.publisher,
            "published_at": item.published_at,
            "provider": item.provider,
            "is_inference": item.is_inference,
            "clickable": bool(item.url and _is_http_url(item.url)),
        }
        for item in items
    ]


def _is_http_url(url: str) -> bool:
    return bool(url) and re.match(r"^https?://", url, re.IGNORECASE) is not None
