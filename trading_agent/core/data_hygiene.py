from __future__ import annotations

import html
import re
from typing import Any

_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any, *, max_chars: int = 1000) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text[:max_chars]


def clean_news_article(article: dict[str, Any], *, max_text_chars: int = 5000) -> dict[str, Any]:
    source = article.get("source")
    if isinstance(source, dict):
        source = source.get("name")
    return {
        "title": clean_text(article.get("title"), max_chars=240),
        "description": clean_text(article.get("description"), max_chars=700),
        "content": clean_text(article.get("content"), max_chars=max_text_chars),
        "source": clean_text(source, max_chars=120),
        "url": clean_text(article.get("url"), max_chars=1000),
        "published_at": clean_text(article.get("published_at") or article.get("publishedAt"), max_chars=80),
        "endpoint": clean_text(article.get("endpoint"), max_chars=80),
        "query_strategy": clean_text(article.get("query_strategy"), max_chars=80),
        "sort_by": clean_text(article.get("sort_by"), max_chars=80),
        "search_in": clean_text(article.get("search_in"), max_chars=120),
    }


def clean_news_items(items: Any, *, max_items: int = 20) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            cleaned.append(clean_news_article(item))
    return cleaned
