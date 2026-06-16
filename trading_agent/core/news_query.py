from __future__ import annotations

ALLOWED_NEWS_STRATEGIES = {"everything", "top_headlines"}
ALLOWED_NEWS_SORT_BY = {"publishedAt", "relevancy", "popularity"}
ALLOWED_NEWS_SEARCH_IN = {
    None,
    "title",
    "description",
    "content",
    "title,description",
    "title,content",
    "description,content",
    "title,description,content",
}


def validate_news_query(
    strategy: str,
    sort_by: str,
    search_in: str | None,
) -> tuple[str, str, str | None]:
    return (
        _validated(strategy, ALLOWED_NEWS_STRATEGIES, "strategy"),
        _validated(sort_by, ALLOWED_NEWS_SORT_BY, "sort_by"),
        _validated(search_in, ALLOWED_NEWS_SEARCH_IN, "search_in"),
    )


def _validated(value, allowed: set, name: str):
    if value not in allowed:
        raise ValueError(f"invalid NewsAPI {name}: {value}")
    return value
