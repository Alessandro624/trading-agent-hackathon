from __future__ import annotations

from urllib.parse import urlencode

from trading_agent.core import validate_news_query
from trading_agent.utils import request_json, require_env


class NewsApiProvider:
    def search_news(self, ticker: str, limit: int = 5) -> list[dict]:
        return self.search_market_news(ticker=ticker, limit=limit)

    def search_market_news(
        self,
        ticker: str,
        strategy: str = "everything",
        limit: int = 5,
        sort_by: str = "publishedAt",
        search_in: str | None = None,
    ) -> list[dict]:
        strategy, sort_by, search_in = validate_news_query(strategy, sort_by, search_in)
        endpoint = "top-headlines" if strategy == "top_headlines" else "everything"
        params = {
            "q": ticker.upper(),
            "pageSize": min(max(int(limit), 1), 20),
            "language": "en",
            "apiKey": require_env("NEWS_API_KEY"),
        }
        if strategy == "everything":
            params["sortBy"] = sort_by
            if search_in:
                params["searchIn"] = search_in
        else:
            params["country"] = "us"
        query = urlencode(params)
        payload = request_json(f"https://newsapi.org/v2/{endpoint}?{query}")
        articles = payload.get("articles", [])
        return [
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("content"),
                "source": (article.get("source") or {}).get("name"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "endpoint": endpoint,
                "query_strategy": strategy,
                "sort_by": sort_by if strategy == "everything" else None,
                "search_in": search_in,
            }
            for article in articles
        ]
