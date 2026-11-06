from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from app.config import Settings

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class WebSearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, query: str) -> list[dict[str, Any]]:
        if not self.settings.web_search_enabled:
            return []
        provider = self.settings.web_search_provider
        key = (self.settings.web_search_api_key or "").strip()

        if provider == "tavily" and key:
            results = await self._tavily(query, key)
        elif provider == "serper" and key:
            results = await self._serper(query, key)
        else:
            results = await self._bing(query)
            if not results:
                results = await self._sogou(query)

        if not results:
            logger.info("Web search provider=%s returned no results", provider)
        return results[: self.settings.web_search_top_k]

    async def search_safe(self, query: str) -> list[dict[str, Any]]:
        try:
            return await self.search(query)
        except Exception as exc:
            logger.warning(
                "Web search failed safely (%s: %s)",
                type(exc).__name__,
                str(exc),
            )
            return []

    async def _bing(self, query: str) -> list[dict[str, Any]]:
        url = f"https://www.bing.com/search?q={quote(query)}&ensearch=0"
        timeout = httpx.Timeout(self.settings.web_search_timeout, connect=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                r = await c.get(url, headers=_DEFAULT_HEADERS)
                r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Bing request failed: %s", type(exc).__name__)
            return []

        results: list[dict[str, Any]] = []
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for li in soup.select("li.b_algo"):
                h2 = li.select_one("h2 a")
                if not h2:
                    continue
                title = h2.get_text(strip=True)
                href = h2.get("href", "")
                if not title or not href or not href.startswith("http"):
                    continue
                snippet_tag = li.select_one("p, .b_caption p, .b_snippet")
                snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
                results.append(
                    {
                        "title": title[:200],
                        "url": href[:500],
                        "snippet": snippet[:500],
                    }
                )
                if len(results) >= self.settings.web_search_top_k:
                    break
        except Exception as exc:
            logger.warning("Bing parse failed: %s", type(exc).__name__)
        return results

    async def _sogou(self, query: str) -> list[dict[str, Any]]:
        url = f"https://www.sogou.com/web?query={quote(query)}"
        timeout = httpx.Timeout(self.settings.web_search_timeout, connect=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                r = await c.get(url, headers=_DEFAULT_HEADERS)
                r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Sogou request failed: %s", type(exc).__name__)
            return []

        results: list[dict[str, Any]] = []
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            for box in soup.select(".vrwrap, .rb"):
                h3 = box.select_one("h3 a")
                if not h3:
                    continue
                title = h3.get_text(strip=True)
                href = h3.get("href", "")
                if not title or not href:
                    continue
                snippet_tag = box.select_one(".str_info, .ft, .txt-info, p")
                snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
                results.append(
                    {
                        "title": title[:200],
                        "url": href[:500],
                        "snippet": snippet[:500],
                    }
                )
                if len(results) >= self.settings.web_search_top_k:
                    break
        except Exception as exc:
            logger.warning("Sogou parse failed: %s", type(exc).__name__)
        return results

    async def _tavily(
        self, query: str, api_key: str
    ) -> list[dict[str, Any]]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": self.settings.web_search_top_k,
            "include_answer": False,
            "include_raw_content": False,
        }
        timeout = httpx.Timeout(self.settings.web_search_timeout, connect=5.0)
        results: list[dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            for item in data.get("results", [])[: self.settings.web_search_top_k]:
                results.append(
                    {
                        "title": str(item.get("title", ""))[:200],
                        "url": str(item.get("url", ""))[:500],
                        "snippet": str(item.get("content", ""))[:500],
                    }
                )
        except httpx.HTTPError as exc:
            logger.warning("Tavily request failed: %s", type(exc).__name__)
        except Exception as exc:
            logger.warning("Tavily parse failed: %s", type(exc).__name__)
        return results

    async def _serper(
        self, query: str, api_key: str
    ) -> list[dict[str, Any]]:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        timeout = httpx.Timeout(self.settings.web_search_timeout, connect=5.0)
        results: list[dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(
                    url,
                    headers=headers,
                    json={"q": query, "num": self.settings.web_search_top_k},
                )
                r.raise_for_status()
                data = r.json()
            for item in data.get("organic", [])[: self.settings.web_search_top_k]:
                results.append(
                    {
                        "title": str(item.get("title", ""))[:200],
                        "url": str(item.get("link", ""))[:500],
                        "snippet": str(item.get("snippet", ""))[:500],
                    }
                )
        except httpx.HTTPError as exc:
            logger.warning("Serper request failed: %s", type(exc).__name__)
        except Exception as exc:
            logger.warning("Serper parse failed: %s", type(exc).__name__)
        return results