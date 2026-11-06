from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
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
        api_results = await self._search_api(query)
        if api_results:
            logger.info(
                "Web search provider=%s result_count=%s",
                self.settings.web_search_provider,
                len(api_results),
            )
            return api_results[: self.settings.web_search_top_k]
        return await self._search_bing(query)

    async def _search_bing(self, query: str) -> list[dict[str, Any]]:
        if not self.settings.web_search_enabled:
            return []
        now = datetime.now(timezone(timedelta(hours=8)))
        query_date = self._resolve_query_date(query, now)
        queries = [query]
        if query_date and query_date not in query:
            queries.append(f"{query} {query_date}")

        is_esports = bool(
            re.search(r"(?:cs2|csgo|counter[- ]?strike)", query, re.I)
        )
        if is_esports:
            date_suffix = f" {query_date}" if query_date else ""
            queries.extend(
                [
                    f"CS2 match results{date_suffix} site:hltv.org",
                    (
                        "CS2 match schedule"
                        f"{date_suffix} site:liquipedia.net/counterstrike"
                    ),
                ]
            )

        results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for current_query in dict.fromkeys(queries):
            current = await self._bing(current_query)
            if not current:
                current = await self._bing_rss(current_query)
            for item in current:
                url = str(item.get("url", ""))
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(item)
            if len(results) >= self.settings.web_search_top_k:
                break

        if is_esports:
            direct_sources: list[dict[str, Any]] = []
            if query_date:
                hltv_url = (
                    "https://www.hltv.org/results"
                    f"?startDate={query_date}&endDate={query_date}"
                )
                direct_sources.append(
                    {
                        "title": f"HLTV CS2 results {query_date}",
                        "url": hltv_url,
                        "snippet": f"HLTV results from {query_date}",
                    }
                )
            direct_sources.append(
                {
                    "title": "Liquipedia Counter-Strike matches",
                    "url": "https://liquipedia.net/counterstrike/Matches",
                    "snippet": "Counter-Strike match schedule and results",
                }
            )
            results = [
                *direct_sources,
                *(
                    item
                    for item in results
                    if item.get("url") not in {
                        source["url"] for source in direct_sources
                    }
                ),
            ]
        if not results:
            logger.info("Bing returned no results")
        return results[: self.settings.web_search_top_k]

    async def _search_api(self, query: str) -> list[dict[str, Any]]:
        provider = self.settings.web_search_provider.strip().lower()
        api_key = self.settings.web_search_api_key.strip()
        if not provider or provider == "bing" or not api_key:
            return []
        try:
            if provider == "tavily":
                return await self._tavily(query, api_key)
            if provider == "serper":
                return await self._serper(query, api_key)
            logger.warning(
                "Unknown web search provider=%s; falling back to Bing",
                provider,
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning(
                "Web search API failed; falling back to Bing (%s)",
                type(exc).__name__,
            )
        return []

    async def _tavily(
        self,
        query: str,
        api_key: str,
    ) -> list[dict[str, Any]]:
        timeout = httpx.Timeout(self.settings.web_search_timeout, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": self.settings.web_search_top_k,
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return [
            {
                "title": str(item.get("title", ""))[:200],
                "url": str(item.get("url", ""))[:500],
                "snippet": str(item.get("content", ""))[:500],
            }
            for item in payload.get("results", [])[
                : self.settings.web_search_top_k
            ]
            if item.get("url")
        ]

    async def _serper(
        self,
        query: str,
        api_key: str,
    ) -> list[dict[str, Any]]:
        timeout = httpx.Timeout(self.settings.web_search_timeout, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "num": self.settings.web_search_top_k,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return [
            {
                "title": str(item.get("title", ""))[:200],
                "url": str(item.get("link", ""))[:500],
                "snippet": str(item.get("snippet", ""))[:500],
            }
            for item in payload.get("organic", [])[
                : self.settings.web_search_top_k
            ]
            if item.get("link")
        ]

    @staticmethod
    def _resolve_query_date(
        query: str,
        now: datetime,
    ) -> str | None:
        explicit = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
        if explicit:
            return explicit.group(1)
        if "前天" in query:
            return (now - timedelta(days=2)).date().isoformat()
        if "昨天" in query or "昨日" in query or "yesterday" in query.lower():
            return (now - timedelta(days=1)).date().isoformat()
        if "明天" in query or "tomorrow" in query.lower():
            return (now + timedelta(days=1)).date().isoformat()
        if any(
            word in query
            for word in ("今天", "今日", "today", "current")
        ):
            return now.date().isoformat()
        return None

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
        url = (
            f"https://www.bing.com/search?q={quote(query)}"
            "&ensearch=0&mkt=zh-CN&setlang=zh-hans&cc=CN"
        )
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

    async def _bing_rss(self, query: str) -> list[dict[str, Any]]:
        url = (
            f"https://www.bing.com/search?q={quote(query)}"
            "&format=rss&mkt=zh-CN&setlang=zh-hans&cc=CN"
        )
        timeout = httpx.Timeout(self.settings.web_search_timeout, connect=5.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url, headers=_DEFAULT_HEADERS)
                response.raise_for_status()
            root = ET.fromstring(response.text)
        except (httpx.HTTPError, ET.ParseError) as exc:
            logger.warning(
                "Bing RSS request/parse failed: %s",
                type(exc).__name__,
            )
            return []

        results: list[dict[str, Any]] = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            href = (item.findtext("link") or "").strip()
            description = (item.findtext("description") or "").strip()
            if not title or not href.startswith("http"):
                continue
            snippet = BeautifulSoup(description, "html.parser").get_text(
                " ",
                strip=True,
            )
            results.append(
                {
                    "title": title[:200],
                    "url": href[:500],
                    "snippet": snippet[:500],
                }
            )
            if len(results) >= self.settings.web_search_top_k:
                break
        return results
