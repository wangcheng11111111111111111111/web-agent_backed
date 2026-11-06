from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

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
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
}

_TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


class WebCrawlerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def extract_url(text: str) -> str | None:
        match = re.search(r"https?://[^\s<>\"'()]+", text)
        if not match:
            return None
        url = match.group(0).rstrip(".,;:!?)]}>\"'")
        return url if urlparse(url).scheme in {"http", "https"} else None

    async def fetch(self, url: str) -> dict[str, Any] | None:
        if not self.settings.web_crawl_enabled:
            return None
        direct = await self._fetch_direct(url)
        if direct is not None:
            return direct
        return await self._fetch_reader(url)

    async def _fetch_direct(self, url: str) -> dict[str, Any] | None:
        timeout = httpx.Timeout(
            self.settings.web_crawl_timeout, connect=min(10.0, self.settings.web_crawl_timeout)
        )
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True, headers=_DEFAULT_HEADERS
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    await response.aread()
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/html" not in content_type and "text/plain" not in content_type:
                        logger.info(
                            "Crawl skipped: unsupported content_type=%s",
                            content_type,
                        )
                        return None
                    text = response.text
        except httpx.HTTPError as exc:
            logger.warning("Crawl request failed: %s", type(exc).__name__)
            return None

        title, body = self._extract(text)
        if not body:
            logger.info("Crawl produced no body: %s", url)
            return None
        return {
            "url": url,
            "title": title[:200],
            "content": body[: self.settings.web_crawl_max_chars],
        }

    async def _fetch_reader(self, url: str) -> dict[str, Any] | None:
        reader_url = f"https://r.jina.ai/{url}"
        timeout = httpx.Timeout(
            self.settings.web_crawl_timeout,
            connect=min(10.0, self.settings.web_crawl_timeout),
        )
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    reader_url,
                    headers={
                        "Accept": "text/plain,text/markdown;q=0.9,*/*;q=0.5",
                        "User-Agent": _DEFAULT_HEADERS["User-Agent"],
                    },
                )
                response.raise_for_status()
                content = response.text.strip()
        except httpx.HTTPError as exc:
            logger.warning(
                "Crawl reader fallback failed: %s url=%s",
                type(exc).__name__,
                url,
            )
            return None
        if not content:
            logger.info("Crawl reader returned no body: %s", url)
            return None
        title = ""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                break
        logger.info(
            "Crawl reader fallback succeeded chars=%s url=%s",
            len(content),
            url,
        )
        return {
            "url": url,
            "title": title[:200],
            "content": content[: self.settings.web_crawl_max_chars],
        }

    async def fetch_safe(self, url: str) -> dict[str, Any] | None:
        try:
            return await self.fetch(url)
        except Exception as exc:
            logger.warning(
                "Crawl failed safely (%s: %s)",
                type(exc).__name__,
                str(exc),
            )
            return None

    def _extract(self, raw: str) -> tuple[str, str]:
        title = ""
        title_match = _TITLE_PATTERN.search(raw)
        if title_match:
            title = title_match.group(1).strip()
        body_parts: list[str] = []
        try:
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
                tag.decompose()
            for block in soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "code"]):
                text = block.get_text(" ", strip=True)
                if text and len(text) > 1:
                    body_parts.append(text)
        except Exception as exc:
            logger.warning("Crawl parse failed: %s", type(exc).__name__)
        if not body_parts:
            plain = re.sub(r"<[^>]+>", " ", raw)
            body_parts = [part for part in (re.split(r"\s+", plain)) if part]
            return title, " ".join(body_parts)
        return title, "\n".join(body_parts)
