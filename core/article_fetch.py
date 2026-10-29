"""
core/article_fetch.py
────────────────────────
Fetches a news article by URL and extracts clean article text, so the
API/UI can accept a link instead of (or alongside) pasted text.

Uses trafilatura, which strips news-site boilerplate (nav, ads,
comments, related-stories rails) far better than a plain regex strip.
Falls back to a BeautifulSoup paragraph-extraction pass if trafilatura
isn't installed or returns nothing, rather than silently failing.

Deliberately does NOT try to execute JavaScript - sites that render
article text client-side (rare for news, but it happens) will fail
extraction here with a clear ArticleFetchError rather than a vague
empty result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

_TIMEOUT_SECONDS = 15
_MIN_EXTRACTED_CHARS = 200
_USER_AGENT = (
    "Mozilla/5.0 (compatible; MachineCredibilityResearchBot/1.0; "
    "academic project, not for redistribution)"
)


class ArticleFetchError(Exception):
    """Raised when a URL can't be fetched, isn't an HTML article page,
    or yields no extractable article text. Callers turn this into a
    clean 4xx instead of a raw stack trace."""


@dataclass
class FetchedArticle:
    url: str
    title: Optional[str]
    text: str
    site_name: Optional[str]
    image_url: Optional[str]


def _validate_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ArticleFetchError("URL is empty.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ArticleFetchError(f"'{url}' is not a valid http(s) URL.")
    return url


def _download(url: str) -> requests.Response:
    try:
        resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ArticleFetchError(f"Could not download '{url}': {exc}") from exc

    content_type = resp.headers.get("Content-Type", "")
    if content_type and "html" not in content_type.lower():
        raise ArticleFetchError(
            f"'{url}' does not look like an HTML article page (Content-Type: {content_type})."
        )
    return resp


def _extract_with_trafilatura(html: str, url: str):
    try:
        import trafilatura
    except ImportError:
        return None, None, None, None
    try:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = getattr(meta, "title", None) if meta else None
        site_name = getattr(meta, "sitename", None) if meta else None
        image_url = getattr(meta, "image", None) if meta else None
        return text, title, site_name, image_url
    except Exception:
        return None, None, None, None


def _fallback_extract_text(html: str) -> Optional[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p.split()) > 5]
    return "\n\n".join(paragraphs) if paragraphs else None


def _fallback_extract_title(html: str) -> Optional[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


def fetch_article_from_url(url: str) -> FetchedArticle:
    """Downloads `url` and extracts the article title + body text.
    Raises ArticleFetchError on any failure so callers can surface a
    clean message instead of a stack trace."""
    url = _validate_url(url)
    html = _download(url).text

    text, title, site_name, image_url = _extract_with_trafilatura(html, url)
    if not text:
        text = _fallback_extract_text(html)

    if not text or len(text.strip()) < _MIN_EXTRACTED_CHARS:
        raise ArticleFetchError(
            f"Could not extract enough article text from '{url}' "
            "(the page may be paywalled, JavaScript-rendered, or not a news article)."
        )

    if not title:
        title = _fallback_extract_title(html)

    return FetchedArticle(url=url, title=title, text=text.strip(), site_name=site_name, image_url=image_url)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    article = fetch_article_from_url(target)
    print(f"title: {article.title}")
    print(f"site:  {article.site_name}")
    print(f"chars: {len(article.text)}")
    print(article.text[:400])
