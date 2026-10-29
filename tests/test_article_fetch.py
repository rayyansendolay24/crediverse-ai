"""
tests/test_article_fetch.py
───────────────────────────────
Mocked tests for core/article_fetch.py (URL input support) - no network
needed. Covers: URL validation, a successful extraction via the
BeautifulSoup fallback path (exercised directly so the test doesn't
depend on whether trafilatura is installed), a non-HTML response, a
network failure, and a page with too little extractable text.

Run with:  python -m tests.test_article_fetch
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from core.article_fetch import ArticleFetchError, fetch_article_from_url

_ARTICLE_HTML = """
<html><head><title>Example Headline - Example News</title></head>
<body>
<nav>Home | World | Sport</nav>
<article>
<h1>Example Headline</h1>
<p>This is the first real paragraph of the article, long enough to count as content.</p>
<p>This is the second real paragraph, continuing the story with more detail and context.</p>
<p>A third paragraph wraps up the piece with a concluding thought about the subject matter.</p>
</article>
<footer>Copyright 2026</footer>
</body></html>
"""


def _fake_response(text: str, content_type: str = "text/html; charset=utf-8", status: int = 200):
    resp = SimpleNamespace()
    resp.text = text
    resp.headers = {"Content-Type": content_type}
    resp.status_code = status

    def raise_for_status():
        if status >= 400:
            raise requests.HTTPError(f"{status} error")

    resp.raise_for_status = raise_for_status
    return resp


def test_rejects_invalid_url():
    for bad in ["", "not a url", "ftp://example.com/file", "example.com/no-scheme"]:
        try:
            fetch_article_from_url(bad)
            raise AssertionError(f"expected ArticleFetchError for {bad!r}")
        except ArticleFetchError:
            pass
    print("test_rejects_invalid_url: OK")


def test_extracts_article_via_fallback():
    # Force the trafilatura path to miss so we exercise the bs4 fallback
    # deterministically regardless of whether trafilatura is installed.
    with patch("requests.get", return_value=_fake_response(_ARTICLE_HTML)), \
         patch("core.article_fetch._extract_with_trafilatura", return_value=(None, None, None, None)):
        article = fetch_article_from_url("https://news.example.com/story")

    assert article.url == "https://news.example.com/story"
    assert "first real paragraph" in article.text
    assert "second real paragraph" in article.text
    assert article.title == "Example Headline - Example News"
    print("test_extracts_article_via_fallback: OK")
    print(f"  extracted {len(article.text)} chars, title={article.title!r}")


def test_rejects_non_html_response():
    with patch("requests.get", return_value=_fake_response("{}", content_type="application/json")):
        try:
            fetch_article_from_url("https://api.example.com/data.json")
            raise AssertionError("expected ArticleFetchError for a non-HTML response")
        except ArticleFetchError as e:
            assert "does not look like an HTML" in str(e)
    print("test_rejects_non_html_response: OK")


def test_wraps_network_failure():
    with patch("requests.get", side_effect=requests.ConnectionError("boom")):
        try:
            fetch_article_from_url("https://unreachable.example.com/story")
            raise AssertionError("expected ArticleFetchError on a network failure")
        except ArticleFetchError as e:
            assert "Could not download" in str(e)
    print("test_wraps_network_failure: OK")


def test_rejects_too_little_text():
    thin_html = "<html><head><title>Thin page</title></head><body><p>Too short.</p></body></html>"
    with patch("requests.get", return_value=_fake_response(thin_html)), \
         patch("core.article_fetch._extract_with_trafilatura", return_value=(None, None, None, None)):
        try:
            fetch_article_from_url("https://news.example.com/stub")
            raise AssertionError("expected ArticleFetchError for too-short extracted text")
        except ArticleFetchError as e:
            assert "Could not extract enough article text" in str(e)
    print("test_rejects_too_little_text: OK")


if __name__ == "__main__":
    test_rejects_invalid_url()
    test_extracts_article_via_fallback()
    test_rejects_non_html_response()
    test_wraps_network_failure()
    test_rejects_too_little_text()
    print("\nALL ARTICLE-FETCH TESTS PASSED")
