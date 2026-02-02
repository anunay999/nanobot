"""Web tools: web_search and web_fetch."""

import html
import json
import os
import re
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


class WebSearchTool(Tool):
    """Search the web using Brave or Firecrawl Search API."""
    
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10},
            "provider": {
                "type": "string",
                "enum": ["auto", "brave", "firecrawl"],
                "description": "Search provider (default: auto)"
            }
        },
        "required": ["query"]
    }
    
    def __init__(
        self,
        brave_api_key: str | None = None,
        firecrawl_api_key: str | None = None,
        max_results: int = 5,
        default_provider: str = "auto",
    ):
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY", "")
        self.firecrawl_api_key = firecrawl_api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self.max_results = max_results
        self.default_provider = default_provider
    
    async def execute(
        self,
        query: str,
        count: int | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> str:
        n = min(max(count or self.max_results, 1), 10)
        selected_provider = (provider or self.default_provider or "auto").lower()
        
        if selected_provider == "auto":
            if self.brave_api_key:
                selected_provider = "brave"
            elif self.firecrawl_api_key:
                selected_provider = "firecrawl"
        
        if selected_provider == "brave":
            if not self.brave_api_key:
                return "Error: BRAVE_API_KEY not configured"
            return await self._search_brave(query, n)
        
        if selected_provider == "firecrawl":
            if not self.firecrawl_api_key:
                return "Error: FIRECRAWL_API_KEY not configured"
            return await self._search_firecrawl(query, n)
        
        return "Error: Unknown provider. Use 'brave', 'firecrawl', or 'auto'."
    
    async def _search_brave(self, query: str, n: int) -> str:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": self.brave_api_key},
                    timeout=10.0,
                )
                r.raise_for_status()
            
            results = r.json().get("web", {}).get("results", [])
            return self._format_results(query, results, n)
        except Exception as e:
            return f"Error: {e}"
    
    async def _search_firecrawl(self, query: str, n: int) -> str:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.firecrawl.dev/v1/search",
                    json={"query": query, "limit": n},
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.firecrawl_api_key}",
                    },
                    timeout=15.0,
                )
                r.raise_for_status()
            
            payload = r.json()
            results = payload.get("data") or payload.get("results") or []
            return self._format_results(query, results, n)
        except Exception as e:
            return f"Error: {e}"
    
    def _format_results(self, query: str, results: list[dict[str, Any]], n: int) -> str:
        if not results:
            return f"No results for: {query}"
        
        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results[:n], 1):
            title = item.get("title", "") or item.get("name", "")
            url = item.get("url", "") or item.get("link", "")
            desc = (
                item.get("description")
                or item.get("snippet")
                or item.get("content")
                or ""
            )
            lines.append(f"{i}. {title}\n   {url}")
            if desc:
                lines.append(f"   {desc}")
        return "\n".join(lines)


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""
    
    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }
    
    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars
    
    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        from readability import Document
        
        max_chars = maxChars or self.max_chars
        
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30.0)
                r.raise_for_status()
            
            ctype = r.headers.get("content-type", "")
            
            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = self._to_markdown(doc.summary()) if extractMode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"
            
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            
            return json.dumps({"url": url, "finalUrl": str(r.url), "status": r.status_code,
                              "extractor": extractor, "truncated": truncated, "length": len(text), "text": text})
        except Exception as e:
            return json.dumps({"error": str(e), "url": url})
    
    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
