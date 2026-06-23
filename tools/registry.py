import logging
import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Agent-Orchestration-System/1.0 (portfolio)"}


def search_sec_edgar(query: str, limit: int = 3) -> list[dict]:
    results = []
    try:
        r = requests.get(
            "https://efts.sec.gov/LATEST/search-index?q=%22" +
            query.replace(" ", "%20") + "%22&dateRange=custom"
            "&startdt=2020-01-01&forms=10-K,10-Q",
            headers=HEADERS, timeout=5,
        )
        hits = r.json().get("hits", {}).get("hits", [])
        for h in hits[:limit]:
            src = h.get("_source", {})
            title = src.get("display_names", [""])[0] or src.get("file_date", "")
            text = f"{src.get('period_of_report', '')} {src.get('form_type', '')}"
            if title:
                results.append({
                    "title": title,
                    "source": "SEC EDGAR",
                    "text": f"{title}. {text}",
                })
    except Exception as e:
        logger.warning(f"SEC EDGAR error: {e}")
    return results


def search_news(query: str, limit: int = 3) -> list[dict]:
    results = []
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "srlimit": limit, "format": "json"},
            headers=HEADERS, timeout=5,
        )
        pages = r.json().get("query", {}).get("search", [])
        for p in pages[:limit]:
            title = p.get("title", "")
            snippet = p.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            if title:
                results.append({
                    "title": title,
                    "source": "Wikipedia/News",
                    "text": f"{title}. {snippet}",
                })
    except Exception as e:
        logger.warning(f"News search error: {e}")
    return results


def extract_document_facts(text: str, max_chars: int = 3000) -> str:
    return text[:max_chars]
