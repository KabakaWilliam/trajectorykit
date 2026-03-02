import requests
from typing import Any, Callable, Dict, List, Optional,Tuple, Union
from .utils import SUPPORTED_LANGUAGES, API_TIMEOUT, MAX_RETRIES, INITIAL_RETRY_DELAY
from .config import MAX_RECURSION_DEPTH, SUB_AGENT_TURN_BUDGET, CONTEXT_WINDOW
from .tracing import EpisodeTrace  # adjust import path to wherever EpisodeTrace lives
import logging
import os
import threading
import time
import traceback
import uuid
import json
import random
from datetime import datetime


# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, just continue
    pass

logger = logging.getLogger(__name__)

ToolReturn = Tuple[str, Optional[EpisodeTrace]]

DEFAULT_NUM_SEARCHES = 5

# ── Search backend selection ─────────────────────────────────────────────
# Set SEARCH_BACKEND env var to switch: "serper" (default) or "serpapi"
# Each backend needs its own API key:
#   - serper:  SERPER_API_KEY  (from serper.dev)
#   - serpapi: SERP_API_KEY    (from serpapi.com)

SEARCH_TIMEOUT = 25   # seconds — complex quoted queries need more time
MAX_SEARCH_RETRIES = 2


def _search_serper(q: str, num_results: int = 5) -> str:
    """Google search via Serper.dev API."""
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return "Error: Serper API key not configured. Set SERPER_API_KEY environment variable."

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": q,
        "num": min(num_results, 10),
        "gl": "us",
        "hl": "en",
    }

    last_error = None
    for attempt in range(MAX_SEARCH_RETRIES + 1):
        try:
            logger.info(f"Serper search (attempt {attempt+1}): {q}")
            response = requests.post(url, json=payload, headers=headers, timeout=SEARCH_TIMEOUT)
            response.raise_for_status()

            data = response.json()

            # Serper returns organic results under "organic"
            results = data.get("organic", [])
            if not results:
                return f"No results found for query: {q}"

            formatted_results = f"Search Results for '{q}':\n\n"
            for i, result in enumerate(results[:num_results], 1):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No snippet")
                formatted_results += f"{i}. {title}\n   URL: {link}\n   {snippet}\n\n"

            logger.info(f"Successfully retrieved {len(results[:num_results])} search results via Serper")
            return formatted_results

        except requests.exceptions.Timeout:
            last_error = (
                f"Search timeout: Query '{q}' timed out after {SEARCH_TIMEOUT}s. "
                "TIP: Simplify your query — remove quoted phrases, reduce to key terms, "
                "or split into multiple simpler searches."
            )
            logger.warning(f"Serper timeout (attempt {attempt+1}/{MAX_SEARCH_RETRIES+1}): {q}")
            if attempt < MAX_SEARCH_RETRIES:
                time.sleep(1 * (attempt + 1))
                continue
            return last_error
        except requests.exceptions.ConnectionError:
            return "Search error: Could not connect to Serper API. Check internet connection."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 401:
                return "Search error: Invalid Serper API key. Check SERPER_API_KEY."
            elif status == 429:
                return "Search error: Rate limit exceeded. Please try again later."
            else:
                error_msg = f"Search HTTP error {status}"
                logger.error(error_msg)
                return error_msg
        except json.JSONDecodeError:
            return "Search error: Invalid JSON response from Serper API"
        except Exception as e:
            error_msg = f"Search error: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    return last_error or "Search error: Unknown failure after retries"


def _search_serpapi(q: str, num_results: int = 5) -> str:
    """Google search via SerpAPI.com."""
    api_key = os.getenv("SERP_API_KEY", "")
    if not api_key:
        return "Error: SerpAPI key not configured. Set SERP_API_KEY environment variable."

    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": q,
        "api_key": api_key,
        "num": min(num_results, 10),
        "hl": "en",
        "gl": "us",
    }

    last_error = None
    for attempt in range(MAX_SEARCH_RETRIES + 1):
        try:
            logger.info(f"SerpAPI search (attempt {attempt+1}): {q}")
            response = requests.get(url, params=params, timeout=SEARCH_TIMEOUT)
            response.raise_for_status()

            data = response.json()

            if "error" in data:
                return f"Search API Error: {data.get('error', 'Unknown error')}"

            results = data.get("organic_results", [])
            if not results:
                return f"No results found for query: {q}"

            formatted_results = f"Search Results for '{q}':\n\n"
            for i, result in enumerate(results[:num_results], 1):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No snippet")
                formatted_results += f"{i}. {title}\n   URL: {link}\n   {snippet}\n\n"

            logger.info(f"Successfully retrieved {len(results[:num_results])} search results via SerpAPI")
            return formatted_results

        except requests.exceptions.Timeout:
            last_error = (
                f"Search timeout: Query '{q}' timed out after {SEARCH_TIMEOUT}s. "
                "TIP: Simplify your query — remove quoted phrases, reduce to key terms, "
                "or split into multiple simpler searches."
            )
            logger.warning(f"SerpAPI timeout (attempt {attempt+1}/{MAX_SEARCH_RETRIES+1}): {q}")
            if attempt < MAX_SEARCH_RETRIES:
                time.sleep(1 * (attempt + 1))
                continue
            return last_error
        except requests.exceptions.ConnectionError:
            return "Search error: Could not connect to SerpAPI. Check internet connection."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 401:
                return "Search error: Invalid API key. Check SERP_API_KEY."
            elif status == 403:
                return "Search error: API key not authorized for this request."
            elif status == 429:
                return "Search error: Rate limit exceeded. Please try again later."
            else:
                error_msg = f"Search HTTP error {status}"
                logger.error(error_msg)
                return error_msg
        except json.JSONDecodeError:
            return "Search error: Invalid JSON response from SerpAPI"
        except Exception as e:
            error_msg = f"Search error: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    return last_error or "Search error: Unknown failure after retries"


def _search_ddg(q: str, num_results: int = 5) -> str:
    """
    Resilient DDG search with caching, dedup, backoff, and concurrency control.
    """
    import random
    from .utils import _ddg_cache, _ddg_rate, _ddg_sem, _inflight, _inflight_lock, _normalize_query
    try:
        from ddgs import DDGS
    except ImportError:
        return "Search error: package not installed. Run: pip install ddgs"

    nq = _normalize_query(q)
    cache_key = f"ddg::{nq}::k{min(num_results,10)}"

    # 1) cache hit
    cached = _ddg_cache.get(cache_key)
    if cached is not None:
        return cached

    # 2) in-flight dedup: if same query already running, wait for it
    with _inflight_lock:
        if cache_key in _inflight:
            evt = _inflight[cache_key]
        else:
            evt = threading.Event()
            _inflight[cache_key] = evt
            evt = None

    if evt is not None:
        # wait for the first caller to fill cache
        evt.wait(timeout=30)
        cached = _ddg_cache.get(cache_key)
        return cached if cached is not None else f"Search error: in-flight query timed out for '{q}'"

    # We are the first caller
    try:
        with _ddg_sem:  # 3) concurrency control
            max_results = min(max(num_results, 1), 10)

            # 4) retries with backoff
            last_err = None
            for attempt in range(5):
                _ddg_rate.wait()  # 5) global rate limiting across threads

                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(q, max_results=max_results))

                    if not results:
                        out = f"No results found for query: {q}"
                        _ddg_cache.set(cache_key, out)
                        return out

                    formatted = f"Search Results for '{q}':\n\n"
                    for i, r in enumerate(results[:num_results], 1):
                        title = r.get("title") or "No title"
                        link = r.get("href") or r.get("link") or "No link"
                        snippet = r.get("body") or r.get("snippet") or "No snippet"
                        formatted += f"{i}. {title}\n   URL: {link}\n   {snippet}\n\n"

                    _ddg_cache.set(cache_key, formatted)
                    return formatted

                except Exception as e:
                    last_err = e
                    # exponential backoff + jitter
                    sleep_s = min(8.0, (0.5 * (2 ** attempt))) + random.uniform(0, 0.25)
                    time.sleep(sleep_s)

            return f"DDG search error after retries: {type(last_err).__name__}: {last_err}"

    finally:
        # release waiters
        with _inflight_lock:
            evt = _inflight.pop(cache_key, None)
            if evt:
                evt.set()


# Error patterns that indicate the primary search backend is exhausted/broken
# and we should fall back to DDG for all remaining queries.
_SEARCH_FALLBACK_ERRORS = (
    "Search HTTP error",
    "Search error: Rate limit exceeded",
    "Search error: Invalid Serper API key",
    "Search error: Invalid API key",
    "Search error: API key not authorized",
    "Search API Error",
)


def search_web(q: str, num_results: int = 5) -> str:
    """
    Execute a web search and return structured results.
    
    Backend priority:
      1. Primary: serper (default) or serpapi (set SEARCH_BACKEND env var)
      2. Fallback: DuckDuckGo (automatic if primary fails with credit/auth errors)
    
    Args:
        q: Search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        Formatted string with top search results or error message.
        Results from the DDG fallback are prefixed with [DDG fallback].
    """
    backend = os.getenv("SEARCH_BACKEND", "serper").lower()
    if backend == "serpapi":
        result = _search_serpapi(q, num_results)
    else:
        result = _search_serper(q, num_results)

    # If primary backend failed with a credit/auth/rate error, fall back to DDG
    if any(result.startswith(err) for err in _SEARCH_FALLBACK_ERRORS):
        primary_error = result
        logger.warning(f"Primary search failed ({primary_error}), falling back to DuckDuckGo")
        ddg_result = _search_ddg(q, num_results)
        if ddg_result.startswith("Search error:") or ddg_result.startswith("DDG search error:"):
            # DDG also failed — return both errors
            return f"{primary_error}\n[DDG fallback also failed: {ddg_result}]"
        # Return clean results — the model doesn't need to know about the fallback.
        # The switch is logged above for traceability.
        return ddg_result

    return result

# ── Browser-grade HTTP infrastructure ─────────────────────────────────
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}

# Persistent cookie jar — survives across requests within one agent run
import httpx as _httpx
_cookie_jar = _httpx.Cookies()

def _get_cookie_jar():
    return _cookie_jar

# Patterns that indicate a site blocked us
_BLOCK_PATTERNS = [
    "access denied", "403 forbidden", "just a moment",
    "checking your browser", "enable javascript", "captcha",
    "cloudflare", "blocked", "unusual traffic", "bot detection",
    "please verify", "security check", "are you a robot",
]

def _is_blocked(text: str, status_code: int) -> tuple[bool, str]:
    """Check if a response looks like a bot-block page."""
    if status_code in (403, 429, 503):
        return True, f"HTTP {status_code}"
    lower = text[:3000].lower()
    for pat in _BLOCK_PATTERNS:
        if pat in lower:
            return True, f"block pattern: {pat}"
    return False, ""


def _jina_reader_fallback(url: str, max_chars: int = 8000) -> str | None:
    """Last-resort fallback: use Jina Reader to render JS-heavy pages.
    
    Returns cleaned text or None on failure.
    """
    import httpx
    jina_url = f"https://r.jina.ai/{url}"
    try:
        resp = httpx.get(
            jina_url,
            headers={"Accept": "text/plain", "User-Agent": _BROWSER_HEADERS["User-Agent"]},
            timeout=25,
            follow_redirects=True,
        )
        if resp.status_code == 200 and len(resp.text.strip()) > 50:
            return resp.text.strip()[:max_chars]
    except Exception as e:
        logger.debug(f"Jina Reader fallback failed for {url}: {e}")
    return None


def fetch_url(url: str, max_chars: int = 8000, extract: str = "text",
              css_selector: str = "", max_retries: int = 2, timeout: int = 20) -> dict:
    """Fetch a web page with browser-grade headers, retries, and Jina fallback.

    3-tier strategy:
      1. Direct fetch with browser headers + persistent cookies
      2. Retry with exponential backoff (if blocked / transient error)
      3. Jina Reader fallback (JS rendering, last resort)

    Args:
        url: URL to fetch.
        max_chars: Max characters of extracted text to return.
        extract: "text" for cleaned page text, "table" for pipe-delimited tables.
        css_selector: Optional CSS selector to target specific elements.
        max_retries: Number of retry attempts on failure.
        timeout: Per-request timeout in seconds.

    Returns:
        Dict with keys: ok, content, url, blocked, reason, status_code, retries,
        and optionally source, hint.
    """
    import httpx
    from bs4 import BeautifulSoup
    import warnings

    cookies = _get_cookie_jar()
    last_error = ""
    last_status = 0
    retries_done = 0

    for attempt in range(1 + max_retries):
        try:
            resp = httpx.get(
                url, headers=_BROWSER_HEADERS, cookies=cookies,
                timeout=timeout, follow_redirects=True,
            )
            # Persist any cookies the server set
            cookies.update(resp.cookies)
            last_status = resp.status_code

            if resp.status_code >= 400:
                blocked, reason = True, f"HTTP {resp.status_code}"
                last_error = reason
                retries_done = attempt
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))  # backoff
                    continue
                break  # fall through to Jina

            # ── Detect binary/PDF responses before parsing as HTML ─────
            content_type = resp.headers.get("content-type", "")
            if "application/pdf" in content_type or (
                resp.content[:5] == b"%PDF-" and "html" not in content_type
            ):
                # URL serves a PDF — extract text with read_pdf instead of
                # returning binary garbage through the HTML parser.
                try:
                    pdf_text = read_pdf(url, max_chars=max_chars)
                    if pdf_text and len(pdf_text.strip()) > 20:
                        return {
                            "ok": True, "content": f"[PDF detected — extracted text]\n\n{pdf_text}",
                            "url": str(resp.url), "blocked": False, "reason": "",
                            "status_code": resp.status_code, "retries": attempt,
                            "source": "read_pdf",
                        }
                except Exception as pdf_err:
                    logger.debug(f"PDF extraction failed for {url}: {pdf_err}")
                # If PDF extraction failed, return structured error
                return {
                    "ok": False, "content": "", "url": str(resp.url),
                    "blocked": False, "reason": "URL serves a PDF file that could not be parsed",
                    "status_code": resp.status_code, "retries": attempt,
                    "hint": "Try read_pdf(url=...) directly, or search for an HTML version.",
                }

            # Check for soft blocks (200 but Cloudflare challenge page, etc.)
            blocked, reason = _is_blocked(resp.text, resp.status_code)
            if blocked:
                last_error = reason
                retries_done = attempt
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break  # fall through to Jina

            # ── Parse successfully ────────────────────────────────────
            content_type = resp.headers.get("content-type", "")
            raw = resp.text

            # Detect XML vs HTML
            if "xml" in content_type or raw.lstrip()[:20].startswith("<?xml"):
                parser = "xml"
            else:
                parser = "html.parser"

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*XMLParsedAsHTMLWarning.*")
                soup = BeautifulSoup(raw, parser)

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Apply CSS selector if provided
            if css_selector:
                selected = soup.select(css_selector)
                if selected:
                    from bs4 import BeautifulSoup as BS
                    combined = "\n".join(str(el) for el in selected)
                    soup = BS(combined, "html.parser")

            if extract == "table":
                tables = soup.find_all("table")
                rows = []
                for tbl in tables:
                    for tr in tbl.find_all("tr"):
                        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                        rows.append(" | ".join(cells))
                content = "\n".join(rows)[:max_chars] if rows else ""
                if not content:
                    content = soup.get_text(separator="\n", strip=True)[:max_chars]
            else:
                content = soup.get_text(separator="\n", strip=True)[:max_chars]

            # Suspiciously short? Might be a JS-only page
            if len(content) < 200 and resp.status_code == 200:
                jina_text = _jina_reader_fallback(url, max_chars)
                if jina_text and len(jina_text) > len(content):
                    return {
                        "ok": True, "content": jina_text, "url": str(resp.url),
                        "blocked": False, "reason": "", "status_code": 200,
                        "retries": attempt, "source": "jina_reader",
                    }

            return {
                "ok": True, "content": content, "url": str(resp.url),
                "blocked": False, "reason": "", "status_code": resp.status_code,
                "retries": attempt,
            }

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            retries_done = attempt
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

    # ── All direct attempts failed — try Jina Reader ──────────────────
    jina_text = _jina_reader_fallback(url, max_chars)
    if jina_text:
        return {
            "ok": True, "content": jina_text, "url": url,
            "blocked": False, "reason": "", "status_code": last_status,
            "retries": retries_done, "source": "jina_reader",
        }

    # ── Total failure ─────────────────────────────────────────────────
    return {
        "ok": False, "content": "", "url": url,
        "blocked": True, "reason": last_error, "status_code": last_status,
        "retries": retries_done,
        "hint": (
            "This site blocked automated access. Do NOT retry this URL. "
            "Search for the same information from a different source, "
            "or try site:web.archive.org in your search query."
        ),
    }

def read_pdf(url: str, max_chars: int = 8000) -> str:
    """Download and extract text from a PDF at a URL."""
    import httpx, io
    import pypdf
    resp = httpx.get(url, timeout=30)
    # Suppress noisy pypdf warnings ("Ignoring wrong pointing object")
    pdf_logger = logging.getLogger("pypdf")
    prev_level = pdf_logger.level
    try:
        pdf_logger.setLevel(logging.ERROR)
        reader = pypdf.PdfReader(io.BytesIO(resp.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        pdf_logger.setLevel(prev_level)
    return text[:max_chars]


# ── extract_tables ────────────────────────────────────────────────────
def extract_tables(url: str, max_chars: int = 12000, css_selector: str = "",
                   timeout: int = 20) -> dict:
    """Extract HTML tables from a URL as structured JSON arrays.

    Returns a dict with 'ok', 'tables' (list of list-of-dicts), 'url', etc.
    Each table is a list of row-dicts keyed by header names. If the table
    has no <thead>/<th>, column names default to "col_0", "col_1", …

    Args:
        url: URL to fetch.
        max_chars: Soft cap on total JSON output size.
        css_selector: Optional CSS selector to target specific table(s).
        timeout: Per-request timeout in seconds.
    """
    import httpx
    from bs4 import BeautifulSoup
    import warnings

    try:
        resp = httpx.get(
            url, headers=_BROWSER_HEADERS, cookies=_get_cookie_jar(),
            timeout=timeout, follow_redirects=True,
        )
        _get_cookie_jar().update(resp.cookies)

        if resp.status_code >= 400:
            return {"ok": False, "tables": [], "url": str(resp.url),
                    "reason": f"HTTP {resp.status_code}"}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*XMLParsedAsHTMLWarning.*")
            soup = BeautifulSoup(resp.text, "html.parser")

        if css_selector:
            containers = soup.select(css_selector)
            tables_html = []
            for c in containers:
                if c.name == "table":
                    tables_html.append(c)
                else:
                    tables_html.extend(c.find_all("table"))
        else:
            tables_html = soup.find_all("table")

        if not tables_html:
            return {"ok": False, "tables": [], "url": str(resp.url),
                    "reason": "No <table> elements found on page",
                    "hint": "Try fetch_url with extract='text' instead."}

        all_tables = []
        total_chars = 0
        for tbl in tables_html:
            # Determine headers
            headers = []
            thead = tbl.find("thead")
            if thead:
                headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
            if not headers:
                first_row = tbl.find("tr")
                if first_row and first_row.find("th"):
                    headers = [th.get_text(strip=True) for th in first_row.find_all("th")]

            rows_data = []
            body_rows = tbl.find_all("tr")
            start_idx = 1 if headers else 0  # skip header row
            for tr in body_rows[start_idx:]:
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if not cells or all(c == "" for c in cells):
                    continue
                if headers:
                    row_dict = {}
                    for i, h in enumerate(headers):
                        row_dict[h or f"col_{i}"] = cells[i] if i < len(cells) else ""
                    for i in range(len(headers), len(cells)):
                        row_dict[f"col_{i}"] = cells[i]
                    rows_data.append(row_dict)
                else:
                    row_dict = {f"col_{i}": c for i, c in enumerate(cells)}
                    rows_data.append(row_dict)

            if rows_data:
                chunk = json.dumps(rows_data, ensure_ascii=False)
                total_chars += len(chunk)
                all_tables.append(rows_data)
                if total_chars > max_chars:
                    break

        return {
            "ok": True,
            "tables": all_tables,
            "table_count": len(all_tables),
            "url": str(resp.url),
        }

    except Exception as e:
        return {"ok": False, "tables": [], "url": url,
                "reason": f"{type(e).__name__}: {e}"}


# ── wikipedia_lookup ──────────────────────────────────────────────────
def wikipedia_lookup(title: str, section: str = "", max_chars: int = 8000) -> dict:
    """Look up a Wikipedia article via the MediaWiki API.

    Returns parsed wikitext for the whole article or a specific section.
    Uses the REST API to get clean HTML, then extracts text. Also pulls
    the infobox as structured key-value pairs when available.

    Args:
        title: Wikipedia article title (e.g. "Blue-Eyes White Dragon").
        section: Optional section heading to extract (case-insensitive substring match).
        max_chars: Maximum characters of text to return.
    """
    import httpx
    from bs4 import BeautifulSoup

    try:
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "parse",
            "page": title,
            "prop": "text|sections",
            "format": "json",
            "redirects": 1,
            "disabletoc": 1,
        }
        wiki_headers = {
            "User-Agent": "TrajectoryKit/1.0 (research agent; https://github.com/KabakaWilliam/trajectorykit)",
            "Accept": "application/json",
        }
        resp = httpx.get(api_url, params=params, headers=wiki_headers, timeout=15, follow_redirects=True)
        data = resp.json()

        if "error" in data:
            return {"ok": False, "content": "", "title": title,
                    "reason": data["error"].get("info", "Article not found"),
                    "hint": "Check spelling or try searching with search_web."}

        html = data["parse"]["text"]["*"]
        actual_title = data["parse"].get("title", title)
        sections_list = [s["line"] for s in data["parse"].get("sections", [])]

        soup = BeautifulSoup(html, "html.parser")

        # Remove navboxes, metadata, edit links, etc.
        for tag in soup.select(".navbox, .metadata, .mw-editsection, .reference, "
                                ".reflist, .sistersitebox, .noprint, style, script"):
            tag.decompose()

        # Extract infobox as structured data
        infobox = {}
        infobox_el = soup.select_one(".infobox")
        if infobox_el:
            for row in infobox_el.find_all("tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    key = th.get_text(strip=True)
                    val = td.get_text(separator=" ", strip=True)
                    if key and val:
                        infobox[key] = val

        # Section targeting
        if section:
            section_lower = section.lower()
            target = None
            for heading in soup.find_all(["h2", "h3", "h4"]):
                heading_text = heading.get_text(strip=True)
                if section_lower in heading_text.lower():
                    target = heading
                    break

            if target:
                tag_name = target.name
                # Modern MediaWiki wraps headings in <div class="mw-heading">.
                # Content paragraphs are siblings of that wrapper div.
                anchor = target.parent if target.parent and "mw-heading" in " ".join(target.parent.get("class", [])) else target

                content_parts = []
                for sibling in anchor.find_next_siblings():
                    if sibling.name in ["h2", "h3", "h4"] and sibling.name <= tag_name:
                        break
                    if sibling.name == "div" and "mw-heading" in " ".join(sibling.get("class", [])):
                        inner = sibling.find(["h2", "h3", "h4"])
                        if inner and inner.name <= tag_name:
                            break
                    text = sibling.get_text(separator=" ", strip=True)
                    if text:
                        content_parts.append(text)
                content = "\n\n".join(content_parts)[:max_chars]
            else:
                content = f"Section '{section}' not found. Available sections: {', '.join(sections_list)}"
        else:
            content = soup.get_text(separator="\n", strip=True)[:max_chars]

        result = {
            "ok": True,
            "title": actual_title,
            "content": content,
            "sections": sections_list,
        }
        if infobox:
            result["infobox"] = infobox
        return result

    except Exception as e:
        return {"ok": False, "content": "", "title": title,
                "reason": f"{type(e).__name__}: {e}"}


# ── fetch_cached (Wayback Machine) ───────────────────────────────────
def fetch_cached(url: str, date: str = "", max_chars: int = 8000) -> dict:
    """Fetch a cached version of a URL from the Wayback Machine.

    Args:
        url: Original URL to look up in the Wayback Machine.
        date: Optional target date as YYYYMMDD. If empty, returns most recent snapshot.
        max_chars: Maximum characters of extracted text to return.
    """
    import httpx
    from bs4 import BeautifulSoup
    import warnings

    try:
        avail_url = "https://archive.org/wayback/available"
        params = {"url": url}
        if date:
            params["timestamp"] = date
        resp = httpx.get(avail_url, params=params, timeout=15)
        data = resp.json()

        snapshots = data.get("archived_snapshots", {})
        closest = snapshots.get("closest")
        if not closest or not closest.get("available"):
            return {"ok": False, "content": "", "url": url,
                    "reason": "No Wayback Machine snapshot found for this URL",
                    "hint": "Try a different URL or date."}

        archive_url = closest["url"]
        snapshot_date = closest.get("timestamp", "")

        resp2 = httpx.get(
            archive_url, headers=_BROWSER_HEADERS,
            timeout=20, follow_redirects=True,
        )
        if resp2.status_code >= 400:
            return {"ok": False, "content": "", "url": archive_url,
                    "reason": f"HTTP {resp2.status_code} fetching archived page"}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*XMLParsedAsHTMLWarning.*")
            soup = BeautifulSoup(resp2.text, "html.parser")

        for wb_el in soup.select("#wm-ipp-base, #wm-ipp, #donato, .wb-autocomplete-suggestions"):
            wb_el.decompose()
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        content = soup.get_text(separator="\n", strip=True)[:max_chars]

        return {
            "ok": True,
            "content": content,
            "url": archive_url,
            "original_url": url,
            "snapshot_date": snapshot_date,
        }

    except Exception as e:
        return {"ok": False, "content": "", "url": url,
                "reason": f"{type(e).__name__}: {e}"}


def spawn_agent(
    task: str,
    context: Optional[str] = None,
    turn_length: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    _depth: int = 0,
    _sandbox_files: Optional[dict] = None,
) -> ToolReturn:
    """
    Spawn a sub-agent to handle a subtask. Calls dispatch() recursively.
    Returns the sub-agent's final response as a string.
    """
    from .agent import dispatch   # lazy import to avoid circular dependency
    from .config import MAX_RECURSION_DEPTH, get_model_profile, MODEL_NAME

    resolved_model = model or MODEL_NAME
    profile = get_model_profile(resolved_model)

    if turn_length is None:
        turn_length = SUB_AGENT_TURN_BUDGET
    if max_tokens is None:
        max_tokens = profile["context_window"]

    if _depth >= MAX_RECURSION_DEPTH:
        return json.dumps({
            "response": f"[RECURSION LIMIT] Depth {_depth} reached (max {MAX_RECURSION_DEPTH}). Cannot spawn further sub-agents.",
            "turns_used": 0,
            "tool_calls_made": 0,
            "depth": _depth + 1,
        }), None

    full_input = f"{context}\n\nTASK: {task}" if context else task

    if _depth > 0:
        logger.info(f"Spawning sub-agent at depth {_depth}: {task[:80]}...")

    result = dispatch(
        user_input=full_input,
        turn_length=turn_length,
        verbose=False,           # sub-agents run silently
        max_tokens=profile["context_window"],
        temperature=temperature,
        model=model,
        reasoning_effort=reasoning_effort,
        _depth=_depth + 1,       # increment depth for recursive calls
        _sandbox_files=_sandbox_files,
    )

    # Extract child trace from the result
    child_trace = result.get("trace", None)

    output = json.dumps({
        "response": result["final_response"],
        "turns_used": result["turns"],
        "tool_calls_made": result["tool_calls"],
        "depth": _depth + 1,
    })

    return output, child_trace

def execute_code(
    code: Optional[str] = None,
    completion: Optional[str] = None,   # legacy alias for 'code'
    stdin: Optional[str] = '',
    compile_timeout: int = 10,
    run_timeout: int = 15,
    memory_limit_mb: int = 512,
    language: str = "python",
    files: Optional[dict[str, str]] = None,      # filename -> base64 content
    fetch_files: Optional[list[str]] = None,      # list of filenames to return
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    
    from .config import SANDBOX_FUSION_URL

    # Accept both 'code' and legacy 'completion' param names
    raw = code or completion
    if not raw:
        return None, "ERROR: 'code' parameter is required"

    extracted = raw
    if "```python" in raw:
        extracted = raw.split("```python")[-1].split("```")[0]
    elif "```" in raw:
        # Handle cases like ```\ncode\n```
        parts = raw.split("```")
        if len(parts) >= 2:
            extracted = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in extracted:
                first_line, rest = extracted.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    extracted = rest
    # If no code block markers, use raw input as-is (don't error)
    code = extracted
    
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,  # Use the passed language parameter
            "files": files or {},
            "fetch_files": fetch_files or [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None  # Store the last error encountered

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {SANDBOX_FUSION_URL}"
            )  # <-- Use internal log_prefix
            response = requests.post(
                SANDBOX_FUSION_URL,
                headers=headers,
                data=payload,
                timeout=request_timeout,  # Use the calculated timeout
            )

            # Check for Gateway Timeout (504) specifically for retrying
            if response.status_code == 504:
                last_error = (
                    f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )  # <-- Use internal log_prefix
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    # Calculate increasing delay (e.g., 1s, 2s, 4s, ...) or (1s, 2s, 3s, ...)
                    # Simple linear increase: delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    # Exponential backoff: delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)  # Using linear increase for simplicity
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")  # <-- Use internal log_prefix
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(
                f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}"
            )  # <-- Use internal log_prefix
            return response.json(), None

        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"

def execute_code_wrapper(**kwargs):
    """Wrapper to call execute_code and format output for vLLM.
    Returns (output_str, None) — no child trace for code execution."""
    try:
        # execute_code returns a tuple: (result_dict, error_string)
        sandbox_result, error = execute_code(**kwargs)
        
        # If there was an error
        if error:
            return f"ERROR: {error}", None
        
        # If no result
        if sandbox_result is None:
            return "ERROR: No result returned from sandbox", None
        
        # Parse the sandbox result
        if isinstance(sandbox_result, dict):
            status = sandbox_result.get("status", "Unknown")
            
            # Check if execution succeeded
            if status == "Success":
                run_result = sandbox_result.get("run_result", {})
                return_code = run_result.get("return_code", -1)
                stdout = run_result.get("stdout", "")
                stderr = run_result.get("stderr", "")
                exec_time = run_result.get("execution_time", 0)
                
                # Format output for vLLM
                output = f"Exit Code: {return_code}\n"
                output += f"Execution Time: {exec_time:.3f}s\n"
                
                if stdout:
                    output += f"STDOUT:\n{stdout}"
                if stderr:
                    output += f"STDERR:\n{stderr}"
                
                # Include any fetched files (base64-encoded) from the sandbox
                fetched = sandbox_result.get("files", {})
                if fetched:
                    output += "\nFETCHED FILES:\n"
                    for fname, b64data in fetched.items():
                        output += f"--- {fname} (base64) ---\n{b64data}\n"
                
                # Warn if files were requested but none came back
                requested_files = kwargs.get("fetch_files", [])
                if requested_files and not fetched:
                    output += (
                        f"\nWARNING: You requested fetch_files={requested_files} but no files were returned. "
                        "Make sure your code writes these files to the working directory "
                        "(e.g., use plt.savefig('output.png') instead of plt.show())."
                    )
                
                result = output if output.strip() else "Code executed successfully with no output"
                return result, None
            else:
                # Execution failed — gather as much detail as possible
                message = sandbox_result.get("message", "")
                run_result = sandbox_result.get("run_result", {}) or {}
                run_status = run_result.get("status", "")
                stderr = run_result.get("stderr", "")
                stdout = run_result.get("stdout", "")
                return_code = run_result.get("return_code")
                exec_time = run_result.get("execution_time", 0)
                
                # Detect specific sandbox failure modes from run_result.status
                if run_status == "TimeLimitExceeded":
                    run_timeout = kwargs.get("run_timeout", 5)
                    error_parts = [
                        f"ERROR: Time limit exceeded — your code ran for {exec_time:.1f}s and was killed (limit: {run_timeout}s).",
                        "Your code took too long to execute. Common causes:",
                        "- Infinite loops (while True) or very long loops",
                        "- Blocking calls (time.sleep, input()) or network waits",
                        "- Computationally expensive operations",
                        "FIX: Rewrite your code to complete within the time limit. Do NOT use infinite loops in the sandbox.",
                    ]
                    if stdout:
                        error_parts.append(f"Partial STDOUT before kill:\n{stdout}")
                elif run_status == "MemoryLimitExceeded":
                    memory_limit = kwargs.get("memory_limit_mb", 128)
                    error_parts = [
                        f"ERROR: Memory limit exceeded (limit: {memory_limit}MB).",
                        "Your code used too much memory. Reduce data size or optimize memory usage.",
                    ]
                else:
                    error_parts = [f"ERROR: Execution failed (status: {status})"]
                    if run_status:
                        error_parts.append(f"Run status: {run_status}")
                    if message:
                        error_parts.append(f"Message: {message}")
                    if return_code is not None:
                        error_parts.append(f"Return code: {return_code}")
                    if stderr:
                        error_parts.append(f"STDERR:\n{stderr}")
                    if stdout:
                        error_parts.append(f"STDOUT:\n{stdout}")
                    if not message and not stderr and not run_status:
                        error_parts.append("No error details provided by sandbox. Check your code for syntax errors or missing imports.")
                
                return "\n".join(error_parts), None
        
        # Fallback
        return f"Unexpected result format: {str(sandbox_result)[:200]}", None
        
    except Exception as e:
        return f"ERROR: {str(e)}", None

def search_web_wrapper(**kwargs):
    """Wrapper for search_web tool. Returns (output, None)."""
    try:
        q = kwargs.get("q")
        num_results = kwargs.get("num_results", DEFAULT_NUM_SEARCHES)

        return search_web(q=q, num_results=num_results), None
    except Exception as e:
        return f"ERROR: {str(e)}", None

def spawn_agent_wrapper(_depth: int = 0, _model: Optional[str] = None, _reasoning_effort: Optional[str] = None, _sandbox_files: Optional[dict] = None, _memory_store=None, **kwargs):
    """Wrapper for spawn_agent tool. Injects _depth, _model, _reasoning_effort from the parent dispatch loop.
    Returns (output_str, child_trace) where child_trace is an EpisodeTrace.
    
    If the model provides memory_keys, the corresponding MemoryStore entries
    are serialized to agent_data.json and pre-loaded into the sub-agent's sandbox.
    """
    try:
        task = kwargs.get("task")
        if not task:
            return "ERROR: 'task' parameter is required", None

        # Build sandbox files for this sub-agent
        # Start with any inherited sandbox files (e.g. from synthesis pipeline)
        child_sandbox_files = dict(_sandbox_files) if _sandbox_files else {}

        # If memory_keys provided, serialize those entries to agent_data.json
        memory_keys = kwargs.pop("memory_keys", None)
        if memory_keys and _memory_store:
            import base64 as _b64
            entries = []
            missing = []
            for k in memory_keys:
                content = _memory_store.get(k)
                if content is not None:
                    entries.append({"key": k, "content": content, "content_length": len(content)})
                else:
                    missing.append(k)
            if entries:
                data_json = json.dumps({"entries": entries, "entry_count": len(entries)}, ensure_ascii=False)
                child_sandbox_files["agent_data.json"] = _b64.b64encode(data_json.encode("utf-8")).decode("ascii")
                # Prepend a note to the task so the worker knows about the file
                file_summary = ", ".join(f"{e['key']} ({e['content_length']:,} chars)" for e in entries)
                task = (
                    f"[PRE-LOADED DATA] The file agent_data.json is available in your sandbox. "
                    f"Read it with: import json; data = json.load(open('agent_data.json'))\n"
                    f"It contains {len(entries)} entries: {file_summary}\n"
                    f"Each entry has 'key', 'content', and 'content_length' fields.\n\n"
                    f"{task}"
                )
            if missing:
                task += f"\n\n(Note: requested memory keys not found: {', '.join(missing)})"
        elif memory_keys and not _memory_store:
            task += "\n\n(Note: memory_keys were requested but memory store is not available at this depth.)"

        from .config import SUB_AGENT_TURN_BUDGET
        output, child_trace = spawn_agent(
            task=task,
            context=kwargs.get("context"),
            turn_length=kwargs.get("turn_length", SUB_AGENT_TURN_BUDGET),
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            model=_model,
            reasoning_effort=_reasoning_effort,
            _depth=_depth,
            _sandbox_files=child_sandbox_files if child_sandbox_files else None,
        )
        return output, child_trace
    except Exception as e:
        return f"ERROR: {str(e)}", None

def final_answer_wrapper(**kwargs):
    """Wrapper for final_answer tool. Returns the answer text directly.
    The agent loop treats this tool call as the termination signal."""
    answer = kwargs.get("answer", "")
    return answer or "", None


def search_available_tools_wrapper(**kwargs):
    """Wrapper for search_available_tools. Returns (output, None)."""
    try:
        tool_name = kwargs.get("tool_name")
        if tool_name:
            # Return full schema for the requested tool
            for tool in TOOLS:
                if tool["function"]["name"] == tool_name:
                    return json.dumps(tool, indent=2), None
            return f"ERROR: No tool named '{tool_name}'. Call search_available_tools with no arguments to see all available tools.", None
        else:
            # Return compact signature + one-liner for each tool
            summary = []
            for tool in TOOLS:
                func = tool["function"]
                name = func["name"]
                # Build a compact signature: name(param: type, param?: type=default, ...)
                params = func.get("parameters", {}).get("properties", {})
                required = set(func.get("parameters", {}).get("required", []))
                parts = []
                for pname, pschema in params.items():
                    ptype = pschema.get("type", "any")
                    default = pschema.get("default")
                    if pname in required:
                        parts.append(f"{pname}: {ptype}")
                    elif default is not None:
                        parts.append(f"{pname}?: {ptype}={default}")
                    else:
                        parts.append(f"{pname}?: {ptype}")
                sig = ", ".join(parts)
                # First sentence of description (split on '. ' to get a real sentence)
                raw_desc = func["description"].replace('\n', ' ')
                first_sentence = raw_desc.split('. ')[0].strip().rstrip('.')
                summary.append(f"- {name}({sig})\n  {first_sentence}.")
            return "Available tools:\n\n" + "\n\n".join(summary), None
    except Exception as e:
        return f"ERROR: {str(e)}", None
    
def fetch_url_wrapper(**kwargs):
    """Wrapper for fetch_url tool. Returns (output, None).
    
    Auto-redirects Wikipedia URLs to wikipedia_lookup for reliable access
    (Wikipedia blocks raw HTTP requests without proper API credentials).
    """
    try:
        url = kwargs.get("url")
        if not url:
            return "ERROR: 'url' parameter is required", None

        # ── A: Auto-redirect Wikipedia URLs → wikipedia_lookup ────────
        import re as _re
        wiki_match = _re.match(
            r'https?://(?:\w+\.)?wikipedia\.org/wiki/(.+)',
            url, _re.IGNORECASE,
        )
        if wiki_match:
            # Extract title from URL path, decode percent-encoding
            from urllib.parse import unquote
            raw_title = unquote(wiki_match.group(1)).replace('_', ' ')
            # Strip any section anchor
            if '#' in raw_title:
                raw_title = raw_title.split('#')[0]
            logger.info(f"[fetch_url] Wikipedia URL detected — redirecting to wikipedia_lookup(title='{raw_title}')")
            result_str, _ = wikipedia_lookup_wrapper(title=raw_title, max_chars=kwargs.get("max_chars", 8000))
            return f"[Auto-redirected to wikipedia_lookup for reliable Wikipedia access]\n\n{result_str}", None

        max_chars = kwargs.get("max_chars", 8000)
        extract = kwargs.get("extract", "text")
        css_selector = kwargs.get("css_selector", "")
        result = fetch_url(url=url, max_chars=max_chars, extract=extract, css_selector=css_selector)

        if result["ok"]:
            output = result["content"]
            if result["retries"] > 0:
                output = f"[Succeeded after {result['retries']} retry(ies), final URL: {result['url']}]\n\n{output}"
            if result.get("source") == "jina_reader":
                output = f"[Rendered via Jina Reader]\n\n{output}"
            return output, None
        else:
            # ── B: Add Wikipedia hint on 403 from Wikipedia domains ───
            if result.get("status_code") == 403 and "wikipedia" in url.lower():
                result["hint"] = (
                    "Wikipedia blocked this request (403 Forbidden). "
                    "Use wikipedia_lookup(title='Article Title') instead — "
                    "it uses the MediaWiki API which is not blocked."
                )
            return json.dumps(result, indent=2), None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def read_pdf_wrapper(**kwargs):
    """Wrapper for read_pdf tool. Returns (output, None)."""
    try:
        url = kwargs.get("url")
        if not url:
            return "ERROR: 'url' parameter is required", None

        max_chars = kwargs.get("max_chars", 8000)
        text = read_pdf(url=url, max_chars=max_chars)
        return text, None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def extract_tables_wrapper(**kwargs):
    """Wrapper for extract_tables tool. Returns (output, None)."""
    try:
        url = kwargs.get("url")
        if not url:
            return "ERROR: 'url' parameter is required", None

        max_chars = kwargs.get("max_chars", 12000)
        css_selector = kwargs.get("css_selector", "")
        result = extract_tables(url=url, max_chars=max_chars, css_selector=css_selector)

        if result["ok"]:
            output = json.dumps({
                "table_count": result["table_count"],
                "url": result["url"],
                "tables": result["tables"],
            }, ensure_ascii=False, indent=2)
            return output, None
        else:
            return json.dumps(result, indent=2), None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def wikipedia_lookup_wrapper(**kwargs):
    """Wrapper for wikipedia_lookup tool. Returns (output, None)."""
    try:
        title = kwargs.get("title")
        if not title:
            return "ERROR: 'title' parameter is required", None

        section = kwargs.get("section", "")
        max_chars = kwargs.get("max_chars", 8000)
        result = wikipedia_lookup(title=title, section=section, max_chars=max_chars)

        if result["ok"]:
            parts = [f"Wikipedia: {result['title']}"]
            if result.get("infobox"):
                parts.append("\n[Infobox]")
                for k, v in result["infobox"].items():
                    parts.append(f"  {k}: {v}")
            parts.append(f"\n{result['content']}")
            if result.get("sections"):
                parts.append(f"\n[Sections: {', '.join(result['sections'][:20])}]")
            return "\n".join(parts), None
        else:
            return json.dumps(result, indent=2), None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def fetch_cached_wrapper(**kwargs):
    """Wrapper for fetch_cached tool. Returns (output, None)."""
    try:
        url = kwargs.get("url")
        if not url:
            return "ERROR: 'url' parameter is required", None

        date = kwargs.get("date", "")
        max_chars = kwargs.get("max_chars", 8000)
        result = fetch_cached(url=url, date=date, max_chars=max_chars)

        if result["ok"]:
            header = f"[Wayback Machine snapshot: {result.get('snapshot_date', '?')}]\n"
            header += f"[Original URL: {result.get('original_url', url)}]\n\n"
            return header + result["content"], None
        else:
            return json.dumps(result, indent=2), None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def recall_memory_wrapper(_memory_store=None, **kwargs):
    """Retrieve a stored tool output from MemoryStore by key. Returns (output, None).
    
    - No key: list all stored keys with sizes
    - Key only: return first 2000 chars of stored content
    - Key + query: return matching lines (case-insensitive grep), capped at 2000 chars
    """
    MAX_RETURN_CHARS = 2000

    try:
        if _memory_store is None:
            return "ERROR: Memory store is not available (recall_memory is only usable at the root orchestrator level).", None

        key = kwargs.get("key")
        query = kwargs.get("query", "")

        # No key → list all stored entries
        if not key:
            if len(_memory_store) == 0:
                return "Memory is empty — no tool outputs have been stored yet.", None
            return _memory_store.summary(), None

        # Retrieve by key
        content = _memory_store.get(key)
        if content is None:
            available = ", ".join(_memory_store.keys()[:15])
            return f"ERROR: No entry with key '{key}'. Available keys: {available}", None

        # Query filter: grep matching lines
        if query:
            q_lower = query.lower()
            matching = [line for line in content.split("\n") if q_lower in line.lower()]
            if not matching:
                return f"No lines matching '{query}' in {key} ({len(content):,} chars total).", None
            filtered = "\n".join(matching)
            if len(filtered) > MAX_RETURN_CHARS:
                filtered = filtered[:MAX_RETURN_CHARS] + f"\n... (truncated, {len(matching)} matching lines total)"
            return f"[{key}] Lines matching '{query}' ({len(matching)} hits):\n{filtered}", None

        # Return content capped at limit
        if len(content) <= MAX_RETURN_CHARS:
            return f"[{key}] ({len(content):,} chars):\n{content}", None
        else:
            truncated = content[:MAX_RETURN_CHARS]
            # Trim to last newline to avoid mid-line cut
            last_nl = truncated.rfind("\n")
            if last_nl > MAX_RETURN_CHARS // 2:
                truncated = truncated[:last_nl]
            return (
                f"[{key}] (showing first {len(truncated):,} of {len(content):,} chars — "
                f"use query= to search within, or spawn_agent for full analysis):\n{truncated}"
            ), None

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def dispatch_tool_call(tool_name: str, tool_args: dict, _depth: int = 0, model: Optional[str] = None, reasoning_effort: Optional[str] = None, _sandbox_files: Optional[dict] = None, _memory_store=None):
    """Route tool calls to appropriate wrapper function.
    
    All wrappers return (output_str, child_trace_or_None).
    
    Args:
        tool_name: Name of the tool to call
        tool_args: Arguments the model provided for the tool
        _depth: Current recursion depth (injected by the agent loop, invisible to the model)
        model: Model name to propagate to sub-agents
        reasoning_effort: Reasoning effort level to propagate to sub-agents
        _sandbox_files: Files to auto-inject into every execute_code sandbox call.
                        Dict of {filename: base64_content}. Merged with model-provided files.
        _memory_store: MemoryStore instance for recall_memory tool (root only).
    
    Returns:
        Tuple of (output_string, child_trace) where child_trace is an
        EpisodeTrace if the tool was spawn_agent, otherwise None.
    """
    if tool_name == "execute_code":
        # Merge framework-injected sandbox files with any model-provided files
        if _sandbox_files:
            model_files = tool_args.get("files") or {}
            merged = {**_sandbox_files, **model_files}  # model files take precedence
            tool_args = {**tool_args, "files": merged}
        return execute_code_wrapper(**tool_args)
    elif tool_name == "search_web":
        return search_web_wrapper(**tool_args)
    elif tool_name == "spawn_agent":
        return spawn_agent_wrapper(_depth=_depth, _model=model, _reasoning_effort=reasoning_effort, _sandbox_files=_sandbox_files, _memory_store=_memory_store, **tool_args)
    elif tool_name == "final_answer":
        return final_answer_wrapper(**tool_args)
    elif tool_name == "search_available_tools":
        return search_available_tools_wrapper(**tool_args)
    elif tool_name == "fetch_url":
        return fetch_url_wrapper(**tool_args)
    elif tool_name == "read_pdf":
        return read_pdf_wrapper(**tool_args)
    elif tool_name == "extract_tables":
        return extract_tables_wrapper(**tool_args)
    elif tool_name == "wikipedia_lookup":
        return wikipedia_lookup_wrapper(**tool_args)
    elif tool_name == "fetch_cached":
        return fetch_cached_wrapper(**tool_args)
    elif tool_name == "recall_memory":
        return recall_memory_wrapper(_memory_store=_memory_store, **tool_args)
    else:
        return f"ERROR: Unknown tool '{tool_name}'", None


# Define the tool schemas for vLLM
TOOLS = [
{
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute code in a sandboxed environment with support for multiple languages, "
            "input/output handling, file I/O, and resource limits. "
            "The code should be provided in a markdown code block (e.g., ```python code here ```). "
            "Returns execution results including stdout, stderr, exit status, and any requested output files.\n\n"
            "FILE HANDLING:\n"
            "- `files`: Upload input files (e.g. CSVs, images, data) into the sandbox before execution. "
            "Provide as a dict of {filename: base64_encoded_string}. "
            "Files are placed in the working directory and can be read by name in your code.\n"
            "- `fetch_files`: Retrieve output files generated by your code after execution (e.g. plots, reports, ZIPs). "
            "Provide a list of filenames your code will write. "
            "They are returned as base64-encoded strings in the response under the 'files' key.\n\n"
            "EXAMPLE WORKFLOW: To generate a chart → write ```python ... plt.savefig('plot.png') ``` "
            "and set fetch_files=['plot.png']. The plot will be returned as base64 for display."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Code to execute. Wrap in markdown code blocks "
                        "(```python ... ``` or ``` ... ```) or provide raw code. "
                        "Code blocks are automatically extracted."
                    )
                },
                "stdin": {
                    "type": "string",
                    "description": "Standard input to pipe into the program (default: '').",
                    "default": ""
                },
                "language": {
                    "type": "string",
                    "description": "Programming language to use (default: 'python').",
                    "default": "python",
                    "enum": SUPPORTED_LANGUAGES
                },
                "files": {
                    "type": "object",
                    "description": (
                        "Input files to upload into the sandbox before execution. "
                        "Keys are filenames (e.g. 'data.csv'), values are base64-encoded file contents. "
                        "Files are written to the working directory and accessible by name in your code."
                    ),
                    "additionalProperties": {
                        "type": "string",
                        "description": "Base64-encoded file content."
                    },
                    "default": {}
                },
                "fetch_files": {
                    "type": "array",
                    "description": (
                        "List of filenames to retrieve from the sandbox after execution. "
                        "Your code must write these files to the working directory. "
                        "They are returned as base64-encoded strings in the response under the 'files' key. "
                        "Use this to retrieve generated plots, CSVs, ZIPs, PDFs, etc."
                    ),
                    "items": {
                        "type": "string",
                        "description": "Filename to fetch (e.g. 'output.png', 'result.csv')."
                    },
                    "default": []
                },
                "compile_timeout": {
                    "type": "integer",
                    "description": "Compilation timeout in seconds (default: 10).",
                    "default": 10
                },
                "run_timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default: 15).",
                    "default": 15
                },
                "memory_limit_mb": {
                    "type": "integer",
                    "description": "Memory limit in megabytes (default: 512).",
                    "default": 512
                }
            },
            "required": ["code"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch a web page with automatic retry, browser-grade headers, and "
            "persistent cookies. Falls back to Jina Reader for JS-heavy pages.\n\n"
            "Returns page text on success. On failure (site blocks, 403, etc.) returns "
            "a structured error with blocked=true and a hint to search elsewhere.\n"
            "DO NOT retry a URL that returns blocked=true — pivot to a different source.\n\n"
            "TIP: Use css_selector to extract only the relevant part of a large page "
            "(e.g. css_selector='table.data-table' or css_selector='div#content')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http/https)."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters of extracted text to return (default: 8000).",
                    "default": 8000
                },
                "extract": {
                    "type": "string",
                    "description": "What to extract: 'text' for page text (default), 'table' for HTML tables as pipe-delimited rows.",
                    "enum": ["text", "table"],
                    "default": "text"
                },
                "css_selector": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector to extract only matching elements "
                        "(e.g. 'table.wikitable', 'div#mw-content-text', 'article'). "
                        "Reduces noise on large pages. Omit to get the full page."
                    ),
                    "default": ""
                }
            },
            "required": ["url"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "extract_tables",
        "description": (
            "Extract HTML tables from a URL as structured JSON (list of row-dicts). "
            "Much better than fetch_url(extract='table') for data analysis — returns "
            "proper column names and cell values you can directly process in code.\n\n"
            "Use this when the answer requires reading specific values from tables "
            "(statistics, rankings, comparisons). Feed the JSON result into "
            "execute_code for filtering, sorting, or computation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the page containing HTML tables."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Soft cap on total JSON output size (default: 12000).",
                    "default": 12000
                },
                "css_selector": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector to target specific table(s) "
                        "(e.g. 'table.wikitable', 'table#stats'). "
                        "Omit to extract all tables on the page."
                    ),
                    "default": ""
                }
            },
            "required": ["url"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "read_pdf",
        "description": (
            "Download a PDF from a URL and extract its text content. "
            "Use this when a search result is a PDF (papers, reports)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Direct URL to a PDF file."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of characters of extracted text to return (default: 8000).",
                    "default": 8000
                }
            },
            "required": ["url"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "wikipedia_lookup",
        "description": (
            "Look up a Wikipedia article directly via the MediaWiki API. "
            "Faster and more reliable than search→fetch for Wikipedia content.\n\n"
            "Returns: article text, section list, and infobox data (structured key-value pairs). "
            "Use the 'section' parameter to get only a specific section.\n\n"
            "PREFER THIS over fetch_url when the question references Wikipedia or "
            "when you need structured facts (dates, stats, lists) from a well-known topic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Wikipedia article title (e.g. 'Blue-Eyes White Dragon', 'Python (programming language)')."
                },
                "section": {
                    "type": "string",
                    "description": (
                        "Optional section heading to extract (case-insensitive substring match). "
                        "Omit to get the full article. Example: 'History', 'Demographics'."
                    ),
                    "default": ""
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters of text to return (default: 8000).",
                    "default": 8000
                }
            },
            "required": ["title"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "fetch_cached",
        "description": (
            "Fetch an archived version of a URL from the Wayback Machine.\n\n"
            "Use this when:\n"
            "- A site blocks automated access (403, Cloudflare) but you need its content\n"
            "- You need historical page content from a specific date\n"
            "- The original page has been taken down or modified\n\n"
            "Returns the page text from the closest available snapshot."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Original URL to look up in the Wayback Machine."
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Target date as YYYYMMDD (e.g. '20231015'). Returns the closest "
                        "snapshot to this date. Omit for the most recent snapshot."
                    ),
                    "default": ""
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters of extracted text to return (default: 8000).",
                    "default": 8000
                }
            },
            "required": ["url"]
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet using Google via SerpAPI. Returns formatted search results with titles, URLs, and snippets. "
                "Use this to find current information, answer factual questions, or research topics. "
                "Automatically handles errors like rate limits, invalid keys, and timeouts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "The search query string"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10, default: 5)",
                        "default": 5
                    }
                },
                "required": ["q"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agent",
            "description": (
                "DELEGATE a subtask to an independent sub-agent. This is your PRIMARY tool for complex tasks. "
                "The sub-agent gets its own fresh context, tools, and turn budget. It does the work and returns ONLY its final result — "
                "keeping your context clean and small.\n\n"
                "YOU MUST USE THIS TOOL WHEN:\n"
                "- The user asks to compare, research, or analyze 2+ items → spawn one sub-agent PER item\n"
                "- A subtask needs search + code execution → delegate it, don't do it yourself\n"
                "- You're about to make 3+ tool calls for one part of the task → that's a sub-agent\n\n"
                "HOW TO WRITE THE TASK STRING:\n"
                "The sub-agent has NO context from your conversation. The task string is ALL it gets.\n"
                "- BAD: 'Research card A' (too vague)\n"
                "- GOOD: 'Find the ATK, DEF, Level, Type, and Attribute of the Yu-Gi-Oh card Blue-Eyes White Dragon. "
                "Search the web for accurate stats. Return the results as a JSON object with keys: name, atk, def, level, type, attribute.'\n\n"
                "PASSING DATA VIA memory_keys:\n"
                "When tool outputs are stored in memory (you see [Stored → key_name]), you can pass that data "
                "to a sub-agent WITHOUT pasting it. Use memory_keys=['key1', 'key2'] and the data will be "
                "pre-loaded as agent_data.json in the sub-agent's sandbox. The sub-agent reads it via execute_code.\n"
                "This keeps YOUR context clean — the data goes straight to the sub-agent's sandbox.\n\n"
                "PATTERN FOR MULTI-ITEM TASKS:\n"
                "1. spawn_agent(task='[detailed task for item 1, specify return format]')\n"
                "2. spawn_agent(task='[detailed task for item 2, specify return format]')\n"
                "3. Collect results, then use execute_code yourself to visualize/synthesize"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear description of the subtask the sub-agent should accomplish"
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional background context or data the sub-agent needs. "
                            "Include any relevant information from the current conversation."
                        )
                    },
                    "memory_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Memory keys to pre-load into the sub-agent's sandbox as agent_data.json. "
                            "The sub-agent can read this file via execute_code: "
                            "import json; data = json.load(open('agent_data.json')). "
                            "Use when you need the sub-agent to analyze large stored tool outputs "
                            "without pasting them into the task string. "
                            "Example: memory_keys=['search_t1_query', 'page_t3_url']"
                        )
                    },
                    "turn_length": {
                        "type": "integer",
                        "description": f"Maximum turns for the sub-agent (default: {SUB_AGENT_TURN_BUDGET})",
                        "default": SUB_AGENT_TURN_BUDGET
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": f"Maximum tokens per generation for the sub-agent (default: {CONTEXT_WINDOW})",
                        "default": CONTEXT_WINDOW
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature for the sub-agent (default: 0.7)",
                        "default": 0.7
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Submit your final answer to the user. You MUST call this tool when you are ready "
                "to deliver your response. Do NOT produce a plain text response — always use this tool. "
                "Put your complete, well-formatted answer in the 'answer' parameter. This should NOT be an empty answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete final answer to the user's question or task."
                    }
                },
                "required": ["answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Retrieve a previously stored tool output from memory by its key. "
                "When tool outputs are large, they are stored in memory and you see a "
                "compact summary with a key like [Stored → search_t1_query]. Use this "
                "tool to retrieve the full content when the summary is insufficient.\n\n"
                "Returns up to 2000 characters of the stored content. For longer content, "
                "use the optional `query` parameter to search within the stored text, or "
                "spawn a sub-agent for detailed analysis.\n\n"
                "Call with no arguments to list all available memory keys and their sizes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "The memory key to retrieve (e.g. 'search_t1_france_gdp'). "
                            "Omit to list all stored keys."
                        )
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional text to search for within the stored content. "
                            "When provided, returns only the lines containing the query "
                            "(case-insensitive), up to the character limit. "
                            "Useful for finding specific facts in large documents."
                        )
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_available_tools",
            "description": (
                "Look up the tools you have available and their parameter schemas. "
                "Call with no arguments to get a compact summary of all tools. "
                "Call with a tool_name to get the full JSON schema for that specific tool, "
                "including all parameters, types, defaults, and descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": (
                            "Name of a specific tool to get its full schema. "
                            "Omit to list all available tools with short descriptions."
                        )
                    }
                },
                "required": []
            }
        }
    }
]
