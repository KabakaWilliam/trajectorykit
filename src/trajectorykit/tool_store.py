import requests
from typing import Any, Callable, Dict, List, Optional,Tuple, Union
from .utils import SUPPORTED_LANGUAGES, API_TIMEOUT, MAX_RETRIES, INITIAL_RETRY_DELAY
from .config import MAX_RECURSION_DEPTH, SUB_AGENT_TURN_BUDGET, CONTEXT_WINDOW  # noqa: F811 — re-import ensures fresh values
from . import config as _cfg
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


def cleanup_sandbox() -> bool:
    """Send a lightweight cleanup request to the sandbox to free /tmp.

    Removes stale temp directories created by previous execute_code calls.
    Safe to call between questions — no code is running at that point.
    Returns True if cleanup succeeded, False on any error.
    """
    from .config import SANDBOX_FUSION_URL
    try:
        resp = requests.post(
            SANDBOX_FUSION_URL,
            json={
                "code": (
                    "import shutil, os\n"
                    "removed = 0\n"
                    "for p in os.listdir('/tmp'):\n"
                    "    fp = os.path.join('/tmp', p)\n"
                    "    try:\n"
                    "        if os.path.isdir(fp):\n"
                    "            shutil.rmtree(fp)\n"
                    "        else:\n"
                    "            os.remove(fp)\n"
                    "        removed += 1\n"
                    "    except OSError:\n"
                    "        pass\n"
                    "print(f'Cleaned {removed} items from /tmp')\n"
                ),
                "language": "python",
                "run_timeout": 5,
                "compile_timeout": 5,
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.debug(f"Sandbox cleanup failed: {e}")
        return False

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


def _search_exa(q: str, num_results: int = 5) -> str:
    """Neural search via Exa.ai API.

    Exa combines keyword and neural search, making it good at finding
    relevant pages even when the query is more conceptual than keyword-exact.
    Requires EXA_API_KEY in the environment.
    """
    import httpx

    api_key = os.getenv("EXA_API_KEY", "")
    if not api_key:
        return "Search error: Exa API key not configured. Set EXA_API_KEY environment variable."

    payload = {
        "query": q,
        "numResults": min(num_results, 10),
        "type": "auto",                  # let Exa pick keyword vs neural
        "contents": {
            "text": {"maxCharacters": 300},   # short snippet per result
        },
    }

    last_error = None
    for attempt in range(MAX_SEARCH_RETRIES + 1):
        try:
            logger.info(f"Exa search (attempt {attempt+1}): {q}")
            resp = httpx.post(
                "https://api.exa.ai/search",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=SEARCH_TIMEOUT,
            )
            if resp.status_code == 401:
                return "Search error: Invalid Exa API key. Check EXA_API_KEY."
            if resp.status_code == 429:
                return "Search error: Exa rate limit exceeded."
            if resp.status_code != 200:
                error_msg = f"Search HTTP error {resp.status_code}"
                logger.error(f"Exa: {error_msg}")
                return error_msg

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return f"No results found for query: {q}"

            formatted = f"Search Results for '{q}':\n\n"
            for i, r in enumerate(results[:num_results], 1):
                title = r.get("title") or "No title"
                link = r.get("url") or "No link"
                snippet = (r.get("text") or "").strip()[:250] or "No snippet"
                formatted += f"{i}. {title}\n   URL: {link}\n   {snippet}\n\n"

            logger.info(f"Successfully retrieved {len(results[:num_results])} search results via Exa")
            return formatted

        except Exception as e:
            last_error = e
            logger.warning(f"Exa search error (attempt {attempt+1}/{MAX_SEARCH_RETRIES+1}): {e}")
            if attempt < MAX_SEARCH_RETRIES:
                time.sleep(1 * (attempt + 1))
                continue

    return f"Exa search error after retries: {type(last_error).__name__}: {last_error}"


# Error patterns that indicate the primary search backend is exhausted/broken
# and we should fall back for all remaining queries.
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
      2. Fallback 1: Exa.ai neural search (if EXA_API_KEY is set)
      3. Fallback 2: DuckDuckGo (no key required)

    Args:
        q: Search query string
        num_results: Number of results to return (default: 5)

    Returns:
        Formatted string with top search results or error message.
    """
    backend = os.getenv("SEARCH_BACKEND", "serper").lower()
    if backend == "serpapi":
        result = _search_serpapi(q, num_results)
    else:
        result = _search_serper(q, num_results)

    # If primary succeeded, return immediately
    if not any(result.startswith(err) for err in _SEARCH_FALLBACK_ERRORS):
        return result

    primary_error = result
    logger.warning(f"Primary search failed ({primary_error}), trying Exa fallback")

    # ── Fallback 1: Exa ──────────────────────────────────────────────
    exa_result = _search_exa(q, num_results)
    if not (exa_result.startswith("Search error:") or exa_result.startswith("Exa search error")):
        return exa_result
    logger.warning(f"Exa fallback failed ({exa_result}), trying DDG")

    # ── Fallback 2: DuckDuckGo ───────────────────────────────────────
    ddg_result = _search_ddg(q, num_results)
    if not (ddg_result.startswith("Search error:") or ddg_result.startswith("DDG search error:")):
        return ddg_result

    # All three failed
    return (
        f"{primary_error}\n"
        f"[Exa fallback also failed: {exa_result}]\n"
        f"[DDG fallback also failed: {ddg_result}]"
    )

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

# ── Thread-safe cookie jar ────────────────────────────────────────────
# Cookie strategy: NO shared jar.  Each httpx.get() manages its own
# redirect-chain cookies internally.  A global jar leaked cookies across
# domains and caused "Multiple cookies exist with name=__cf_bm" errors
# when several Cloudflare-fronted sites each set that cookie.
import httpx as _httpx
import threading as _threading

# ── Per-domain fetch rate limiter ─────────────────────────────────────
# Prevents rapid-fire requests to the same domain, which is the #1 cause
# of Cloudflare/WAF blocks. Each domain gets its own last-request timestamp.
_FETCH_MIN_INTERVAL = 1.5  # seconds between requests to the same domain
_domain_fetch_times: dict[str, float] = {}
_domain_fetch_lock = _threading.Lock()

def _domain_rate_wait(domain: str) -> None:
    """Block until at least _FETCH_MIN_INTERVAL has passed since the last
    request to this domain. Thread-safe."""
    if not domain:
        return
    with _domain_fetch_lock:
        now = time.time()
        last = _domain_fetch_times.get(domain, 0.0)
        wait = last + _FETCH_MIN_INTERVAL - now
        if wait > 0:
            _domain_fetch_times[domain] = last + _FETCH_MIN_INTERVAL
        else:
            _domain_fetch_times[domain] = now
    # Sleep OUTSIDE the lock so other domains aren't blocked
    if wait > 0:
        time.sleep(wait)

# ── Known-hostile domains (always skip direct, go Jina first) ─────────
# These domains are known to aggressively block automated access.
# Going direct wastes time and poisons the session blocklist.
_JINA_FIRST_DOMAINS: frozenset[str] = frozenset({
    # Data / analytics platforms
    "www.statista.com", "statista.com",
    "www.similarweb.com", "similarweb.com",
    "www.semrush.com", "semrush.com",
    # News paywalls
    "www.wsj.com", "wsj.com",
    "www.ft.com", "ft.com",
    "www.nytimes.com", "nytimes.com",
    "www.washingtonpost.com", "washingtonpost.com",
    "www.bloomberg.com", "bloomberg.com",
    "www.economist.com", "economist.com",
    "www.theatlantic.com", "theatlantic.com",
    # News — heavy CF / JS-rendered pages
    "www.theguardian.com", "theguardian.com",
    # Heavy CF protection
    "www.glassdoor.com", "glassdoor.com",
    "www.linkedin.com", "linkedin.com",
    "www.indeed.com", "indeed.com",
    "www.zillow.com", "zillow.com",
    "www.realtor.com", "realtor.com",
    # Academic paywalls
    "www.sciencedirect.com", "sciencedirect.com",
    "www.jstor.org", "jstor.org",
    "www.springer.com", "springer.com",
    "link.springer.com",
})

# Patterns that indicate a site blocked us
_BLOCK_PATTERNS = [
    "access denied", "403 forbidden", "just a moment",
    "checking your browser", "enable javascript", "captcha",
    "blocked", "unusual traffic", "bot detection",
    "please verify", "security check", "are you a robot",
]

# Cloudflare-specific patterns (subset — these mean retries are pointless)
_CLOUDFLARE_PATTERNS = [
    "cloudflare", "cf-browser-verification", "cf-challenge",
    "cf-chl-bypass", "_cf_chl", "ray id",
]

# Session-level domain blocklist with TTL: Cloudflare blocks are often
# transient (CDN-edge variance, rotating challenges).  Entries expire after
# _BLOCKED_DOMAIN_TTL seconds so a momentary block doesn't permanently
# degrade a domain for the rest of the eval run.
_BLOCKED_DOMAIN_TTL = 300  # seconds (5 min)
_blocked_domains: dict[str, float] = {}   # domain → timestamp of block
_blocked_domains_lock = _threading.Lock()

# ── URL response cache: avoids re-fetching the same page across workers ──
# Stores raw HTML keyed by URL. Parsing (extract, css_selector, max_chars)
# still happens per-call so different callers all benefit from one fetch.
_URL_CACHE_TTL = 600  # seconds (10 min)
_url_cache: dict[str, tuple[str, str, int, float, str]] = {}  # url → (html, final_url, status, ts, content_type)
_url_cache_lock = _threading.Lock()

# ── URL fetch counter + content fingerprint ───────────────────────────
# Tracks how many times each URL has been requested across all agents in
# the same process, plus a content hash so we can tell agents "you are
# seeing identical content that N other agents already saw."
import hashlib as _hashlib
_url_fetch_counts: dict[str, int] = {}     # url → total fetch count
_url_content_hashes: dict[str, str] = {}   # url → md5 hex of last content
_url_fetch_meta_lock = _threading.Lock()

def _record_url_access(url: str, content: str) -> tuple[int, bool]:
    """Record a URL access.  Returns (fetch_count, content_unchanged).

    fetch_count:       total number of times this URL has been requested.
    content_unchanged: True if the content hash is the same as the previous
                       fetch (i.e. the page hasn't changed).
    """
    h = _hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
    with _url_fetch_meta_lock:
        _url_fetch_counts[url] = _url_fetch_counts.get(url, 0) + 1
        count = _url_fetch_counts[url]
        prev = _url_content_hashes.get(url)
        _url_content_hashes[url] = h
        unchanged = prev is not None and prev == h
    return count, unchanged

# ── Parsed-text page cache: full cleaned text, keyed by URL ──────────
# Populated by _parse_html_content *before* truncation so that read_page
# can serve arbitrary offsets without re-parsing.  Same TTL as URL cache.
_page_text_cache: dict[str, tuple[str, float]] = {}  # url → (full_text, timestamp)
_page_text_cache_lock = _threading.Lock()

from urllib.parse import urlparse as _urlparse


def _is_domain_blocked(domain: str) -> bool:
    """Check if domain is in the blocklist and its TTL hasn't expired."""
    with _blocked_domains_lock:
        ts = _blocked_domains.get(domain)
        if ts is None:
            return False
        if time.time() - ts > _BLOCKED_DOMAIN_TTL:
            del _blocked_domains[domain]
            logger.debug(f"Domain {domain} blocklist entry expired (TTL={_BLOCKED_DOMAIN_TTL}s)")
            return False
        return True


def _block_domain(domain: str) -> None:
    """Add a domain to the blocklist with current timestamp.

    Infrastructure domains (Wayback, Jina) are never blocklisted because
    doing so would disable our own fallback tiers.
    """
    _NEVER_BLOCK = frozenset({"web.archive.org", "r.jina.ai", "s.jina.ai"})
    if domain in _NEVER_BLOCK:
        logger.debug(f"Refusing to blocklist infrastructure domain {domain}")
        return
    with _blocked_domains_lock:
        _blocked_domains[domain] = time.time()


def reset_fetch_state() -> None:
    """Clear all session-level fetch state between questions.

    Call this between eval questions so that transient blocks,
    rate-limit timestamps, and other per-session state don't leak
    across independent questions.
    """
    with _blocked_domains_lock:
        cleared = len(_blocked_domains)
        _blocked_domains.clear()
    with _domain_fetch_lock:
        _domain_fetch_times.clear()
    with _url_cache_lock:
        cached = len(_url_cache)
        _url_cache.clear()
    with _page_text_cache_lock:
        pages = len(_page_text_cache)
        _page_text_cache.clear()
    if cleared or cached or pages:
        logger.debug(f"reset_fetch_state: cleared {cleared} blocked domains, {cached} cached URLs, {pages} parsed pages")


def _extract_domain(url: str) -> str:
    """Extract domain from URL for blocklist lookups."""
    try:
        return _urlparse(url).netloc.lower()
    except Exception:
        return ""


def _get_cached_full_length(url: str) -> int:
    """Return full-text length from _page_text_cache, or 0 if not cached."""
    with _page_text_cache_lock:
        cached = _page_text_cache.get(url)
    return len(cached[0]) if cached else 0


def _extract_api_error_label(body) -> str | None:
    """Try to extract a human-readable error message from a JSON API error body.

    Handles common patterns from APIs like Eurostat, REST frameworks, etc.:
      {"error": [{"status": 400, "label": "Invalid value for ..."}]}
      {"error": "some message"}
      {"message": "some message"}
      {"detail": "some message"}
      {"errors": [{"message": "..."}]}
    Returns None if no recognisable error structure is found.
    """
    if not isinstance(body, dict):
        return None
    # {"error": [{"label": "..."}]}  (Eurostat style)
    err = body.get("error")
    if isinstance(err, list):
        labels = [e.get("label") or e.get("message") or e.get("detail") or ""
                  for e in err if isinstance(e, dict)]
        labels = [l for l in labels if l]
        if labels:
            return "; ".join(labels)
    # {"error": "string message"}
    if isinstance(err, str) and err:
        return err
    # {"message": "..."}
    msg = body.get("message")
    if isinstance(msg, str) and msg:
        return msg
    # {"detail": "..."}
    detail = body.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    # {"errors": [{"message": "..."}]}
    errs = body.get("errors")
    if isinstance(errs, list):
        msgs = [e.get("message") or "" for e in errs if isinstance(e, dict)]
        msgs = [m for m in msgs if m]
        if msgs:
            return "; ".join(msgs)
    return None


def _status_code_hint(status_code: int, api_label: str | None = None) -> str:
    """Return a status-code-appropriate hint for the model.

    Critical: distinguishes between 'blocked' (403/CF) and
    'bad request' (400) / 'not found' (404) which are API-level
    errors where retrying a different renderer (Jina/Wayback) won't help.
    """
    if api_label:
        _label = api_label.rstrip(". ")
        return (
            f"The remote API rejected this request (HTTP {status_code}): {_label}. "
            "Fix the query parameters and retry with corrected values. "
            "Do NOT retry the exact same URL."
        )
    if status_code == 400:
        return (
            "Bad Request (HTTP 400): the server understood the URL but the query "
            "parameters are invalid. Check the API documentation for correct "
            "parameter names/values and retry with a fixed URL."
        )
    if status_code == 404:
        return (
            "Not Found (HTTP 404): the resource does not exist at this URL. "
            "The path, dataset ID, or endpoint name may be wrong. "
            "Search for the correct URL or try a different data source."
        )
    if status_code == 403:
        return (
            "Forbidden (HTTP 403): the site blocked automated access. "
            "Do NOT retry this URL. Search for the same information from a "
            "different source, or try wikipedia_lookup() if applicable."
        )
    if status_code == 429:
        return (
            "Rate Limited (HTTP 429): too many requests. "
            "Wait before retrying, or search for the data from a different source."
        )
    if status_code >= 500:
        return (
            f"Server Error (HTTP {status_code}): the remote server had an internal "
            "error. This may be transient — you can retry once, or try a different source."
        )
    return (
        f"HTTP {status_code} error. Do NOT retry the exact same URL. "
        "Search for the same information from a different source."
    )


def _generate_structured_preview(raw_text: str, content_type: str) -> str:
    """Generate a smart preview of structured data (JSON/CSV/XML).

    Instead of blind truncation that breaks records mid-field, this
    produces a human-readable summary: schema, record count, and
    a few sample records.  The full data remains in _page_text_cache
    for read_page() and in memory for sub-agent consumption.
    """
    lines: list[str] = []
    total = len(raw_text)

    # ── JSON ──────────────────────────────────────────────────────
    if "json" in content_type or raw_text.lstrip()[:1] in ("{" , "["):
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            lines.append(f"[⚠ PREVIEW — Malformed JSON, {total:,} chars total, showing first 3,000]")
            lines.append(raw_text[:3000])
            return "\n".join(lines)

        if isinstance(data, list):
            lines.append(f"[⚠ PREVIEW — JSON Array, {len(data):,} records, {total:,} chars total. Full data is in memory.]")
            if data and isinstance(data[0], dict):
                lines.append(f"Record keys: {', '.join(data[0].keys())}")
            for i, rec in enumerate(data[:3]):
                lines.append(f"  [{i}] {json.dumps(rec, ensure_ascii=False)[:600]}")
            if len(data) > 3:
                lines.append(f"  ... ({len(data) - 3:,} more records)")
        elif isinstance(data, dict):
            lines.append(f"[⚠ PREVIEW — JSON Object, {len(data)} top-level keys, {total:,} chars total. Full data is in memory.]")
            for k, v in data.items():
                if isinstance(v, list):
                    lines.append(f"  '{k}': array[{len(v)}]")
                    if v and isinstance(v[0], dict):
                        lines.append(f"    item keys: {', '.join(v[0].keys())}")
                        lines.append(f"    sample: {json.dumps(v[0], ensure_ascii=False)[:400]}")
                    elif v:
                        lines.append(f"    sample: {json.dumps(v[:5], ensure_ascii=False)[:400]}")
                elif isinstance(v, dict):
                    sub_keys = list(v.keys())
                    lines.append(f"  '{k}': object({len(sub_keys)} keys)")
                    if len(sub_keys) <= 8:
                        lines.append(f"    keys: {', '.join(str(sk) for sk in sub_keys)}")
                    else:
                        lines.append(f"    keys (first 8): {', '.join(str(sk) for sk in sub_keys[:8])} ...")
                else:
                    lines.append(f"  '{k}': {json.dumps(v, ensure_ascii=False)[:200]}")
        else:
            lines.append(f"[⚠ PREVIEW — JSON primitive, {total:,} chars total]")
            lines.append(json.dumps(data, ensure_ascii=False)[:2000])
        return "\n".join(lines)

    # ── CSV ───────────────────────────────────────────────────────
    if "csv" in content_type:
        text_lines = raw_text.split("\n")
        non_empty = [l for l in text_lines if l.strip()]
        lines.append(f"[⚠ PREVIEW — CSV Data, {len(non_empty):,} rows, {total:,} chars total. Full data is in memory.]")
        for i, line in enumerate(non_empty[:5]):
            prefix = "  HEADER: " if i == 0 else f"  [{i}] "
            lines.append(f"{prefix}{line[:500]}")
        if len(non_empty) > 5:
            lines.append(f"  ... ({len(non_empty) - 5:,} more rows)")
        return "\n".join(lines)

    # ── XML ───────────────────────────────────────────────────────
    if "xml" in content_type:
        import re as _re_xml
        tags = _re_xml.findall(r"<(\w+)[\s>]", raw_text[:10000])
        unique_tags = list(dict.fromkeys(tags))[:20]
        lines.append(f"[⚠ PREVIEW — XML Data, {total:,} chars total. Full data is in memory.]")
        if unique_tags:
            lines.append(f"Elements: {', '.join(unique_tags)}")
        lines.append(raw_text[:3000])
        return "\n".join(lines)

    # Fallback
    lines.append(f"[⚠ PREVIEW — Structured data, {total:,} chars total. Full data is in memory.]")
    lines.append(raw_text[:3000])
    return "\n".join(lines)


def _detect_jina_api_error(jina_content: str) -> tuple[int, str] | None:
    """Detect API errors rendered through Jina Reader.

    Jina renders the target page even when it returns an HTTP error.
    The output looks like:
        Title: ...
        URL Source: ...
        Warning: Target URL returned error 400: Bad Request
        Markdown Content:
        { "error": [{"status": 400, "label": "Invalid value for ..."}]}

    Returns (status_code, error_label) if an API error is found, else None.
    Does NOT trigger on generic HTML error pages (only structured JSON bodies).
    """
    import re as _re

    # 1. Check for Jina's "Warning: Target URL returned error NNN" header
    warn_match = _re.search(
        r'Warning:\s*Target URL returned error (\d{3})', jina_content
    )
    if not warn_match:
        return None

    err_code = int(warn_match.group(1))
    # Only intercept client errors that indicate bad parameters, not blocks
    if err_code not in (400, 404, 422):
        return None

    # 2. Try to extract a JSON error body from the content after "Markdown Content:"
    mc_idx = jina_content.find("Markdown Content:")
    if mc_idx < 0:
        mc_idx = 0
    remainder = jina_content[mc_idx:]

    # Look for JSON object in the remainder
    json_match = _re.search(r'\{[^{}]*"error"[^{}]*\}', remainder)
    if not json_match:
        # Try to find a nested JSON structure
        json_match = _re.search(r'\{.*\}', remainder, _re.DOTALL)

    if json_match:
        try:
            err_body = json.loads(json_match.group())
            label = _extract_api_error_label(err_body)
            if label:
                return err_code, label
        except (json.JSONDecodeError, ValueError):
            pass

    # Even without a parseable JSON body, the Jina warning itself tells us
    # the API returned an error — surface a generic message
    return err_code, f"HTTP {err_code} error from the target API"

def _is_blocked(text: str, status_code: int) -> tuple[bool, str, bool]:
    """Check if a response looks like a bot-block page.

    Returns:
        (is_blocked, reason, is_cloudflare) — the third element indicates
        whether this is specifically a Cloudflare/WAF challenge, meaning
        retries are pointless and the domain should be blocklisted.
    """
    if status_code in (403, 429, 503):
        # Check body for CF signatures even on error status codes
        lower = text[:3000].lower()
        for cf_pat in _CLOUDFLARE_PATTERNS:
            if cf_pat in lower:
                return True, f"Cloudflare ({cf_pat})", True
        return True, f"HTTP {status_code}", False

    lower = text[:3000].lower()

    # Check Cloudflare first (more specific) — always checked
    for cf_pat in _CLOUDFLARE_PATTERNS:
        if cf_pat in lower:
            return True, f"Cloudflare ({cf_pat})", True

    # Generic block patterns — only on SHORT pages.
    # Real block/challenge pages are tiny (<5 KB).  A long page that
    # incidentally contains "blocked" or "access denied" in its body
    # is almost certainly legitimate content.  This prevents false
    # positives from articles about ad-blockers, road blocks, etc.
    if len(text) < 5000:
        for pat in _BLOCK_PATTERNS:
            if pat in lower:
                return True, f"block pattern: {pat}", False

    return False, "", False


def _jina_reader_fallback(url: str, max_chars: int = 12000,
                         accept: str = "text/plain",
                         target_selector: str = "",
                         wait_for_selector: str = "",
                         min_content_len: int = 200) -> str | None:
    """Fallback: use Jina Reader (headless Chrome) to render JS-heavy pages.

    Two-tier strategy:
      1. GET  r.jina.ai/{url} — fast, benefits from Jina's CDN cache.
      2. POST r.jina.ai/       — slower, forces full network-idle wait.
         Only used when GET returns too little content (likely JS-rendered).

    Args:
        url: The original URL to fetch via Jina.
        max_chars: Truncate returned text to this many characters.
        accept: MIME type for the Accept header.
            - "text/plain" (default): returns clean Markdown text.
            - "text/html": returns rendered HTML — needed when the
              caller will parse <table> elements from the output.
        target_selector: CSS selector for Jina's x-target-selector header.
            When set, Jina returns only content within the matched element.
        wait_for_selector: CSS selector for Jina's x-wait-for-selector header.
            When set, Jina waits for this element to render before returning.
        min_content_len: If GET returns fewer chars than this, escalate to POST.

    Returns cleaned text/HTML or None on failure.
    """
    import httpx
    _wfs = wait_for_selector or "article, main, #content, .content, table"
    headers = {
        "Accept": accept,
        "User-Agent": _BROWSER_HEADERS["User-Agent"],
        "x-timeout": "30",
        "x-wait-for-selector": _wfs,
    }
    # Authenticate if a Jina API key is available (higher rate limits + priority)
    _jina_key = os.getenv("JINA_API_KEY", "")
    if _jina_key:
        headers["Authorization"] = f"Bearer {_jina_key}"
    if target_selector:
        headers["x-target-selector"] = target_selector

    # ── Tier 1: GET (fast, cacheable) ─────────────────────────────────
    get_text = None
    try:
        jina_url = f"https://r.jina.ai/{url}"
        resp = httpx.get(
            jina_url,
            headers=headers,
            timeout=30,
            follow_redirects=True,
        )
        if resp.status_code == 200 and len(resp.text.strip()) > 50:
            get_text = resp.text.strip()
    except Exception as e:
        logger.debug(f"Jina GET failed for {url}: {e}")

    if get_text and len(get_text) >= min_content_len:
        # Cache full text for read_page before truncating
        with _page_text_cache_lock:
            _page_text_cache[url] = (get_text, time.time())
        return get_text[:max_chars]

    # ── Tier 2: POST (forces network-idle wait, handles SPAs / hash routes) ─
    logger.debug(
        f"Jina GET {'returned only ' + str(len(get_text)) + ' chars' if get_text else 'failed'} "
        f"for {url} — escalating to POST"
    )
    try:
        resp = httpx.post(
            "https://r.jina.ai/",
            headers=headers,
            data={"url": url},
            timeout=35,
            follow_redirects=True,
        )
        if resp.status_code == 200 and len(resp.text.strip()) > 50:
            post_text = resp.text.strip()
            # Use POST result only if it's actually better than GET
            if get_text and len(post_text) <= len(get_text):
                best = get_text
            else:
                best = post_text
            with _page_text_cache_lock:
                _page_text_cache[url] = (best, time.time())
            return best[:max_chars]
    except Exception as e:
        logger.debug(f"Jina POST failed for {url}: {e}")

    # Return whatever GET got, even if short — better than nothing
    if get_text:
        with _page_text_cache_lock:
            _page_text_cache[url] = (get_text, time.time())
        return get_text[:max_chars]
    return None


def _exa_contents_fallback(url: str, max_chars: int = 12000) -> str | None:
    """Fallback: use Exa.ai contents API to retrieve page text.

    Exa maintains its own crawl index and can serve content even when
    the live page blocks direct/Jina access or requires JS/cookies.

    Sits between Jina and Wayback in the fallback chain.
    Requires EXA_API_KEY in the environment (.env).

    Returns cleaned text or None on failure / missing API key.
    """
    import httpx

    api_key = os.getenv("EXA_API_KEY", "")
    if not api_key:
        return None

    try:
        resp = httpx.post(
            "https://api.exa.ai/contents",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "ids": [url],
                "text": {"maxCharacters": max_chars},
            },
            timeout=20,
        )
        if resp.status_code != 200:
            logger.debug(f"Exa contents API returned {resp.status_code} for {url}")
            return None

        data = resp.json()
        results = data.get("results", [])
        if not results:
            logger.debug(f"Exa contents API returned no results for {url}")
            return None

        text = results[0].get("text", "").strip()
        if len(text) < 50:
            logger.debug(f"Exa contents returned only {len(text)} chars for {url}")
            return None

        # Cache for read_page / memory
        with _page_text_cache_lock:
            _page_text_cache[url] = (text, time.time())

        logger.info(f"[fetch_url] Exa contents fallback succeeded for {url} ({len(text)} chars)")
        return text[:max_chars]

    except Exception as e:
        logger.debug(f"Exa contents fallback failed for {url}: {e}")
        return None


def _parse_html_content(resp_text: str, resp_url, resp_status_code: int,
                        extract: str, css_selector: str, max_chars: int,
                        attempt: int) -> dict:
    """Parse an HTML response into structured content. Shared by direct + Jina paths."""
    from bs4 import BeautifulSoup
    import warnings

    content_type_hdr = ""  # already validated by caller
    raw = resp_text

    # Detect XML vs HTML
    if raw.lstrip()[:20].startswith("<?xml"):
        parser = "xml"
    else:
        parser = "html.parser"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*XMLParsedAsHTMLWarning.*")
        soup = BeautifulSoup(raw, parser)

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # ── Inject CSS class metadata into table rows as visible text ─────
    # Many data sites (FishBase, GBIF, etc.) encode row status
    # (introduced, native, endemic, endangered) purely via CSS classes.
    # Before text extraction strips the HTML, surface these as text.
    # Only needed for text mode — table mode captures classes via _row_class.
    if extract != "table":
        _STYLE_NOISE_INJ = frozenset({
            "odd", "even", "alt", "row", "tr", "td", "first", "last",
            "active", "hover", "selected", "highlight", "striped",
            "border", "padding", "center", "left", "right", "top",
            "visible", "hidden", "collapsed", "expanded",
        })
        for tr_tag in soup.find_all("tr"):
            tr_classes = set(tr_tag.get("class", []))
            for td_tag in tr_tag.find_all(["td", "th"]):
                tr_classes.update(td_tag.get("class", []))
            semantic = [
                c for c in tr_classes
                if c and c.lower() not in _STYLE_NOISE_INJ
                and not c.startswith("col-")
            ]
            if semantic:
                last_cell = tr_tag.find_all(["td", "th"])
                if last_cell:
                    tag_text = soup.new_string(
                        f" [{', '.join(sorted(semantic))}]"
                    )
                    last_cell[-1].append(tag_text)

    # Apply CSS selector if provided
    if css_selector:
        selected = soup.select(css_selector)
        if selected:
            from bs4 import BeautifulSoup as BS
            combined = "\n".join(str(el) for el in selected)
            soup = BS(combined, "html.parser")

    if extract == "table":
        # Use shared table parser — same logic as extract_tables
        parsed = _parse_tables_from_soup(soup, max_chars=max_chars, css_selector="")
        if parsed["tables"]:
            lines = []
            for tbl_data in parsed["tables"]:
                meta = tbl_data["meta"]
                # Table header line
                label_parts = []
                if meta.get("nearest_heading"):
                    label_parts.append(meta["nearest_heading"])
                if meta.get("caption"):
                    label_parts.append(meta["caption"])
                if label_parts:
                    lines.append(f"=== {' — '.join(label_parts)} ===")
                elif len(parsed["tables"]) > 1:
                    lines.append(f"=== Table {meta['table_index']} ===")

                if tbl_data["rows"]:
                    # Check if any rows carry CSS class metadata
                    has_row_classes = any(
                        "_row_class" in row for row in tbl_data["rows"]
                    )
                    # Column headers
                    if meta["headers"]:
                        hdr_line = " | ".join(meta["headers"])
                        if has_row_classes:
                            hdr_line += " | [status]"
                        lines.append(hdr_line)
                        lines.append("-" * min(len(hdr_line), 80))
                    # Data rows
                    for row in tbl_data["rows"]:
                        row_class = row.pop("_row_class", "")
                        row.pop("_data", None)  # drop for text rendering
                        vals = " | ".join(str(v) for v in row.values())
                        if has_row_classes and row_class:
                            vals += f" | {row_class}"
                        lines.append(vals)
                else:
                    # Truncated table — show preview only
                    lines.append(f"[TABLE TRUNCATED — {meta['total_rows']} rows. "
                                 f"Headers: {', '.join(meta['headers']) if meta['headers'] else 'none'}]")
                    if meta.get("preview_rows"):
                        lines.append("Preview:")
                        for pr in meta["preview_rows"]:
                            lines.append("  " + " | ".join(str(v) for v in pr.values()))
                lines.append("")  # blank line between tables

            # Truncation footer
            if parsed["truncated"]:
                lines.append(
                    f"[TRUNCATED: showed {parsed['tables_returned']} of "
                    f"{parsed['tables_found']} tables. "
                    f"Use extract_tables(css_selector='...') to target a specific table.]"
                )
            content = "\n".join(lines)
        else:
            content = soup.get_text(separator="\n", strip=True)
    else:
        # ── trafilatura extraction (preferred for article pages) ─────
        # trafilatura is trained on thousands of websites and does a much
        # better job than hand-rolled heuristics at isolating the main
        # article body.  We fall through to BeautifulSoup if it's not
        # installed or returns too little content (e.g. non-article pages).
        content = None
        if not css_selector:
            try:
                import trafilatura
                _traf = trafilatura.extract(
                    resp_text,
                    include_links=True,
                    include_tables=True,
                    favor_recall=True,
                )
                if _traf and len(_traf) > 200:
                    content = _traf
            except Exception:
                pass  # trafilatura not installed or crashed — fall through

        # ── BeautifulSoup fallback (css_selector, or trafilatura too short)
        if content is None:
            _content_soup = None
            if not css_selector:
                for _sel in ("article", "main", "[role='main']", ".post-content", ".entry-content", "#content", "#mw-content-text"):
                    _found = soup.select(_sel)
                    if _found:
                        _candidate_text = "\n".join(el.get_text(separator="\n", strip=True) for el in _found)
                        _full_text_len = len(soup.get_text())
                        if len(_candidate_text) > max(200, _full_text_len * 0.3):
                            _content_soup = _candidate_text
                            break

            if _content_soup:
                content = _content_soup
            else:
                content = soup.get_text(separator="\n", strip=True)

        # Collapse excessive whitespace
        import re as _re_clean
        content = _re_clean.sub(r'\n{3,}', '\n\n', content)
        _lines = content.split('\n')
        _cleaned = []
        _short_streak = 0
        for _line in _lines:
            _stripped = _line.strip()
            if len(_stripped) < 4 and _stripped:
                _short_streak += 1
                if _short_streak <= 2:
                    _cleaned.append(_stripped)
            else:
                _short_streak = 0
                _cleaned.append(_stripped)
        content = '\n'.join(_cleaned)

    # ── Cache full parsed text before truncation ──────────────────
    full_length = len(content)
    _url_str = str(resp_url)
    if full_length > 0:
        with _page_text_cache_lock:
            _page_text_cache[_url_str] = (content, time.time())

    content = content[:max_chars]

    return {
        "ok": True, "content": content, "url": _url_str,
        "blocked": False, "reason": "", "status_code": resp_status_code,
        "retries": attempt, "_full_length": full_length,
    }


def fetch_url(url: str, max_chars: int = 12000, extract: str = "text",
              css_selector: str = "", max_retries: int = 2, timeout: int = 20) -> dict:
    """Fetch a web page with smart routing, domain blocklist, and multi-tier fallback.

    Strategy:
      1. If domain is in session blocklist → skip direct, go Jina first.
      2. Otherwise: single direct probe (no retries on block/CF).
         - If Cloudflare/WAF detected → blocklist domain, go Jina immediately.
         - If transient error (timeout, 500) → retry once with backoff.
      3. If direct content is suspiciously short → try Jina.
      4. If Jina fails → Wayback Machine auto-fallback.

    Args:
        url: URL to fetch.
        max_chars: Max characters of extracted text to return.
        extract: "text" for cleaned page text, "table" for pipe-delimited tables.
        css_selector: Optional CSS selector to target specific elements.
        max_retries: Number of retry attempts for transient (non-block) errors.
        timeout: Per-request timeout in seconds.

    Returns:
        Dict with keys: ok, content, url, blocked, reason, status_code, retries,
        and optionally source, hint.
    """
    import httpx
    import warnings

    last_error = ""
    last_status = 0
    retries_done = 0
    domain = _extract_domain(url)

    # ── Fast path: domain already known to be blocked ─────────────────
    _is_archive_url = "web.archive.org" in domain
    _is_jina_url = "r.jina.ai" in domain or "s.jina.ai" in domain

    if _is_domain_blocked(domain):
        logger.debug(f"Domain {domain} is blocklisted — skipping direct, trying fallbacks")
        jina_text = None
        if not _is_jina_url:
            jina_text = _jina_reader_fallback(url, max_chars)
        if jina_text:
            _jina_lower = jina_text[:1500].lower()
            if "page not found" not in _jina_lower and "404" not in _jina_lower:
                # For table extraction from Jina text, do a quick parse
                if extract == "table" or css_selector:
                    parsed = _parse_html_content(
                        jina_text, url, 200, extract, css_selector, max_chars, 0)
                    parsed["source"] = "jina_reader (blocklisted domain)"
                    return parsed
                return {
                    "ok": True, "content": jina_text, "url": url,
                    "blocked": False, "reason": "", "status_code": 200,
                    "retries": 0, "source": "jina_reader (blocklisted domain)",
                    "_full_length": _get_cached_full_length(url),
                }
        # Jina failed or skipped — try Exa contents
        exa_text = _exa_contents_fallback(url, max_chars)
        if exa_text:
            return {
                "ok": True, "content": exa_text, "url": url,
                "blocked": False, "reason": "", "status_code": 200,
                "retries": 0, "source": "exa_contents (blocklisted domain)",
                "_full_length": len(exa_text),
            }
        # Exa failed — try Wayback (skip if URL is already a Wayback URL)
        if not _is_archive_url:
            try:
                wb_result = fetch_cached(url=url, max_chars=max_chars)
                if wb_result.get("ok") and wb_result.get("content", "").strip():
                    wb_result["source"] = "wayback (blocklisted domain)"
                    return wb_result
            except Exception:
                pass
        # Last resort: try direct anyway — the block may have been transient
        _domain_rate_wait(domain)
        try:
            _lr_resp = httpx.get(url, headers=_BROWSER_HEADERS, timeout=timeout, follow_redirects=True)
            if _lr_resp.status_code < 400:
                _lr_blocked, _, _ = _is_blocked(_lr_resp.text, _lr_resp.status_code)
                if not _lr_blocked and len(_lr_resp.text.strip()) > 200:
                    parsed = _parse_html_content(
                        _lr_resp.text, _lr_resp.url, _lr_resp.status_code,
                        extract, css_selector, max_chars, 0)
                    parsed["source"] = "direct (last-resort, blocklisted domain)"
                    return parsed
        except Exception:
            pass
        return {
            "ok": False, "content": "", "url": url,
            "blocked": True, "reason": f"Domain {domain} is blocklisted; Jina, Wayback, and direct all failed",
            "status_code": 0, "retries": 0,
            "hint": "Do NOT retry this URL. Search for the same information from a different source.",
        }

    # ── Check URL response cache ─────────────────────────────────────
    with _url_cache_lock:
        _cached = _url_cache.get(url)
    if _cached:
        _c_html, _c_final_url, _c_status, _c_ts, _c_ct = _cached
        if time.time() - _c_ts < _URL_CACHE_TTL:
            logger.debug(f"URL cache hit for {url} ({len(_c_html)} chars, age {time.time()-_c_ts:.0f}s, ct={_c_ct})")

            # Structured data must go through the preview path, not HTML parser
            _is_struct_cached = (
                "application/json" in _c_ct
                or "text/json" in _c_ct
                or "text/csv" in _c_ct
                or "text/xml" in _c_ct
                or "application/xml" in _c_ct
                or "application/vnd" in _c_ct
            )
            if _is_struct_cached:
                full_length = len(_c_html)
                _STRUCT_INLINE_LIMIT = 3_000
                if full_length > _STRUCT_INLINE_LIMIT:
                    preview = _generate_structured_preview(_c_html, _c_ct)
                    return {
                        "ok": True, "content": preview, "url": _c_final_url,
                        "blocked": False, "reason": "",
                        "status_code": _c_status, "retries": 0,
                        "source": "cache",
                        "_full_length": full_length,
                        "_is_structured": True,
                        "_content_type": _c_ct.split(";")[0].strip(),
                    }
                else:
                    return {
                        "ok": True, "content": _c_html, "url": _c_final_url,
                        "blocked": False, "reason": "",
                        "status_code": _c_status, "retries": 0,
                        "source": "cache",
                        "_full_length": full_length,
                        "_is_structured": True,
                        "_content_type": _c_ct.split(";")[0].strip(),
                    }

            parsed = _parse_html_content(
                _c_html, _c_final_url, _c_status,
                extract, css_selector, max_chars, 0)
            parsed["source"] = "cache"
            return parsed
        else:
            # Expired — remove
            with _url_cache_lock:
                _url_cache.pop(url, None)

    # ── Normal path: single direct probe ──────────────────────────────
    # Known-hostile domains: skip direct entirely, go Jina first
    if domain in _JINA_FIRST_DOMAINS:
        logger.debug(f"Domain {domain} in JINA_FIRST — skipping direct fetch")
        jina_text = _jina_reader_fallback(url, max_chars)
        if jina_text:
            _jina_lower = jina_text[:1500].lower()
            if "page not found" not in _jina_lower and "404" not in _jina_lower:
                if extract == "table" or css_selector:
                    parsed = _parse_html_content(
                        jina_text, url, 200, extract, css_selector, max_chars, 0)
                    parsed["source"] = "jina_reader (jina-first domain)"
                    return parsed
                return {
                    "ok": True, "content": jina_text, "url": url,
                    "blocked": False, "reason": "", "status_code": 200,
                    "retries": 0, "source": "jina_reader (jina-first domain)",
                    "_full_length": _get_cached_full_length(url),
                }
        # Jina failed — try Exa contents
        exa_text = _exa_contents_fallback(url, max_chars)
        if exa_text:
            return {
                "ok": True, "content": exa_text, "url": url,
                "blocked": False, "reason": "", "status_code": 200,
                "retries": 0, "source": "exa_contents (jina-first domain)",
                "_full_length": len(exa_text),
            }
        # Exa failed — try Wayback
        try:
            wb_result = fetch_cached(url=url, max_chars=max_chars)
            if wb_result.get("ok") and wb_result.get("content", "").strip():
                wb_result["source"] = "wayback (jina-first domain)"
                return wb_result
        except Exception:
            pass
        # Last resort: try direct anyway — even hostile domains sometimes
        # serve partial content (paywall intro, first paragraphs) that's
        # better than returning nothing at all.
        _domain_rate_wait(domain)
        try:
            _lr_resp = httpx.get(url, headers=_BROWSER_HEADERS, timeout=timeout, follow_redirects=True)
            if _lr_resp.status_code < 400:
                _lr_blocked, _, _ = _is_blocked(_lr_resp.text, _lr_resp.status_code)
                if not _lr_blocked and len(_lr_resp.text.strip()) > 200:
                    parsed = _parse_html_content(
                        _lr_resp.text, _lr_resp.url, _lr_resp.status_code,
                        extract, css_selector, max_chars, 0)
                    parsed["source"] = "direct (last-resort, jina-first domain)"
                    return parsed
        except Exception:
            pass
        return {
            "ok": False, "content": "", "url": url,
            "blocked": True, "reason": f"Domain {domain} is hostile; Jina, Wayback, and direct all failed",
            "status_code": 0, "retries": 0,
            "hint": "Do NOT retry this URL. Search for the same information from a different source.",
        }

    # Per-domain rate limiting: wait if we recently fetched from this domain
    _domain_rate_wait(domain)

    for attempt in range(1 + max_retries):
        try:
            resp = httpx.get(
                url, headers=_BROWSER_HEADERS,
                timeout=timeout, follow_redirects=True,
            )
            last_status = resp.status_code

            if resp.status_code >= 400:
                blocked, reason, is_cf = _is_blocked(resp.text, resp.status_code)
                last_error = reason if blocked else f"HTTP {resp.status_code}"
                retries_done = attempt
                if is_cf:
                    # Cloudflare — blocklist and skip to Jina immediately
                    _block_domain(domain)
                    logger.debug(f"Cloudflare detected on {domain} — blocklisted, going to Jina")
                    break

                # ── Fix C: Detect structured JSON API errors ──────────
                # APIs (Eurostat, REST services) return clear JSON error
                # bodies on 400/404/422.  These are *not* access blocks —
                # Jina/Wayback won't help.  Return immediately with a
                # clear, actionable error message.
                _ct = resp.headers.get("content-type", "")
                if resp.status_code in (400, 404, 422) and (
                    "json" in _ct or resp.text.lstrip()[:1] == "{"
                ):
                    try:
                        _err_body = json.loads(resp.text)
                        _api_label = _extract_api_error_label(_err_body)
                        if _api_label:
                            logger.info(
                                f"[fetch_url] API error {resp.status_code} from {domain}: {_api_label}"
                            )
                            return {
                                "ok": False, "content": "", "url": str(resp.url),
                                "blocked": False,
                                "reason": f"API error (HTTP {resp.status_code}): {_api_label}",
                                "status_code": resp.status_code,
                                "retries": attempt, "source": "direct",
                                "hint": _status_code_hint(resp.status_code, _api_label),
                            }
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Non-CF error: retry once for transient issues (500, 502, timeout)
                if attempt < max_retries and resp.status_code in (500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break  # 4xx or exhausted retries

            # ── Detect binary/PDF responses before parsing as HTML ─────
            content_type = resp.headers.get("content-type", "")
            if "application/pdf" in content_type or (
                resp.content[:5] == b"%PDF-" and "html" not in content_type
            ):
                try:
                    _pdf_result = read_pdf(url, max_chars=max_chars)
                    pdf_text = _pdf_result.get("content", "") if isinstance(_pdf_result, dict) else str(_pdf_result)
                    if pdf_text and len(pdf_text.strip()) > 20:
                        return {
                            "ok": True, "content": f"[PDF detected — extracted text]\n\n{pdf_text}",
                            "url": str(resp.url), "blocked": False, "reason": "",
                            "status_code": resp.status_code, "retries": attempt,
                            "source": "read_pdf",
                        }
                except Exception as pdf_err:
                    logger.debug(f"PDF extraction failed for {url}: {pdf_err}")
                return {
                    "ok": False, "content": "", "url": str(resp.url),
                    "blocked": False, "reason": "URL serves a PDF file that could not be parsed",
                    "status_code": resp.status_code, "retries": attempt,
                    "hint": "Try read_pdf(url=...) directly, or search for an HTML version.",
                }

            # ── Detect structured data (JSON / CSV / XML) ─────────
            # APIs return machine-readable formats that should NOT be
            # parsed through BeautifulSoup.  Cache the full raw text
            # (for read_page + memory) and return a smart preview if
            # the response exceeds max_chars.
            _ct_lower = content_type.lower()
            _is_structured_ct = (
                "application/json" in _ct_lower
                or "text/json" in _ct_lower
                or "text/csv" in _ct_lower
                or "text/xml" in _ct_lower
                or "application/xml" in _ct_lower
                or "application/vnd" in _ct_lower   # vendor APIs
            )
            if _is_structured_ct:
                raw_text = resp.text
                _url_str = str(resp.url)
                full_length = len(raw_text)

                # Cache full content for read_page + memory
                # Cache under both final URL and original URL (may differ
                # after redirects); nodes.py looks up by original URL.
                with _page_text_cache_lock:
                    _page_text_cache[_url_str] = (raw_text, time.time())
                    if url != _url_str:
                        _page_text_cache[url] = (raw_text, time.time())
                with _url_cache_lock:
                    _url_cache[url] = (raw_text, _url_str, resp.status_code, time.time(), _ct_lower)

                # Structured data should be analyzed with code, not
                # eyeballed in context.  Always return a preview unless
                # the response is tiny (<3K).  The full raw data is in
                # _page_text_cache (served by read_page) and will be
                # stored in MemoryStore by nodes.py, so sub-agents can
                # receive it via memory_keys → agent_data.json →
                # execute_code(pandas/json).
                _STRUCT_INLINE_LIMIT = 3_000  # ~1.2K tokens for JSON

                if full_length > _STRUCT_INLINE_LIMIT:
                    preview = _generate_structured_preview(raw_text, _ct_lower)
                    logger.info(
                        f"[fetch_url] Structured data ({_ct_lower.split(';')[0]}) "
                        f"{full_length:,} chars from {domain} — returning preview"
                    )
                    return {
                        "ok": True, "content": preview, "url": _url_str,
                        "blocked": False, "reason": "",
                        "status_code": resp.status_code,
                        "retries": attempt, "source": "direct",
                        "_full_length": full_length,
                        "_is_structured": True,
                        "_content_type": _ct_lower.split(";")[0].strip(),
                    }
                else:
                    # Small enough — return raw content as-is
                    return {
                        "ok": True, "content": raw_text, "url": _url_str,
                        "blocked": False, "reason": "",
                        "status_code": resp.status_code,
                        "retries": attempt, "source": "direct",
                        "_full_length": full_length,
                        "_is_structured": True,
                        "_content_type": _ct_lower.split(";")[0].strip(),
                    }

            # Check for soft blocks (200 but Cloudflare challenge page, etc.)
            blocked, reason, is_cf = _is_blocked(resp.text, resp.status_code)
            if blocked:
                last_error = reason
                retries_done = attempt
                if is_cf:
                    _block_domain(domain)
                    logger.debug(f"Soft Cloudflare block on {domain} — blocklisted, going to Jina")
                    break  # skip to Jina immediately, no retries
                # Non-CF soft block: retry once
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break

            # ── Parse successfully ────────────────────────────────────
            parsed = _parse_html_content(
                resp.text, resp.url, resp.status_code,
                extract, css_selector, max_chars, attempt)

            # Suspiciously short? Might be a JS-only page → try Jina
            if len(parsed["content"]) < 200 and resp.status_code == 200:
                jina_text = _jina_reader_fallback(url, max_chars)
                if jina_text and len(jina_text) > len(parsed["content"]):
                    return {
                        "ok": True, "content": jina_text, "url": str(resp.url),
                        "blocked": False, "reason": "", "status_code": 200,
                        "retries": attempt, "source": "jina_reader",
                        "_full_length": _get_cached_full_length(url),
                    }

            # ── Cache the raw HTML for future calls to the same URL ────
            with _url_cache_lock:
                _url_cache[url] = (resp.text, str(resp.url), resp.status_code, time.time(), content_type)

            return parsed

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            retries_done = attempt
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

    # ── Direct failed — try Jina Reader ───────────────────────────────
    jina_text = _jina_reader_fallback(url, max_chars)
    if jina_text:
        _jina_lower = jina_text[:1500].lower()
        if "page not found" not in _jina_lower and "404" not in _jina_lower:
            return {
                "ok": True, "content": jina_text, "url": url,
                "blocked": False, "reason": "", "status_code": last_status,
                "retries": retries_done, "source": "jina_reader",
                "_full_length": _get_cached_full_length(url),
            }
        logger.debug(f"Jina Reader returned a 404 page for {url}, trying Exa")

    # ── Exa contents fallback ─────────────────────────────────────────
    exa_text = _exa_contents_fallback(url, max_chars)
    if exa_text:
        return {
            "ok": True, "content": exa_text, "url": url,
            "blocked": False, "reason": "", "status_code": last_status,
            "retries": retries_done, "source": "exa_contents",
            "_full_length": len(exa_text),
        }

    # ── Wayback Machine auto-fallback ─────────────────────────────────
    try:
        wb_result = fetch_cached(url=url, max_chars=max_chars)
        if wb_result.get("ok") and wb_result.get("content", "").strip():
            wb_result["source"] = "wayback_auto_fallback"
            return wb_result
    except Exception as wb_err:
        logger.debug(f"Wayback auto-fallback failed for {url}: {wb_err}")

    # ── Total failure ─────────────────────────────────────────────────
    # Fix B: Generate status-code-appropriate hint instead of always
    # saying "blocked".  A 404 means "not found", not "blocked".
    _is_actually_blocked = last_status in (403, 429, 503) or "cloudflare" in last_error.lower()
    return {
        "ok": False, "content": "", "url": url,
        "blocked": _is_actually_blocked, "reason": last_error,
        "status_code": last_status, "retries": retries_done,
        "hint": _status_code_hint(last_status),
    }

def read_pdf(url: str, max_chars: int = 12000) -> dict:
    """Download and extract text from a PDF at a URL.

    Returns a dict with keys: ok, content, full_length, url.
    Caches full text in _page_text_cache so read_page() works for PDFs.
    """
    import httpx, io
    import pypdf
    resp = httpx.get(url, timeout=30)
    # Suppress noisy pypdf warnings ("Ignoring wrong pointing object",
    # "Advanced encoding ... not implemented yet")
    pdf_logger = logging.getLogger("pypdf")
    prev_level = pdf_logger.level
    try:
        pdf_logger.setLevel(logging.ERROR)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Advanced encoding")
            reader = pypdf.PdfReader(io.BytesIO(resp.content))
            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        pdf_logger.setLevel(prev_level)

    # Cache full text so read_page() can paginate through it
    with _page_text_cache_lock:
        _page_text_cache[url] = (full_text, time.time())

    return {
        "ok": True,
        "content": full_text[:max_chars],
        "full_length": len(full_text),
        "url": url,
        "page_count": len(reader.pages),
    }


# ── Shared table parsing ──────────────────────────────────────────────
def _parse_tables_from_soup(soup, max_chars: int = 12000,
                            css_selector: str = "") -> dict:
    """Parse <table> elements from a BeautifulSoup object into structured data.

    Shared by extract_tables and fetch_url(extract="table").
    Returns a dict with rich per-table metadata (headers, caption,
    nearest heading, row count, preview rows) and truncation info.

    Args:
        soup: BeautifulSoup object (already cleaned of script/style/nav).
        max_chars: Soft cap on total JSON output size for table data.
        css_selector: Optional CSS selector to target specific table(s).

    Returns:
        Dict with keys:
          tables: list of table dicts (each with 'meta' and 'rows')
          tables_found: total tables on page (before filtering)
          tables_returned: tables included in output
          truncated: whether any tables were dropped due to char limit
          layout_tables_skipped: count of trivial layout tables filtered
    """
    # Locate tables
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

    total_found = len(tables_html)

    # Filter out layout tables (≤2 rows AND ≤2 columns)
    data_tables = []
    layout_skipped = 0
    for tbl in tables_html:
        rows = tbl.find_all("tr")
        if len(rows) <= 2:
            max_cols = max(
                (len(tr.find_all(["td", "th"])) for tr in rows), default=0
            )
            if max_cols <= 2:
                layout_skipped += 1
                continue
        data_tables.append(tbl)

    if not data_tables:
        return {
            "tables": [],
            "tables_found": total_found,
            "tables_returned": 0,
            "truncated": False,
            "layout_tables_skipped": layout_skipped,
        }

    all_tables = []
    total_chars = 0
    truncated = False

    for idx, tbl in enumerate(data_tables):
        # ── Extract headers ───────────────────────────────────────
        headers = []
        thead = tbl.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
        if not headers:
            first_row = tbl.find("tr")
            if first_row and first_row.find("th"):
                headers = [th.get_text(strip=True) for th in first_row.find_all("th")]

        # ── Extract all data rows ─────────────────────────────────
        rows_data = []
        body_rows = tbl.find_all("tr")
        start_idx = 1 if headers else 0
        for tr in body_rows[start_idx:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if not cells or all(c == "" for c in cells):
                continue

            # ── Capture CSS class metadata from <tr> and cells ────
            # Many data sites encode row status (introduced, native,
            # endemic, etc.) via CSS classes rather than visible text.
            row_classes = " ".join(tr.get("class", []))
            # Also check for semantic classes on individual <td> cells
            cell_tags = tr.find_all(["td", "th"])
            cell_classes = set()
            cell_data_attrs: dict = {}
            for td_tag in cell_tags:
                for cls in td_tag.get("class", []):
                    cell_classes.add(cls)
                # data-* attributes often carry structured info
                for attr_name, attr_val in td_tag.attrs.items():
                    if attr_name.startswith("data-") and isinstance(attr_val, str):
                        cell_data_attrs[attr_name] = attr_val

            # Merge row + cell classes, filter out pure styling noise
            _STYLE_NOISE = frozenset({
                "odd", "even", "alt", "row", "tr", "td", "first", "last",
                "active", "hover", "selected", "highlight", "striped",
                "border", "padding", "center", "left", "right", "top",
                "visible", "hidden", "collapsed", "expanded",
            })
            all_classes = set(row_classes.split()) | cell_classes
            semantic_classes = sorted(
                c for c in all_classes
                if c and c.lower() not in _STYLE_NOISE and not c.startswith("col-")
            )

            if headers:
                row_dict = {}
                for i, h in enumerate(headers):
                    row_dict[h or f"col_{i}"] = cells[i] if i < len(cells) else ""
                for i in range(len(headers), len(cells)):
                    row_dict[f"col_{i}"] = cells[i]
            else:
                row_dict = {f"col_{i}": c for i, c in enumerate(cells)}

            # Attach semantic metadata if any was found
            if semantic_classes:
                row_dict["_row_class"] = " ".join(semantic_classes)
            if cell_data_attrs:
                row_dict["_data"] = cell_data_attrs

            rows_data.append(row_dict)

        if not rows_data:
            continue

        # ── Table context metadata ────────────────────────────────
        # Caption
        caption_tag = tbl.find("caption")
        caption = caption_tag.get_text(strip=True) if caption_tag else ""

        # Nearest heading (walk backwards through previous siblings / parents)
        nearest_heading = ""
        for prev in tbl.find_all_previous(["h1", "h2", "h3", "h4"]):
            nearest_heading = prev.get_text(strip=True)
            break

        # Table id / class for re-targeting with css_selector
        tbl_id = tbl.get("id", "")
        tbl_classes = " ".join(tbl.get("class", []))

        meta = {
            "table_index": idx,
            "headers": headers,
            "caption": caption,
            "nearest_heading": nearest_heading,
            "total_rows": len(rows_data),
            "preview_rows": rows_data[:2],
        }
        if tbl_id:
            meta["id"] = tbl_id
        if tbl_classes:
            meta["class"] = tbl_classes

        # ── Budget check ──────────────────────────────────────────
        chunk = json.dumps(rows_data, ensure_ascii=False)
        if total_chars + len(chunk) > max_chars and all_tables:
            # Over budget — stop adding full tables but keep meta
            truncated = True
            # Add just the metadata (no rows) so model knows what was skipped
            all_tables.append({"meta": meta, "rows": []})
            continue

        total_chars += len(chunk)
        all_tables.append({"meta": meta, "rows": rows_data})

    return {
        "tables": all_tables,
        "tables_found": total_found,
        "tables_returned": sum(1 for t in all_tables if t["rows"]),
        "truncated": truncated,
        "layout_tables_skipped": layout_skipped,
    }


# ── extract_tables ────────────────────────────────────────────────────
def extract_tables(url: str, max_chars: int = 12000, css_selector: str = "",
                   timeout: int = 20) -> dict:
    """Extract HTML tables from a URL as structured JSON arrays.

    Returns a dict with 'ok', 'tables' (list of table dicts with 'meta' and
    'rows'), truncation info, and per-table context metadata.

    Each table dict has:
      meta: {table_index, headers, caption, nearest_heading, total_rows,
             preview_rows, id?, class?}
      rows: list of row-dicts keyed by header names (empty if truncated)

    The function uses the same robust fetch pipeline as fetch_url:
    direct probe → Cloudflare detection → Jina Reader fallback → Wayback.

    Args:
        url: URL to fetch.
        max_chars: Soft cap on total JSON output size.
        css_selector: Optional CSS selector to target specific table(s).
        timeout: Per-request timeout in seconds.
    """
    import httpx
    from bs4 import BeautifulSoup
    import warnings

    domain = _extract_domain(url)
    html_text = None
    final_url = url
    source = "direct"

    # ── Fetch with robustness (mirrors fetch_url strategy) ────────────
    # 1. If domain is blocklisted or known-hostile, skip direct
    if _is_domain_blocked(domain) or domain in _JINA_FIRST_DOMAINS:
        _skip_reason = "blocklisted" if _is_domain_blocked(domain) else "jina-first"
        logger.debug(f"[extract_tables] Domain {domain} {_skip_reason} — going Jina")
    else:
        # Per-domain rate limiting before direct probe
        _domain_rate_wait(domain)
        # Direct probe
        try:
            resp = httpx.get(
                url, headers=_BROWSER_HEADERS,
                timeout=timeout, follow_redirects=True,
            )
            final_url = str(resp.url)

            if resp.status_code < 400:
                blocked, reason, is_cf = _is_blocked(resp.text, resp.status_code)
                if blocked:
                    if is_cf:
                        _block_domain(domain)
                        logger.debug(f"[extract_tables] Cloudflare on {domain} — blocklisted")
                    # Fall through to Jina
                else:
                    html_text = resp.text
            else:
                blocked, reason, is_cf = _is_blocked(resp.text, resp.status_code)
                if is_cf:
                    _block_domain(domain)
                logger.debug(f"[extract_tables] HTTP {resp.status_code} from {url}")
        except Exception as e:
            logger.debug(f"[extract_tables] Direct fetch failed: {e}")

    # 2. Jina Reader fallback — request HTML so we get <table> elements
    #    (text/plain returns Markdown which has no parseable <table> tags)
    #    Skip if URL is already a Jina URL (avoids circular fetch)
    _is_archive_url = "web.archive.org" in domain
    _is_jina_url = "r.jina.ai" in domain or "s.jina.ai" in domain

    if html_text is None and not _is_jina_url:
        jina_text = _jina_reader_fallback(
            url, max_chars=100000, accept="text/html",
            wait_for_selector="table",
        )
        if jina_text and len(jina_text.strip()) > 50:
            html_text = jina_text
            source = "jina_reader"
            logger.debug(f"[extract_tables] Using Jina Reader (HTML) for {url}")

    # 3. Wayback fallback (skip if URL is already a Wayback URL)
    if html_text is None and not _is_archive_url:
        try:
            wb_result = fetch_cached(url=url, max_chars=100000)
            if wb_result.get("ok") and wb_result.get("content", "").strip():
                html_text = wb_result["content"]
                source = "wayback"
                logger.debug(f"[extract_tables] Using Wayback for {url}")
        except Exception:
            pass

    # 4. Last-resort direct fetch for blocklisted/jina-first domains
    #    (skipped above, but Jina+Wayback both failed — try direct anyway)
    if html_text is None and (_is_domain_blocked(domain) or domain in _JINA_FIRST_DOMAINS):
        _domain_rate_wait(domain)
        try:
            resp = httpx.get(
                url, headers=_BROWSER_HEADERS,
                timeout=timeout, follow_redirects=True,
            )
            final_url = str(resp.url)
            if resp.status_code < 400:
                _lr_blocked, _, _ = _is_blocked(resp.text, resp.status_code)
                if not _lr_blocked and len(resp.text.strip()) > 200:
                    html_text = resp.text
                    source = "direct (last-resort)"
                    logger.debug(f"[extract_tables] Last-resort direct succeeded for {url}")
        except Exception as e:
            logger.debug(f"[extract_tables] Last-resort direct failed: {e}")

    if html_text is None:
        return {
            "ok": False, "tables": [], "url": final_url,
            "tables_found": 0, "tables_returned": 0, "truncated": False,
            "reason": f"Could not fetch {url} (direct, Jina, and Wayback all failed)",
            "hint": "Do NOT retry this URL. Search for the same data from a different source.",
        }

    # ── Parse HTML → soup ─────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*XMLParsedAsHTMLWarning.*")
            soup = BeautifulSoup(html_text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        result = _parse_tables_from_soup(soup, max_chars=max_chars, css_selector=css_selector)

        if not result["tables"]:
            # ── Heuristic retry: if direct fetch returned 0 tables but the URL
            #    looks like a data page, the tables may be JS-rendered.  Try
            #    Jina HTML mode before giving up (Jina renders JS).
            _table_url_hints = ("data", "table", "stat", "rank", "list",
                                "score", "result", "standing", "leaderboard",
                                "dashboard", "chart", "report", "index")
            _url_lower = url.lower()
            _looks_like_data_page = any(h in _url_lower for h in _table_url_hints)

            if source == "direct" and _looks_like_data_page:
                logger.debug(f"[extract_tables] 0 tables from direct but URL hints at data — retrying via Jina HTML")
                jina_html = _jina_reader_fallback(
                    url, max_chars=100000, accept="text/html",
                    wait_for_selector="table",
                )
                if jina_html and len(jina_html.strip()) > 200:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*XMLParsedAsHTMLWarning.*")
                        soup2 = BeautifulSoup(jina_html, "html.parser")
                    for tag in soup2(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    result2 = _parse_tables_from_soup(soup2, max_chars=max_chars, css_selector=css_selector)
                    if result2["tables"]:
                        result2["source"] = "jina_reader (table-retry)"
                        result2["ok"] = True
                        result2["url"] = final_url
                        return result2
                    logger.debug(f"[extract_tables] Jina HTML retry also returned 0 tables")

            return {
                "ok": False, "tables": [], "url": final_url,
                "tables_found": result["tables_found"],
                "tables_returned": 0,
                "truncated": False,
                "layout_tables_skipped": result["layout_tables_skipped"],
                "reason": "No data tables found on page (layout-only tables were filtered)",
                "hint": "Try fetch_url with extract='text' instead, or use css_selector to target a specific element.",
                "source": source,
            }

        return {
            "ok": True,
            "tables": result["tables"],
            "tables_found": result["tables_found"],
            "tables_returned": result["tables_returned"],
            "truncated": result["truncated"],
            "layout_tables_skipped": result["layout_tables_skipped"],
            "table_count": result["tables_returned"],
            "url": final_url,
            "source": source,
        }

    except Exception as e:
        return {"ok": False, "tables": [], "url": final_url,
                "reason": f"{type(e).__name__}: {e}"}


# ── wikipedia_lookup ──────────────────────────────────────────────────
def wikipedia_lookup(title: str, section: str = "", max_chars: int = 12000) -> dict:
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
                content = "\n\n".join(content_parts)
                full_content = content
                content = content[:max_chars]
            else:
                content = f"Section '{section}' not found. Available sections: {', '.join(sections_list)}"
                full_content = content
        else:
            full_content = soup.get_text(separator="\n", strip=True)
            content = full_content[:max_chars]

        # Cache full text in _page_text_cache so read_page() works for Wikipedia
        wiki_url = f"https://en.wikipedia.org/wiki/{actual_title.replace(' ', '_')}"
        with _page_text_cache_lock:
            _page_text_cache[wiki_url] = (full_content, time.time())
            # Also cache under the title as passed (common lookup key)
            _page_text_cache[f"wikipedia:{title}"] = (full_content, time.time())

        result = {
            "ok": True,
            "title": actual_title,
            "content": content,
            "_full_length": len(full_content),
            "_cache_url": wiki_url,
            "sections": sections_list,
        }
        if infobox:
            result["infobox"] = infobox
        return result

    except Exception as e:
        return {"ok": False, "content": "", "title": title,
                "reason": f"{type(e).__name__}: {e}"}


# ── fetch_cached (Wayback Machine) ───────────────────────────────────
def fetch_cached(url: str, date: str = "", max_chars: int = 12000) -> dict:
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

        full_content = soup.get_text(separator="\n", strip=True)
        content = full_content[:max_chars]

        # Cache full text so read_page() works for Wayback results
        with _page_text_cache_lock:
            _page_text_cache[url] = (full_content, time.time())
            # Also cache under archive URL for direct lookups
            if archive_url != url:
                _page_text_cache[archive_url] = (full_content, time.time())

        return {
            "ok": True,
            "content": content,
            "_full_length": len(full_content),
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
    from .config import MAX_RECURSION_DEPTH, SUB_AGENT_TURN_BUDGET, get_model_profile, MODEL_NAME

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
                _total_chars = sum(e["content_length"] for e in entries)
                _code_hint = ""
                if _total_chars > 15000:
                    _code_hint = (
                        "⚡ The pre-loaded data is large. Use execute_code to search it efficiently:\n"
                        "  execute_code(code=\"\"\"```python\n"
                        "  import json, re\n"
                        "  data = json.load(open('agent_data.json'))\n"
                        "  for entry in data['entries']:\n"
                        "      matches = re.findall(r'YOUR_PATTERN', entry['content'], re.I)\n"
                        "      if matches: print(f\"{entry['key']}: {matches}\")\n"
                        "  ```\"\"\")\n"
                        "Do NOT try to eyeball-read large entries. Use code to search/filter.\n\n"
                    )
                task = (
                    f"[PRE-LOADED DATA] The file agent_data.json is available in your sandbox. "
                    f"Read it with: import json; data = json.load(open('agent_data.json'))\n"
                    f"It contains {len(entries)} entries ({_total_chars:,} chars total): {file_summary}\n"
                    f"Each entry has 'key', 'content', and 'content_length' fields.\n"
                    f"{_code_hint}"
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


# ── Direct data-file download helper ──────────────────────────────────
# Bypasses Jina/JS rendering for URLs that clearly point to data files.
# Uses a raw httpx.get() with browser headers + redirect following.
# On success, caches the content and returns a preview with guidance.

_DATA_DL_TIMEOUT = 60       # Data files can be large — generous timeout
_DATA_DL_MAX_BYTES = 50_000_000  # 50 MB hard limit
_DATA_FILE_EXTS = (".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls",
                   ".parquet", ".xml", ".ndjson")

def _try_direct_data_download(url: str, max_chars: int = 12000) -> str | None:
    """Attempt a raw HTTP download of a data-file URL.

    Returns a formatted output string on success, or None on failure
    (so the caller can fall through to regular fetch_url).
    """
    import httpx
    from urllib.parse import urlparse
    import hashlib as _hl

    domain = _extract_domain(url)
    _domain_rate_wait(domain)

    try:
        with httpx.stream("GET", url, headers=_BROWSER_HEADERS,
                          timeout=_DATA_DL_TIMEOUT, follow_redirects=True) as resp:
            if resp.status_code >= 400:
                logger.debug(f"[data_download] HTTP {resp.status_code} for {url} — falling back")
                # For 403 on data files, steer toward execute_code instead of
                # letting fetch_url retry (Jina can't render CSVs/JSONs anyway).
                if resp.status_code in (403, 401):
                    _path_lc = urlparse(url).path.lower().split("?")[0]
                    _pandas_reader = (
                        "pd.read_csv" if _path_lc.endswith((".csv", ".tsv")) else
                        "pd.read_json" if _path_lc.endswith((".json", ".jsonl", ".ndjson")) else
                        "pd.read_excel" if _path_lc.endswith((".xlsx", ".xls")) else
                        "pd.read_parquet" if _path_lc.endswith(".parquet") else
                        "pd.read_csv"
                    )
                    return (
                        f"FETCH FAILED: HTTP {resp.status_code} — the server blocked direct download "
                        f"of this data file.\nDo NOT retry this URL with fetch_url. "
                        f"Instead, try loading it programmatically with execute_code:\n"
                        f"  execute_code(code=\"\"\"```python\n"
                        f"  import pandas as pd\n"
                        f"  df = {_pandas_reader}('{url}')\n"
                        f"  print(df.shape)\n"
                        f"  print(df.head(10))\n"
                        f"  ```\"\"\")\n"
                        f"Some servers allow programmatic Python access even when they "
                        f"block browser-style downloads."
                    )
                return None

            # Read content with size limit
            chunks = []
            total = 0
            for chunk in resp.iter_bytes(chunk_size=65536):
                chunks.append(chunk)
                total += len(chunk)
                if total > _DATA_DL_MAX_BYTES:
                    logger.warning(f"[data_download] {url} exceeds {_DATA_DL_MAX_BYTES/1e6:.0f} MB — truncating")
                    break
            raw_bytes = b"".join(chunks)

            ct = resp.headers.get("content-type", "").lower()
            final_url = str(resp.url)

            # Determine file type from content-type or URL extension
            _path = urlparse(url).path.lower().split("?")[0]
            is_csv = "csv" in ct or _path.endswith((".csv", ".tsv"))
            is_json = "json" in ct or _path.endswith((".json", ".jsonl", ".ndjson"))
            is_excel = "spreadsheet" in ct or "excel" in ct or _path.endswith((".xlsx", ".xls"))
            is_parquet = "parquet" in ct or _path.endswith(".parquet")
            is_xml = "xml" in ct or _path.endswith(".xml")

            # Try to decode as text
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = raw_bytes.decode("latin-1")
                except Exception:
                    text = None

            if text is None and not (is_excel or is_parquet):
                # Binary file we can't decode — fall through
                logger.debug(f"[data_download] Binary content from {url}, can't decode — falling back")
                return None

            # ── Cache raw text for read_page / memory ─────────────────
            url_hash = _hl.md5(url.encode()).hexdigest()[:8]

            if text:
                full_length = len(text)
                with _page_text_cache_lock:
                    _page_text_cache[url] = (text, time.time())
                    if final_url != url:
                        _page_text_cache[final_url] = (text, time.time())
                with _url_cache_lock:
                    _url_cache[url] = (text, final_url, resp.status_code, time.time(), ct)

                # ── Build preview ─────────────────────────────────────
                if is_csv:
                    file_label = "CSV"
                    sandbox_name = f"data_{url_hash}.csv"
                    lines = text.split("\n")
                    header = lines[0] if lines else ""
                    n_rows = len(lines) - 1  # approximate
                    preview_lines = lines[:min(6, len(lines))]
                    preview = "\n".join(preview_lines)
                    if n_rows > 5:
                        preview += f"\n... ({n_rows:,} total rows)"
                    hint = (
                        f"import pandas as pd\n"
                        f"df = pd.read_csv('{sandbox_name}')\n"
                        f"print(df.shape)\n"
                        f"print(df.head())"
                    )
                elif is_json:
                    file_label = "JSON"
                    sandbox_name = f"data_{url_hash}.json"
                    preview = text[:max_chars] if full_length > max_chars else text
                    if full_length > max_chars:
                        preview += f"\n... ({full_length:,} chars total)"
                    hint = (
                        f"import json\n"
                        f"data = json.load(open('{sandbox_name}'))\n"
                        f"print(type(data), len(data) if isinstance(data, list) else list(data.keys())[:10])"
                    )
                elif is_xml:
                    file_label = "XML"
                    sandbox_name = f"data_{url_hash}.xml"
                    preview = text[:max_chars] if full_length > max_chars else text
                    if full_length > max_chars:
                        preview += f"\n... ({full_length:,} chars total)"
                    hint = (
                        f"import xml.etree.ElementTree as ET\n"
                        f"tree = ET.parse('{sandbox_name}')\n"
                        f"root = tree.getroot()\n"
                        f"print(root.tag, len(root))"
                    )
                else:
                    file_label = "DATA"
                    sandbox_name = f"data_{url_hash}.txt"
                    preview = text[:max_chars] if full_length > max_chars else text
                    hint = f"text = open('{sandbox_name}').read()"

                output = (
                    f"[DATA FILE DOWNLOADED: {file_label}, {full_length:,} chars]\n\n"
                    f"{preview}\n\n"
                    f"[FULL FILE ({full_length:,} chars) AVAILABLE AS '{sandbox_name}' IN SANDBOX.\n"
                    f" Analyze it with execute_code:\n"
                    f"  execute_code(code=\"\"\"```python\n"
                    f"  {hint}\n"
                    f"  ```\"\"\")]"
                )
                logger.info(
                    f"[data_download] {file_label} from {domain}: "
                    f"{full_length:,} chars → sandbox '{sandbox_name}'"
                )
                return output

            elif is_excel or is_parquet:
                # Binary structured data — encode and provide sandbox file
                import base64 as _b64dl
                ext = "xlsx" if is_excel else "parquet"
                file_label = "Excel" if is_excel else "Parquet"
                sandbox_name = f"data_{url_hash}.{ext}"
                # We can't preview binary, but we can tell the model to use pandas
                output = (
                    f"[DATA FILE DOWNLOADED: {file_label}, {len(raw_bytes):,} bytes]\n\n"
                    f"Binary file — cannot preview inline.\n\n"
                    f"[FULL FILE ({len(raw_bytes):,} bytes) AVAILABLE AS '{sandbox_name}' IN SANDBOX.\n"
                    f" Load it with execute_code:\n"
                    f"  execute_code(code=\"\"\"```python\n"
                    f"  import pandas as pd\n"
                    f"  df = pd.read_{'excel' if is_excel else 'parquet'}('{sandbox_name}')\n"
                    f"  print(df.shape)\n"
                    f"  print(df.head())\n"
                    f"  ```\"\"\")]"
                )
                # Cache as base64 so sandbox injection picks it up
                with _page_text_cache_lock:
                    _b64_str = _b64dl.b64encode(raw_bytes).decode("ascii")
                    _page_text_cache[url] = (_b64_str, time.time())
                logger.info(
                    f"[data_download] {file_label} from {domain}: "
                    f"{len(raw_bytes):,} bytes → sandbox '{sandbox_name}'"
                )
                return output

    except Exception as e:
        logger.debug(f"[data_download] Direct download failed for {url}: {e}")
        return None

    return None

    
def fetch_url_wrapper(**kwargs):
    """Wrapper for fetch_url tool. Returns (output, None).
    
    Smart routing before hitting fetch_url:
    1. Wikipedia URLs → wikipedia_lookup (MediaWiki API, not blocked)
    2. .pdf URLs → read_pdf first (proper PDF extraction)
    Then normal fetch_url with Jina + Wayback fallbacks.
    """
    try:
        url = kwargs.get("url")
        if not url:
            return "ERROR: 'url' parameter is required", None

        # ── Auto-redirect Wikipedia URLs → wikipedia_lookup ───────────
        import re as _re
        wiki_match = _re.match(
            r'https?://(?:\w+\.)?wikipedia\.org/wiki/(.+)',
            url, _re.IGNORECASE,
        )
        if wiki_match:
            from urllib.parse import unquote
            raw_title = unquote(wiki_match.group(1)).replace('_', ' ')
            if '#' in raw_title:
                raw_title = raw_title.split('#')[0]
            logger.info(f"[fetch_url] Wikipedia URL detected — redirecting to wikipedia_lookup(title='{raw_title}')")
            result_str, _ = wikipedia_lookup_wrapper(title=raw_title, max_chars=kwargs.get("max_chars", 12000))
            return f"[Auto-redirected to wikipedia_lookup for reliable Wikipedia access]\n\n{result_str}", None

        # ── Early PDF detection by URL extension ──────────────────────
        from urllib.parse import urlparse
        url_path = urlparse(url).path.lower()
        # Strip ?query from extension check
        _ext_path = url_path.split("?")[0]
        if _ext_path.endswith(".pdf"):
            try:
                logger.info(f"[fetch_url] PDF URL detected — trying read_pdf first: {url}")
                _pdf_result = read_pdf(url=url, max_chars=kwargs.get("max_chars", 12000))
                pdf_text = _pdf_result.get("content", "") if isinstance(_pdf_result, dict) else str(_pdf_result)
                if pdf_text and len(pdf_text.strip()) > 30:
                    return f"[PDF detected by URL — extracted with read_pdf]\n\n{pdf_text}", None
            except Exception as pdf_err:
                logger.debug(f"[fetch_url] read_pdf failed for {url}: {pdf_err} — falling through to normal fetch")

        # ── Early data-file detection: direct download to sandbox ─────
        # When the URL clearly points to a data file (CSV, JSON, XLSX,
        # etc.), attempt a raw httpx.get() first.  This avoids the
        # Jina/JS-rendering pipeline which can timeout or 403 on direct
        # file downloads.  On success, cache the raw content and return
        # a preview + sandbox injection notice.
        _is_data_url = _ext_path.endswith(_DATA_FILE_EXTS) or "download=1" in url.lower()
        if _is_data_url:
            _dl_result = _try_direct_data_download(url, kwargs.get("max_chars", 12000))
            if _dl_result is not None:
                return _dl_result, None

        max_chars = kwargs.get("max_chars", 12000)
        extract = kwargs.get("extract", "text")
        css_selector = kwargs.get("css_selector", "")
        result = fetch_url(url=url, max_chars=max_chars, extract=extract, css_selector=css_selector)

        # ── Track cross-agent fetch count + content fingerprint ────────
        _content_for_hash = result.get("content", "") if result["ok"] else ""
        _fetch_count, _content_unchanged = _record_url_access(url, _content_for_hash)

        if result["ok"]:
            output = result["content"]

            # ── Fix A: Detect Jina-rendered API error pages ───────────
            # Jina returns ok=True even when the target returned an HTTP
            # error.  It embeds "Warning: Target URL returned error NNN"
            # in the text.  If the actual content is a JSON API error,
            # we should surface it as a clear failure, not success.
            # Check both status_code (when inherited from direct) and
            # the content text (Jina may report 200 but embed the warning).
            _may_be_api_err = (
                result.get("source") == "jina_reader"
                and (
                    result.get("status_code", 200) >= 400
                    or "Target URL returned error" in output[:500]
                )
            )
            if _may_be_api_err:
                _api_err = _detect_jina_api_error(output)
                if _api_err:
                    _err_code, _err_label = _api_err
                    err_result = {
                        "ok": False, "content": "", "url": url,
                        "blocked": False,
                        "reason": f"API error (HTTP {_err_code}): {_err_label}",
                        "status_code": _err_code, "retries": result.get("retries", 0),
                        "source": "jina_reader",
                        "hint": _status_code_hint(_err_code, _err_label),
                    }
                    return json.dumps(err_result, indent=2), None

            if result["retries"] > 0:
                output = f"[Succeeded after {result['retries']} retry(ies), final URL: {result['url']}]\n\n{output}"
            if result.get("source") == "jina_reader":
                output = f"[Rendered via Jina Reader]\n\n{output}"

            # ── Repetition warning: other agents already saw this page ─
            if _fetch_count > 1:
                _rep_note = (
                    f"\n\n[REPEATED URL ({_fetch_count}x): This URL has been fetched "
                    f"{_fetch_count} times across agents in this session"
                )
                if _content_unchanged:
                    _rep_note += (
                        " and returned identical content each time. "
                        "Other agents already tried to extract information from this "
                        "page and could not get more than what you see above"
                    )
                _rep_note += (
                    ". Do NOT spend additional effort on this URL — search for "
                    "the information from a different source instead.]"
                )
                output += _rep_note

            # ── Structured data hint ─────────────────────────────────
            if result.get("_is_structured"):
                full_len = result.get("_full_length", 0)
                shown_len = len(result["content"])
                ct = result.get("_content_type", "structured data")
                _was_previewed = "⚠ PREVIEW" in result["content"][:80]
                if _was_previewed:
                    # Data was too large — model sees only a preview
                    output += (
                        f"\n\n[STRUCTURED DATA: {ct}, {full_len:,} chars total "
                        f"(only a preview is shown above). Full data is cached and will "
                        f"be auto-loaded as a file in your sandbox when you call execute_code.\n"
                        f"Options:\n"
                        f"  1. execute_code(code=\"import json; data = json.loads(open('data_XXXX.txt').read()); ...\")\n"
                        f"     The filename will appear in a follow-up notice.\n"
                        f"  2. conduct_research(task='Analyze the {ct} data from {url[:80]}...', "
                        f"memory_keys=['<key from [Stored → ...]>']) to delegate to a sub-agent.\n"
                        f"Do NOT eyeball-parse large structured data — use code or delegate.]"
                    )
                else:
                    # Full data returned inline — just tag it
                    output += (
                        f"\n\n[STRUCTURED DATA: {ct}, {shown_len:,} chars. "
                        f"Use execute_code() to parse/filter this data programmatically, "
                        f"or pass to a sub-agent via conduct_research(memory_keys=['<key from [Stored → ...]>'])."
                        f"]"
                    )

            # ── Truncation hint: tell the agent full page is cached ───
            full_len = result.get("_full_length", 0)
            shown_len = len(result["content"])
            if full_len > shown_len + 200 and not result.get("_is_structured"):
                output += (
                    f"\n\n[PAGE TRUNCATED: showing {shown_len:,} of {full_len:,} chars. "
                    f"Full page is cached — call read_page(url=\"{url}\", offset={shown_len}) "
                    f"to read more, or spawn_agent with this URL for focused analysis.]"
                )

            return output, None
        else:
            # ── B: Add Wikipedia hint on 403 from Wikipedia domains ───
            if result.get("status_code") == 403 and "wikipedia" in url.lower():
                result["hint"] = (
                    "Wikipedia blocked this request (403 Forbidden). "
                    "Use wikipedia_lookup(title='Article Title') instead — "
                    "it uses the MediaWiki API which is not blocked."
                )
            # ── Return a clear, human-readable error — not raw JSON ───
            # The model often just passes through opaque JSON errors into
            # its draft ("blocked", etc.). Give it a sentence it can act on.
            _sc = result.get("status_code", 0)
            _reason = result.get("reason", "")
            _hint = result.get("hint", "")
            _src_url = result.get("url", url)
            _parts = [f"FETCH FAILED: {_src_url}"]
            if _sc:
                _parts.append(f"HTTP {_sc}")
            if _reason:
                _parts.append(_reason)
            _error_line = " — ".join(_parts)
            if _hint:
                _error_line += f"\n{_hint}"
            return _error_line, None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def read_page(url: str, offset: int = 0, max_chars: int = 12000) -> str:
    """Read a section of a previously-fetched page from the parsed-text cache.

    Pages are cached automatically when fetch_url succeeds.  This tool lets
    you paginate through the full content without re-fetching or re-parsing.

    Args:
        url: The same URL passed to the earlier fetch_url call.
        offset: Character offset to start reading from (0-based).
        max_chars: Maximum characters to return (default: 12000).

    Returns:
        The requested text slice, or an error if the page isn't cached.
    """
    with _page_text_cache_lock:
        cached = _page_text_cache.get(url)
    if not cached:
        return f"ERROR: No cached page for '{url}'. Call fetch_url(url=...) first."

    full_text, ts = cached
    if time.time() - ts > _URL_CACHE_TTL:
        with _page_text_cache_lock:
            _page_text_cache.pop(url, None)
        return f"ERROR: Cached page for '{url}' has expired. Call fetch_url(url=...) again."

    total = len(full_text)
    if offset >= total:
        return f"ERROR: offset {offset} is past end of page ({total:,} chars)."

    chunk = full_text[offset : offset + max_chars]
    end_pos = offset + len(chunk)
    remaining = total - end_pos

    header = f"[Page: {url}] chars {offset:,}–{end_pos:,} of {total:,}"
    if remaining > 0:
        header += f" ({remaining:,} remaining — read_page(offset={end_pos}) for next chunk)"

    return f"{header}\n\n{chunk}"


def read_page_wrapper(**kwargs):
    """Wrapper for read_page tool. Returns (output, None)."""
    try:
        url = kwargs.get("url")
        if not url:
            return "ERROR: 'url' parameter is required", None
        offset = int(kwargs.get("offset", 0))
        max_chars_val = int(kwargs.get("max_chars", 12000))
        return read_page(url=url, offset=offset, max_chars=max_chars_val), None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def read_pdf_wrapper(**kwargs):
    """Wrapper for read_pdf tool. Returns (output, None)."""
    try:
        url = kwargs.get("url")
        if not url:
            return "ERROR: 'url' parameter is required", None

        max_chars = kwargs.get("max_chars", 12000)
        result = read_pdf(url=url, max_chars=max_chars)

        if not result.get("ok", False):
            return json.dumps(result, indent=2), None

        output = result["content"]
        shown = len(result["content"])
        full = result.get("full_length", shown)
        pages = result.get("page_count", "?")

        if full > shown + 200:
            output += (
                f"\n\n[PDF TRUNCATED: showing {shown:,} of {full:,} chars "
                f"({pages} pages). Full text is cached — use "
                f"read_page(url=\"{url}\", offset={shown}) to continue reading, "
                f"or use execute_code() to search/parse the full text programmatically.]"
            )

        return output, None
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
            out_dict = {
                "table_count": result["table_count"],
                "tables_found": result["tables_found"],
                "tables_returned": result["tables_returned"],
                "truncated": result["truncated"],
                "url": result["url"],
                "tables": result["tables"],
            }
            if result.get("source") and result["source"] != "direct":
                out_dict["source"] = result["source"]
            if result.get("layout_tables_skipped"):
                out_dict["layout_tables_skipped"] = result["layout_tables_skipped"]
            if result["truncated"]:
                out_dict["hint"] = (
                    f"Output was truncated — showed {result['tables_returned']} of "
                    f"{result['tables_found']} tables. Use css_selector to target "
                    "a specific table. Truncated tables include metadata (headers, "
                    "caption, heading) but no rows — re-fetch with their css_selector."
                )
            output = json.dumps(out_dict, ensure_ascii=False, indent=2)
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
        max_chars = kwargs.get("max_chars", 12000)
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

            output = "\n".join(parts)

            # Truncation notice
            shown = len(result["content"])
            full = result.get("_full_length", shown)
            cache_url = result.get("_cache_url", "")
            if full > shown + 200 and cache_url:
                output += (
                    f"\n\n[ARTICLE TRUNCATED: showing {shown:,} of {full:,} chars. "
                    f"Full text is cached — use read_page(url=\"{cache_url}\", "
                    f"offset={shown}) for more, or pass a section= parameter to "
                    f"target a specific section.]"
                )

            return output, None
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
        max_chars = kwargs.get("max_chars", 12000)
        result = fetch_cached(url=url, date=date, max_chars=max_chars)

        if result["ok"]:
            header = f"[Wayback Machine snapshot: {result.get('snapshot_date', '?')}]\n"
            header += f"[Original URL: {result.get('original_url', url)}]\n\n"
            output = header + result["content"]

            # Truncation notice
            shown = len(result["content"])
            full = result.get("_full_length", shown)
            if full > shown + 200:
                output += (
                    f"\n\n[PAGE TRUNCATED: showing {shown:,} of {full:,} chars. "
                    f"Full page is cached — use read_page(url=\"{url}\", "
                    f"offset={shown}) for more, or use execute_code() to "
                    f"search/parse the full text programmatically.]"
                )

            return output, None
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
    elif tool_name == "read_page":
        return read_page_wrapper(**tool_args)
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
            "WHEN TO USE:\n"
            "- SEARCH truncated pages: when a page is truncated, the full text is auto-loaded "
            "as a .txt file in your sandbox (e.g. 'page_abc123.txt'). Open it and search with regex.\n"
            "- COMPUTE derived values: averages, percentages, date math, unit conversions.\n"
            "- PARSE structured data: load JSON/CSV with pandas or json module, filter rows, aggregate.\n"
            "- ANALYZE agent_data.json: when pre-loaded data is available, use code to search/filter it.\n"
            "- CROSS-REFERENCE: compare data from multiple fetched pages programmatically.\n\n"
            "PATTERNS:\n"
            "  # Search a truncated page (full text auto-loaded in sandbox):\n"
            "  import re\n"
            "  text = open('page_abc123.txt').read()  # filename from truncation notice\n"
            "  matches = re.findall(r'population[:\\s]+(\\d[\\d,]+)', text, re.I)\n"
            "  print(matches)\n\n"
            "  # Parse pre-loaded data from orchestrator:\n"
            "  import json\n"
            "  data = json.load(open('agent_data.json'))\n"
            "  for entry in data['entries']:\n"
            "      if 'GDP' in entry['content']: print(entry['key'], entry['content'][:200])\n\n"
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
            "persistent cookies. Falls back to Jina Reader for JS-heavy pages, "
            "then Wayback Machine if all else fails.\n\n"
            "Returns page text on success. On failure (site blocks, 403, etc.) returns "
            "a structured error with blocked=true and a hint to search elsewhere.\n"
            "DO NOT retry a URL that returns blocked=true — pivot to a different source.\n\n"
            "TABLE EXTRACTION: Set extract='table' to get HTML tables as pipe-delimited rows. "
            "This is a good fallback if extract_tables fails or returns no tables, since "
            "fetch_url has a full retry chain (Jina + Wayback) that extract_tables lacks.\n\n"
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
                    "description": "Maximum characters of extracted text to return (default: 12000).",
                    "default": 12000
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
            "execute_code for filtering, sorting, or computation.\n\n"
            "FALLBACK: If this tool returns no tables or the site blocks access, "
            "try fetch_url(url=..., extract='table') instead — it has a stronger "
            "retry chain (Jina Reader + Wayback Machine) that can reach more sites."
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
                    "description": "Maximum number of characters of extracted text to return (default: 12000).",
                    "default": 12000
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
                    "description": "Maximum characters of text to return (default: 12000).",
                    "default": 12000
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
                    "description": "Maximum characters of extracted text to return (default: 12000).",
                    "default": 12000
                }
            },
            "required": ["url"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "read_page",
        "description": (
            "Read a section of a previously-fetched page from cache. "
            "Pages are cached automatically when fetch_url succeeds.\\n\\n"
            "Use this when fetch_url returned a [PAGE TRUNCATED] notice and you need "
            "to read more of the page. Supports pagination via the offset parameter.\\n\\n"
            "This is MUCH cheaper than re-calling fetch_url — no network request, instant response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL previously fetched with fetch_url."
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start reading from (default: 0). Use the offset from the truncation notice.",
                    "default": 0
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 12000).",
                    "default": 12000
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
            "name": "update_draft",
            "description": (
                "Save a full snapshot of your current draft answer. Call this tool after each "
                "major research wave to capture your evolving understanding of the answer.\n\n"
                "WHEN TO CALL:\n"
                "  1. After the first wave of sub-agents returns — write an initial draft\n"
                "  2. After gap-filling research — update with new findings\n"
                "  3. Before calling final_answer — save your polished version\n\n"
                "The draft is your safety net: if you run out of turns before calling "
                "final_answer, the system will use your latest draft. Write each draft as "
                "a COMPLETE, self-contained answer (not a diff or notes)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "Your complete draft answer. Write it as if it were the final answer — "
                            "well-structured, cited, and directly addressing the question. "
                            "Each call REPLACES the previous draft."
                        )
                    }
                },
                "required": ["content"]
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
            "name": "think",
            "description": (
                "Pause and reflect on your progress. Use this after every search or "
                "fetch to assess what you found before taking the next action.\n\n"
                "WHEN TO CALL:\n"
                "  - After search_web: What results look promising? What's missing?\n"
                "  - After fetch_url: Did I find the specific data I need?\n"
                "  - After execute_code: Did the computation succeed? Is the result reasonable?\n"
                "  - Before final_answer: Is my answer complete and well-sourced?\n\n"
                "This costs nothing — it just helps you stay organized and avoid "
                "wasting turns on unproductive searches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your assessment: what you found, what's missing, what to do next."
                    }
                },
                "required": ["thought"]
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


# ── Root-only tools ──────────────────────────────────────────────────────
# These are the ONLY tools the root orchestrator sees.  They are virtual
# tools — intercepted in agent.py before reaching dispatch_tool_call.
# Sub-agents continue to use `TOOLS` as before.

ROOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "conduct_research",
            "description": (
                "Delegate a research task to a specialized sub-agent. "
                "The sub-agent gets its own fresh context, tools (search, fetch, "
                "code execution, etc.), and turn budget. It performs the research "
                "and returns ONLY its final result — keeping your context clean.\n\n"
                "WRITE SELF-CONTAINED TASKS:\n"
                "The sub-agent has NO context from your conversation. The task "
                "string is ALL it gets. Include:\n"
                "  - Exactly what to find or compute\n"
                "  - Any background context it needs\n"
                "  - What format to return results in\n\n"
                "PASSING DATA VIA memory_keys:\n"
                "When tool outputs are stored in memory (you see [Stored → key]), "
                "pass them to the sub-agent via memory_keys=['key1','key2']. "
                "The data is pre-loaded as agent_data.json in the sub-agent's sandbox.\n\n"
                "PATTERN FOR MULTI-ITEM TASKS:\n"
                "  conduct_research(task='[detailed task for item 1]')\n"
                "  conduct_research(task='[detailed task for item 2]')\n"
                "  → Review results → refine_draft with findings"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear, self-contained description of the research task."
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional background context or data the sub-agent needs."
                        )
                    },
                    "memory_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Memory keys to pre-load into the sub-agent's sandbox "
                            "as agent_data.json. Example: ['search_t1_query', 'page_t3_url']"
                        )
                    },
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refine_draft",
            "description": (
                "Write or update your draft report. Each call REPLACES the entire "
                "draft with the new content you provide.\n\n"
                "Your draft is a living document — the central artifact of your "
                "research session. Write it as a complete, self-contained answer "
                "to the original question every time.\n\n"
                "WHEN TO CALL:\n"
                "  1. After the first wave of research returns — write Draft v1\n"
                "  2. After each round of gap-filling research — update with new findings\n"
                "  3. When you're satisfied — call research_complete to publish it\n\n"
                "IMPORTANT: You cannot call research_complete until you have a draft. "
                "Your draft IS your answer — research_complete simply publishes it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "Your complete draft answer. Write it as if it were the "
                            "final answer — well-structured, cited, and directly "
                            "addressing the question. Each call REPLACES the previous draft."
                        )
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_draft",
            "description": (
                "Read a previous draft version, list all versions, or view "
                "verifier feedback on a rejected draft.\n\n"
                "USE CASES:\n"
                "  - read_draft(list_versions=true): see all versions with status\n"
                "  - read_draft(version=2): read draft v2 in full\n"
                "  - read_draft(version=2, include_feedback=true): read v2 + "
                "the verifier's rejection feedback\n"
                "  - read_draft(include_feedback=true): latest draft + feedback\n\n"
                "Your current draft is already shown to you each turn — use this "
                "tool to review PREVIOUS versions or verifier feedback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {
                        "type": "integer",
                        "description": "Draft version number to read (1-based). Omit for latest."
                    },
                    "list_versions": {
                        "type": "boolean",
                        "description": "If true, return a compact list of all draft versions instead of content."
                    },
                    "include_feedback": {
                        "type": "boolean",
                        "description": "If true, append the verifier's rejection feedback (if any) for the requested version."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "research_complete",
            "description": (
                "Signal that your research is done and publish your draft as the "
                "final answer. This tool takes NO content — it reads your latest "
                "draft and delivers it.\n\n"
                "PREREQUISITES:\n"
                "  - You must have called refine_draft at least once\n"
                "  - Your draft should be a complete, well-cited answer\n\n"
                "Before calling this, review your draft one last time:\n"
                "  - Does it answer every part of the question?\n"
                "  - Are claims supported by research findings?\n"
                "  - Is it well-structured with citations?"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Record your reasoning, planning, or reflection. Use this tool "
                "when you need to think through a problem, plan your next research "
                "steps, or reflect on findings before acting.\n\n"
                "This is a thinking-out-loud tool — your thought is recorded and "
                "returned to you. It costs nothing and helps you stay organized.\n\n"
                "USE FOR:\n"
                "  - Planning which research tasks to delegate next\n"
                "  - Analyzing what gaps remain in your draft\n"
                "  - Deciding whether your draft is ready to publish\n"
                "  - Reflecting on conflicting findings"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning, plan, or reflection."
                    }
                },
                "required": ["thought"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_webpage",
            "description": (
                "Fetch a URL and produce a focused, LLM-generated summary. "
                "Use when you already know a specific URL and want distilled "
                "information without spawning a full research sub-agent.\n\n"
                "This is a lightweight tool (single fetch + single LLM call) — "
                "much faster than conduct_research for targeted page reads.\n\n"
                "USE WHEN:\n"
                "  - A sub-agent returned a URL you want to read in more detail\n"
                "  - You need specific data from a known page\n"
                "  - You want a focused extraction (e.g. 'revenue figures only')\n\n"
                "DO NOT USE WHEN:\n"
                "  - You need to search for pages (use conduct_research instead)\n"
                "  - You need multi-step research (search → fetch → analyze)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch and summarize."
                    },
                    "focus": {
                        "type": "string",
                        "description": (
                            "What to focus on when summarizing. Guides the LLM to "
                            "extract specific information. E.g. 'revenue figures for "
                            "2024', 'main arguments about climate policy', 'biographical "
                            "details and career timeline'."
                        )
                    }
                },
                "required": ["url"]
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
                "Call with a tool_name to get the full JSON schema for that specific tool."
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
