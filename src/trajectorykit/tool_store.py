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


# Define tool implementations
def get_current_time() -> str:
    """Get the current date and time"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_numbers(a: float, b: float) -> str:
    """Add two numbers together"""
    result = a + b
    return f"{a} + {b} = {result}"


# ── Search backend selection ─────────────────────────────────────────────
# Set SEARCH_BACKEND env var to switch: "serper" (default) or "serpapi"
# Each backend needs its own API key:
#   - serper:  SERPER_API_KEY  (from serper.dev)
#   - serpapi: SERP_API_KEY    (from serpapi.com)


SEARCH_TIMEOUT = 25   # seconds — complex quoted queries need more time
MAX_SEARCH_RETRIES = 2
DEFAULT_NUM_SEARCHES = 10


def _search_serper(q: str, num_results: int = DEFAULT_NUM_SEARCHES) -> str:
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


def _search_serpapi(q: str, num_results: int = DEFAULT_NUM_SEARCHES) -> str:
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


def _search_ddg(q: str, num_results: int = DEFAULT_NUM_SEARCHES) -> str:
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


def search_web(q: str, num_results: int = DEFAULT_NUM_SEARCHES) -> str:
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
              max_retries: int = 2, timeout: int = 20) -> dict:
    """Fetch a web page with browser-grade headers, retries, and Jina fallback.

    3-tier strategy:
      1. Direct fetch with browser headers + persistent cookies
      2. Retry with exponential backoff (if blocked / transient error)
      3. Jina Reader fallback (JS rendering, last resort)

    Args:
        url: URL to fetch.
        max_chars: Max characters of extracted text to return.
        extract: "text" for cleaned page text, "table" for pipe-delimited tables.
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
    reader = pypdf.PdfReader(io.BytesIO(resp.content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text[:max_chars]

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
    _shell=None,
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
        _shell=_shell,           # share parent's persistent shell
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
    completion: str,
    stdin: Optional[str] = '',
    compile_timeout: int = 10,
    run_timeout: int = 15,
    memory_limit_mb: int = 512,
    language: str = "python",
    files: Optional[dict[str, str]] = None,      # filename -> base64 content
    fetch_files: Optional[list[str]] = None,      # list of filenames to return
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    
    from .config import SANDBOX_FUSION_URL

    code = completion
    if "```python" in completion:
        code = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            code = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in code:
                first_line, rest = code.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    code = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]
    
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

def get_current_time_wrapper(**kwargs):
    """Wrapper for get_current_time tool. Returns (output, None)."""
    try:
        result = get_current_time()
        return f"Current time: {result}", None
    except Exception as e:
        return f"ERROR: {str(e)}", None

def add_numbers_wrapper(**kwargs):
    """Wrapper for add_numbers tool. Returns (output, None)."""
    try:
        a = kwargs.get("a")
        b = kwargs.get("b")
        
        if a is None or b is None:
            return "ERROR: Both 'a' and 'b' parameters are required", None
        
        result = add_numbers(a, b)
        return result, None
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

def spawn_agent_wrapper(_depth: int = 0, _model: Optional[str] = None, _reasoning_effort: Optional[str] = None, _sandbox_files: Optional[dict] = None, _shell=None, **kwargs):
    """Wrapper for spawn_agent tool. Injects _depth, _model, _reasoning_effort from the parent dispatch loop.
    Returns (output_str, child_trace) where child_trace is an EpisodeTrace."""
    try:
        task = kwargs.get("task")
        if not task:
            return "ERROR: 'task' parameter is required", None

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
            _sandbox_files=_sandbox_files,
            _shell=_shell,
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
    """Wrapper for fetch_url tool. Returns (output, None)."""
    try:
        url = kwargs.get("url")
        if not url:
            return "ERROR: 'url' parameter is required", None

        max_chars = kwargs.get("max_chars", 8000)
        extract = kwargs.get("extract", "text")
        result = fetch_url(url=url, max_chars=max_chars, extract=extract)

        if result["ok"]:
            output = result["content"]
            if result["retries"] > 0:
                output = f"[Succeeded after {result['retries']} retry(ies), final URL: {result['url']}]\n\n{output}"
            if result.get("source") == "jina_reader":
                output = f"[Rendered via Jina Reader]\n\n{output}"
            return output, None
        else:
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


def sandbox_shell_wrapper(_shell=None, **kwargs):
    """Wrapper for sandbox_shell tool. Runs a command in a persistent shell session.
    Returns (output_str, None)."""
    try:
        command = kwargs.get("command")
        if not command:
            return "ERROR: 'command' parameter is required", None
        if _shell is None:
            return "ERROR: No sandbox shell session available", None

        timeout = kwargs.get("timeout", 30)
        result = _shell.run(command, timeout=timeout)

        # Format output like a terminal
        output = f"Exit Code: {result.exit_code}\n"
        output += f"Duration: {result.duration:.2f}s\n"
        if result.timed_out:
            output += "⚠️ TIMED OUT\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}"
        if not result.stdout and not result.stderr:
            output += "(no output)"

        return output.strip(), None
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}", None


def dispatch_tool_call(tool_name: str, tool_args: dict, _depth: int = 0, model: Optional[str] = None, reasoning_effort: Optional[str] = None, _sandbox_files: Optional[dict] = None, _shell=None):
    """Route tool calls to appropriate wrapper function.
    
    All wrappers return (output_str, child_trace_or_None).
    
    Args:
        tool_name: Name of the tool to call
        tool_args: Arguments the model provided for the tool
        _depth: Current recursion depth (injected by the agent loop, invisible to the model)
        model: Model name to propagate to sub-agents
        reasoning_effort: Reasoning effort level to propagate to sub-agents
        _sandbox_files: Files to auto-load into the sandbox for synthesis sub-agents.
                        When set, signals synthesis mode. Dict of {filename: base64_content}.
                        If a shell is available, files are uploaded to the shell workspace
                        instead; this dict is kept as a mode flag + fallback.
    
    Returns:
        Tuple of (output_string, child_trace) where child_trace is an
        EpisodeTrace if the tool was spawn_agent, otherwise None.
    """
    if tool_name == "sandbox_shell":
        return sandbox_shell_wrapper(_shell=_shell, **tool_args)
    elif tool_name == "execute_code":
        # Legacy: still routed for backward compatibility / synthesis sub-agent
        if _sandbox_files:
            model_files = tool_args.get("files") or {}
            merged = {**_sandbox_files, **model_files}
            tool_args = {**tool_args, "files": merged}
        return execute_code_wrapper(**tool_args)
    elif tool_name == "get_current_time":
        return get_current_time_wrapper(**tool_args)
    elif tool_name == "add_numbers":
        return add_numbers_wrapper(**tool_args)
    elif tool_name == "search_web":
        return search_web_wrapper(**tool_args)
    elif tool_name == "spawn_agent":
        return spawn_agent_wrapper(_depth=_depth, _model=model, _reasoning_effort=reasoning_effort, _sandbox_files=_sandbox_files, _shell=_shell, **tool_args)
    elif tool_name == "final_answer":
        return final_answer_wrapper(**tool_args)
    elif tool_name == "search_available_tools":
        return search_available_tools_wrapper(**tool_args)
    elif tool_name == "fetch_url":
        return fetch_url_wrapper(**tool_args)
    elif tool_name == "read_pdf":
        return read_pdf_wrapper(**tool_args)
    else:
        return f"ERROR: Unknown tool '{tool_name}'", None


# ── Legacy execute_code schema (kept for synthesis sub-agent) ─────────────
EXECUTE_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute code in a sandboxed environment. "
            "Code must be in a markdown block (```python ...```). "
            "Returns stdout, stderr, exit status, and any requested output files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "completion": {
                    "type": "string",
                    "description": "Code to execute in a markdown code block."
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (default: 'python').",
                    "default": "python",
                    "enum": SUPPORTED_LANGUAGES
                },
                "files": {
                    "type": "object",
                    "description": "Input files as {filename: base64_string}.",
                    "additionalProperties": {"type": "string"},
                    "default": {}
                },
                "fetch_files": {
                    "type": "array",
                    "description": "Filenames to retrieve after execution.",
                    "items": {"type": "string"},
                    "default": []
                },
                "run_timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default: 15).",
                    "default": 15
                }
            },
            "required": ["completion"]
        }
    }
}

# Define the tool schemas for vLLM
TOOLS = [
{
    "type": "function",
    "function": {
        "name": "sandbox_shell",
        "description": (
            "Run a shell command in a persistent Linux sandbox (Apptainer container). "
            "State persists across calls within this conversation: working directory, "
            "environment variables, installed packages, and created files all carry over.\n\n"
            "USE THIS FOR:\n"
            "- Running Python/Node/bash scripts and commands\n"
            "- Installing packages (pip install, apt-get, npm)\n"
            "- Iterative development: write code → run → inspect output → fix → re-run\n"
            "- Data processing: download files, parse, transform, analyze\n"
            "- Any multi-step workflow that builds on previous results\n\n"
            "TIPS:\n"
            "- Write Python scripts with: cat > script.py << 'EOF'\n...\nEOF\n"
            "  then run with: python3 script.py\n"
            "- For long scripts, write to a file first, then execute\n"
            "- Files persist in /workspace/ across calls\n"
            "- pip packages stay installed for the session"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (runs in bash)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30). Increase for long-running tasks.",
                    "default": 30
                }
            },
            "required": ["command"]
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
            "DO NOT retry a URL that returns blocked=true — pivot to a different source."
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
            "name": "get_current_time",
            "description": "Get the current date and time in YYYY-MM-DD HH:MM:SS format",
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
            "name": "add_numbers",
            "description": "Add two numbers together and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number to add"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number to add"
                    }
                },
                "required": ["a", "b"]
            }
        }
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
                "PATTERN FOR MULTI-ITEM TASKS:\n"
                "1. spawn_agent(task='[detailed task for item 1, specify return format]')\n"
                "2. spawn_agent(task='[detailed task for item 2, specify return format]')\n"
                "3. Collect results, then use sandbox_shell yourself to visualize/synthesize"
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
