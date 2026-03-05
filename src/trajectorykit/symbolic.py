"""
Symbolic references for tool outputs at the root orchestrator level.

Instead of dumping full tool output into the LLM context, outputs exceeding
a threshold are replaced with a compact summary + a MemoryStore key.  The
full data remains in MemoryStore for synthesis and sub-agent access.

This dramatically reduces context consumption at the orchestrator level:
  - search_web: 3K → ~400 chars
  - fetch_url:  3K → ~400 chars
  - execute_code: 13K → ~300 chars
  - spawn_agent: already compact, passed through unchanged
"""

import re
from typing import Optional


# ── Configuration ──────────────────────────────────────────────────────
SYMBOLIC_THRESHOLD = 500      # chars — outputs shorter than this pass through
SUMMARY_TARGET = 400          # target length for summaries


def make_symbolic(
    tool_name: str,
    tool_args: dict,
    output: str,
    memory_key: str,
) -> str:
    """Compress a tool output into a compact summary + memory reference.

    Returns the original output unchanged if it is below SYMBOLIC_THRESHOLD.
    """
    if len(output) <= SYMBOLIC_THRESHOLD:
        return output

    summary = _summarize(tool_name, tool_args, output)

    return (
        f"[Stored → {memory_key}]\n"
        f"{summary}\n"
        f"({len(output):,} chars in memory — "
        f"use read_page(url=..., offset=...) to paginate, "
        f"or spawn_agent(memory_keys=['{memory_key}']) for deep analysis)"
    )


# ── Per-tool summarization strategies ─────────────────────────────────

def _summarize(tool_name: str, tool_args: dict, output: str) -> str:
    """Dispatch to tool-specific summarizer."""
    fn = _SUMMARIZERS.get(tool_name, _summarize_generic)
    return fn(tool_args, output)


def _summarize_search_web(args: dict, output: str) -> str:
    """Keep first 3 result titles + URLs + lead snippet sentence."""
    query = args.get("q", "")
    lines = output.split("\n")
    results = []
    current: dict = {}

    for line in lines:
        line_s = line.strip()
        if line_s.startswith("Title:"):
            if current:
                results.append(current)
            current = {"title": line_s[6:].strip()}
        elif line_s.startswith("URL:") or line_s.startswith("Link:"):
            # handle both "URL:" and "Link:" prefixes
            current["url"] = line_s.split(":", 1)[1].strip()
        elif line_s.startswith("Snippet:"):
            # first sentence only
            snippet = line_s[8:].strip()
            first_sent = re.split(r"(?<=[.!?])\s", snippet, maxsplit=1)[0]
            current["snippet"] = first_sent[:120]

    if current:
        results.append(current)

    top = results[:3]
    parts = [f'Search: "{query}"']
    for i, r in enumerate(top, 1):
        title = r.get("title", "?")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        part = f"  {i}. {title}"
        if url:
            part += f"\n     {url}"
        if snippet:
            part += f"\n     {snippet}"
        parts.append(part)

    total = len(results)
    if total > 3:
        parts.append(f"  ... and {total - 3} more results")

    return "\n".join(parts)


def _summarize_fetch_url(args: dict, output: str) -> str:
    """First ~350 chars of extracted text."""
    url = args.get("url", "")
    # Strip leading metadata lines (e.g. "Extracted text from ...")
    text = output
    for prefix in ("Extracted text from", "Content from", "Page text"):
        if text.lower().startswith(prefix.lower()):
            nl = text.find("\n")
            if nl > 0:
                text = text[nl + 1:]
            break
    text = text.strip()
    head = text[:350]
    if len(text) > 350:
        # trim to last word boundary
        cut = head.rfind(" ")
        if cut > 200:
            head = head[:cut]
        head += " …"
    return f"Fetched {url}\n{head}"


def _summarize_execute_code(args: dict, output: str) -> str:
    """Keep tail of stdout (results are usually at the end)."""
    # Split by exit code line if present
    lines = output.split("\n")

    # Find exit code line
    exit_line = ""
    for i, line in enumerate(lines):
        if line.strip().startswith("Exit code:") or line.strip().startswith("exit code:"):
            exit_line = line.strip()
            break

    # Take last 250 chars of content (skip exit code lines at end)
    content_lines = [l for l in lines if not l.strip().startswith("Exit code:")]
    content = "\n".join(content_lines).strip()

    tail = content[-250:] if len(content) > 250 else content
    if len(content) > 250:
        # Trim to first newline to avoid mid-word
        nl = tail.find("\n")
        if nl > 0 and nl < 50:
            tail = tail[nl + 1:]
        tail = "… " + tail

    parts = []
    if exit_line:
        parts.append(exit_line)
    parts.append(tail)
    return "\n".join(parts)


def _summarize_read_pdf(args: dict, output: str) -> str:
    """First ~300 chars + page count."""
    url = args.get("url", "")
    # Try to find page count
    page_match = re.search(r"(\d+)\s*pages?", output[:200], re.IGNORECASE)
    page_info = f" ({page_match.group(0)})" if page_match else ""

    text = output.strip()
    head = text[:300]
    if len(text) > 300:
        cut = head.rfind(" ")
        if cut > 150:
            head = head[:cut]
        head += " …"
    return f"PDF{page_info}: {url}\n{head}"


def _summarize_wikipedia(args: dict, output: str) -> str:
    """First ~300 chars of article text."""
    title = args.get("title", args.get("query", ""))
    section = args.get("section", "")
    text = output.strip()
    head = text[:300]
    if len(text) > 300:
        cut = head.rfind(" ")
        if cut > 150:
            head = head[:cut]
        head += " …"
    label = f"Wikipedia: {title}"
    if section:
        label += f" § {section}"
    return f"{label}\n{head}"


def _summarize_generic(args: dict, output: str) -> str:
    """Fallback: first 350 chars."""
    text = output.strip()
    head = text[:350]
    if len(text) > 350:
        cut = head.rfind(" ")
        if cut > 200:
            head = head[:cut]
        head += " …"
    return head


# ── Dispatch map ───────────────────────────────────────────────────────
_SUMMARIZERS = {
    "search_web":      _summarize_search_web,
    "fetch_url":       _summarize_fetch_url,
    "read_pdf":        _summarize_read_pdf,
    "execute_code":    _summarize_execute_code,
    "wikipedia_lookup": _summarize_wikipedia,
    "fetch_cached":    _summarize_fetch_url,      # same format as fetch_url
    "extract_tables":  _summarize_generic,         # typically small already
}
