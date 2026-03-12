"""
Tool handler nodes for the trajectorykit agent loop.

Each public function handles a single tool-call type.  Signature:

    def handle_<tool>(state, tool_call, tool_args, turn_record, *,
                      args_were_malformed=False) -> NodeResult

NodeResult is a (final_answer_content | None, should_break) tuple.
The runner inspects it after each call:
    - (None, False)    → continue to next tool call
    - (content, False) → finalise with *content*
    - (None, True)     → break to synthesis pipeline
"""

from __future__ import annotations

import json
import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .chain import ChainPlan, ChainStep

from .agent_state import AgentState
from . import config as _cfg
from .symbolic import make_symbolic
from .tool_store import dispatch_tool_call, fetch_url
from .tracing import ToolCallRecord, TurnRecord

import logging

logger = logging.getLogger(__name__)

# ── Return type ────────────────────────────────────────────────────────
# (final_answer_content_or_None, should_break_to_synthesis)
NodeResult = Tuple[Optional[str], bool]

_CONTINUE: NodeResult = (None, False)
_BREAK: NodeResult = (None, True)


def _finalize(content: str) -> NodeResult:
    return (content, False)


# ─────────────────────────────────────────────────────────────────────
# DISPATCH TABLE  (tool_name → handler)
# Built at module load time.  The runner does:
#     handler = TOOL_HANDLERS.get(tool_name, handle_generic_tool)
# ─────────────────────────────────────────────────────────────────────

TOOL_HANDLERS: Dict[str, Any] = {}  # populated at the bottom of this file


# ═══════════════════════════════════════════════════════════════════════
# VIRTUAL TOOLS (all depths)
# ═══════════════════════════════════════════════════════════════════════

def handle_think(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Process a think() call — reflection / scratch-pad."""
    thought = tool_args.get("thought", "").strip()
    _MIN_THINK_CHARS = 20
    if len(thought) < _MIN_THINK_CHARS:
        think_output = (
            "[Think rejected \u2014 too brief] Your think() call must be a "
            "substantive reflection. Include:\n"
            "  1. What data you have gathered so far\n"
            "  2. What is still missing for the task\n"
            "  3. Your plan for the next 1-2 tool calls\n"
            "Call think() again with at least a few sentences of reflection."
        )
        if state.verbose:
            print(f"       \U0001f4ad [rejected \u2014 {len(thought)} chars < {_MIN_THINK_CHARS}]")
    else:
        think_output = f"[Thought recorded]\n{thought}"
        # Only substantive thinks break the consecutive search chain
        state.consecutive_search_count = 0
        if state.verbose:
            _preview = thought[:120] + ("\u2026" if len(thought) > 120 else "")
            print(f"       \U0001f4ad {_preview}")

    tc_record = ToolCallRecord(
        tool_name="think",
        tool_args=tool_args,
        tool_call_id=tool_call["id"],
        output=think_output,
        duration_s=0,
        child_trace=None,
    )
    turn_record.tool_calls.append(tc_record)
    state.messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": think_output,
    })
    return _CONTINUE


def handle_search_available_tools(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """List available tools (filtered to this agent's actual tool set)."""
    _sat_tool_name = tool_args.get("tool_name")
    if _sat_tool_name:
        _sat_match = next(
            (t for t in state.available_tools if t["function"]["name"] == _sat_tool_name),
            None,
        )
        if _sat_match:
            _sat_output = json.dumps(_sat_match, indent=2)
        else:
            _sat_names = [t["function"]["name"] for t in state.available_tools]
            _sat_output = (
                f"No tool named '{_sat_tool_name}'. "
                f"Available tools: {', '.join(_sat_names)}"
            )
    else:
        _sat_lines = []
        for _sat_t in state.available_tools:
            _sat_func = _sat_t["function"]
            _sat_name = _sat_func["name"]
            _sat_params = _sat_func.get("parameters", {}).get("properties", {})
            _sat_required = set(_sat_func.get("parameters", {}).get("required", []))
            _sat_parts = []
            for _pname, _pschema in _sat_params.items():
                _ptype = _pschema.get("type", "any")
                if _pname in _sat_required:
                    _sat_parts.append(f"{_pname}: {_ptype}")
                else:
                    _sat_parts.append(f"{_pname}?: {_ptype}")
            _sat_sig = f"{_sat_name}({', '.join(_sat_parts)})"
            _sat_desc = _sat_func.get("description", "")[:80]
            _sat_lines.append(f"  {_sat_sig}\n    {_sat_desc}")
        _sat_output = f"Available tools ({len(state.available_tools)}):\n" + "\n".join(_sat_lines)

    tc_record = ToolCallRecord(
        tool_name="search_available_tools",
        tool_args=tool_args,
        tool_call_id=tool_call["id"],
        output=_sat_output,
        duration_s=0,
        child_trace=None,
    )
    turn_record.tool_calls.append(tc_record)
    state.messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": _sat_output,
    })
    if state.verbose:
        print(f"       \U0001f4cb Listed {len(state.available_tools)} available tools")
    return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# ROOT-ONLY VIRTUAL TOOLS (depth == 0)
# ═══════════════════════════════════════════════════════════════════════

def handle_refine_draft(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Save / update the draft report (root only)."""
    if state.depth != 0:
        return _CONTINUE  # only root

    draft_content = tool_args.get("content", "")
    if len(draft_content.strip()) < 50:
        draft_output = (
            "Draft too short \u2014 write a complete, self-contained answer. "
            "Include findings, citations, and structure. "
            "Each call replaces the entire draft."
        )
    else:
        state.draft_revised_since_rejection = True
        state.draft_versions.append((state.turn, draft_content))
        ver = len(state.draft_versions)
        if state.draft_path:
            try:
                with open(state.draft_path, "w", encoding="utf-8") as f:
                    f.write(f"<!-- Draft v{ver} | turn {state.turn} -->\n")
                    f.write(draft_content)
            except Exception as e:
                if state.verbose:
                    print(f"       \u26a0\ufe0f  Draft file write failed: {e}")
        state.memory.upsert(
            key="draft_latest",
            tool_name="refine_draft",
            turn=state.turn,
            content=draft_content,
            description="latest draft answer",
        )
        draft_output = (
            f"\u2705 Draft v{ver} saved ({len(draft_content):,} chars).\n\n"
            f"Next steps:\n"
            f"  - To research gaps: conduct_research(task='...')\n"
            f"  - To update draft: refine_draft(content='...')\n"
            f"  - To publish: research_complete()"
        )
        if state.verbose:
            print(f"       \U0001f4dd Draft v{ver} saved ({len(draft_content):,} chars) \u2192 {state.draft_path}")

    tc_record = ToolCallRecord(
        tool_name="refine_draft",
        tool_args=tool_args,
        tool_call_id=tool_call["id"],
        output=draft_output,
        duration_s=0,
        child_trace=None,
    )
    turn_record.tool_calls.append(tc_record)
    state.messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": draft_output,
    })
    return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# READ_DRAFT  (root only — view previous drafts + feedback)
# ═══════════════════════════════════════════════════════════════════════

def handle_read_draft(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Return a previous draft version and/or its verifier feedback."""
    if state.depth != 0:
        return _CONTINUE

    if not state.draft_versions:
        rd_output = "No drafts have been written yet. Use refine_draft(content='...') first."
    else:
        version = tool_args.get("version")
        list_versions = tool_args.get("list_versions", False)
        include_feedback = tool_args.get("include_feedback", False)

        if list_versions:
            # Show a compact table of all versions
            lines = [f"Draft history ({len(state.draft_versions)} version{'s' if len(state.draft_versions) != 1 else ''}):"]
            for i, (turn, text) in enumerate(state.draft_versions, 1):
                fb = state.draft_feedback.get(i)
                fb_tag = " \u274c rejected" if fb else ""
                lines.append(f"  v{i}: turn {turn}, {len(text):,} chars{fb_tag}")
            rd_output = "\n".join(lines)

        elif version is not None:
            # Return a specific version
            try:
                v = int(version)
            except (ValueError, TypeError):
                v = -1
            if v < 1 or v > len(state.draft_versions):
                rd_output = (
                    f"Version {version} does not exist. "
                    f"Valid versions: 1\u2013{len(state.draft_versions)}. "
                    f"Call read_draft(list_versions=true) to see all."
                )
            else:
                turn, text = state.draft_versions[v - 1]
                rd_output = (
                    f"\U0001f4c4 DRAFT v{v} (turn {turn}, {len(text):,} chars):\n\n"
                    f"{text}"
                )
                if include_feedback:
                    fb = state.draft_feedback.get(v)
                    if fb:
                        rd_output += f"\n\n\u2500\u2500\u2500 VERIFIER FEEDBACK (v{v}) \u2500\u2500\u2500\n{fb}"
                    else:
                        rd_output += f"\n\n(No verifier feedback recorded for v{v})"
        else:
            # Default: return the latest draft (same as what's injected)
            v = len(state.draft_versions)
            turn, text = state.draft_versions[-1]
            rd_output = (
                f"\U0001f4c4 CURRENT DRAFT v{v} (turn {turn}, {len(text):,} chars):\n\n"
                f"{text}"
            )
            if include_feedback:
                fb = state.draft_feedback.get(v)
                if fb:
                    rd_output += f"\n\n\u2500\u2500\u2500 VERIFIER FEEDBACK (v{v}) \u2500\u2500\u2500\n{fb}"
                else:
                    rd_output += f"\n\n(No verifier feedback recorded for v{v})"

    tc_record = ToolCallRecord(
        tool_name="read_draft", tool_args=tool_args,
        tool_call_id=tool_call["id"], output=rd_output,
        duration_s=0, child_trace=None,
    )
    turn_record.tool_calls.append(tc_record)
    state.messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": rd_output,
    })
    return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# RESEARCH_COMPLETE  (root only — verification + spot-check pipeline)
# ═══════════════════════════════════════════════════════════════════════

def _focused_page_summary(
    url: str,
    focus: str,
    model: str,
    vllm_url: str,
    max_fetch_chars: int = 200_000,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Fetch a URL and produce an LLM-generated summary.

    Shared by summarize_webpage and the spot-check evidence pipeline.
    The LLM is asked to wrap its output in <summary> tags so callers can
    robustly extract the core content even if the model adds preamble.
    """
    result = fetch_url(url, max_chars=max_fetch_chars)
    if not result.get("ok"):
        return f"Failed to fetch {url}: {result.get('reason', 'unknown error')}."
    page_content = result["content"]
    if len(page_content.strip()) < 50:
        return (
            f"Page fetched but content is too short ({len(page_content)} chars). "
            f"The page may be behind a paywall or require JavaScript."
        )
    focus_instruction = f"\n\nFOCUS: Pay special attention to: {focus}" if focus else ""
    msgs = [
        {
            "role": "system",
            "content": (
                "You are tasked with summarizing the raw content of a webpage. "
                "Your goal is to create a summary that preserves the most important "
                "information from the original web page. This summary will be used "
                "by a downstream research agent, so it's crucial to maintain the "
                "key details without losing essential information.\n\n"
                "RULES:\n"
                "  - Preserve specific facts: names, numbers, dates, statistics\n"
                "  - Keep source attributions (who said what)\n"
                "  - Maintain the logical structure of arguments\n"
                "  - Note any data tables or lists of items\n"
                "  - Flag if the page seems incomplete or paywalled\n"
                "  - Be concise but never drop quantitative data\n"
                "  - Be brief — aim for the most information-dense summary\n"
                "    possible, not a lengthy rewrite of the page\n\n"
                "OUTPUT FORMAT:\n"
                "  Wrap your entire summary inside <summary> and </summary> XML tags.\n"
                "  Example:\n"
                "  <summary>\n"
                "  Key finding: X was Y in 2023 (Source: Z).\n"
                "  Table data: A=10, B=20, C=30.\n"
                "  </summary>"
                f"{focus_instruction}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Summarize the following webpage content from: {url}\n\n"
                f"Produce a structured summary inside <summary></summary> tags."
                f"--- PAGE CONTENT ---\n{page_content}\n--- END ---\n\n"
            ),
        },
    ]
    try:
        _summary_payload: dict = {
            "model": model,
            "messages": msgs,
        }
        if temperature is not None:
            _summary_payload["temperature"] = temperature
        if max_tokens is not None:
            _summary_payload["max_tokens"] = max_tokens
        _summary_payload["reasoning_effort"] = "high"
        resp = requests.post(
            f"{vllm_url}/chat/completions",
            json=_summary_payload,
        )
        if resp.status_code == 200:
            choices = resp.json().get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                if text and text.strip():
                    return _extract_summary_tags(text.strip())
                return f"LLM returned empty summary. Raw page excerpt:\n{page_content[:3000]}"
            return f"LLM returned no choices. Raw page excerpt:\n{page_content[:3000]}"
        return (
            f"Summarization LLM call failed (HTTP {resp.status_code}). "
            f"Raw page excerpt:\n{page_content[:3000]}"
        )
    except Exception as e:
        return f"Summarization error: {e}. Raw page excerpt:\n{page_content[:3000]}"


def _extract_xml_tag(text: str, tag: str = "summary") -> str:
    """Extract content from <tag>...</tag> XML wrapper.

    Handles edge cases: missing tags, partial/truncated tags, nested
    content.  Falls back to full text if no tags found.

    Parameters
    ----------
    text : str
        Raw LLM output that may contain XML-wrapped content.
    tag : str
        The XML tag name to extract (e.g. "summary", "verdict", "claims",
        "compressed").
    """
    # Try to find content between <tag> and </tag>
    match = re.search(
        rf'<{tag}>\s*(.*?)\s*</{tag}>',
        text, re.DOTALL | re.IGNORECASE,
    )
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted

    # Fallback: opening tag exists but closing tag is missing (truncated output)
    match_open = re.search(
        rf'<{tag}>\s*(.*)',
        text, re.DOTALL | re.IGNORECASE,
    )
    if match_open:
        extracted = match_open.group(1).strip()
        # Strip a trailing partial closing tag
        extracted = re.sub(
            rf'</{tag[:4]}(?:{re.escape(tag[4:])}?)?\s*$', '',
            extracted, flags=re.IGNORECASE,
        ).strip()
        if extracted:
            return extracted

    # No tags at all — return full text as-is
    return text.strip()


# Backwards-compatible alias used by _focused_page_summary
_extract_summary_tags = lambda text: _extract_xml_tag(text, "summary")


def _run_verification(state: AgentState, draft_content: str) -> Tuple[bool, str, str, list]:
    """Run Stage-1 LLM verifier.  Returns (should_publish, feedback, raw_text, suspicious_claims)."""
    if (
        not _cfg.VERIFY_BEFORE_PUBLISH
        or not _cfg.VERIFIER_PROMPT
        or state.verification_rejections >= state.MAX_VERIFICATION_REJECTIONS
    ):
        if state.verification_rejections >= state.MAX_VERIFICATION_REJECTIONS and state.verbose:
            print(f"       \u23e9 Verifier: max rejections reached, force-publishing")
        return True, "", "", []

    try:
        # Resolve verifier model + endpoint: "external" uses verifier.model, "self" uses the local model
        use_external = (
            getattr(_cfg, "VERIFIER_STAGE1_PROVIDER", "self") == "external"
            and _cfg.VERIFIER_MODEL
        )
        v_model = _cfg.VERIFIER_MODEL if use_external else state.model
        v_api_url = (_cfg.VERIFIER_API_URL or _cfg.VLLM_API_URL) if use_external else _cfg.VLLM_API_URL

        # External model gets ONLY question + draft (no findings) for unbiased
        # cross-checking using its own training knowledge.
        # Local model gets research findings for context-aware verification.
        findings_section = ""
        if not use_external and state.findings:
            lines = []
            total = 0
            for i, f in enumerate(state.findings, 1):
                entry = f"{i}. {f}"
                if total + len(entry) > 60000:
                    lines.append(f"... ({len(state.findings) - i + 1} more entries truncated)")
                    break
                lines.append(entry)
                total += len(entry)
            findings_section = (
                "\n\nRESEARCH FINDINGS (raw worker outputs for cross-checking):\n"
                + "\n".join(lines)
            )

        # External model uses a dedicated prompt that outputs SUSPICIOUS_CLAIMS
        v_prompt = _cfg.VERIFIER_PROMPT
        if use_external and getattr(_cfg, "VERIFIER_EXTERNAL_PROMPT", ""):
            v_prompt = _cfg.VERIFIER_EXTERNAL_PROMPT

        verify_messages = [
            {"role": "system", "content": v_prompt},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{state.user_input}\n\n"
                    f"DRAFT ANSWER:\n{draft_content}"
                    f"{findings_section}"
                ),
            },
        ]
        payload = {
            "model": v_model,
            "messages": verify_messages,
            "reasoning_effort": "high",
        }
        # OpenAI newer models require max_completion_tokens; vLLM uses max_tokens.
        # External reasoning models (gpt-5-*) only support temperature=1 and have
        # lower max-completion-token ceilings, so we cap and omit temperature.
        if use_external:
            payload["max_completion_tokens"] = min(state.max_tokens, 100_000)
        else:
            payload["max_tokens"] = state.max_tokens
            payload["temperature"] = _cfg.VERIFIER_TEMPERATURE if _cfg.VERIFIER_TEMPERATURE is not None else state.temperature
        v_label = v_model if v_model != state.model else "same model"
        if state.verbose:
            print(f"       \U0001f50d Verifier ({v_label}): auditing draft (attempt {state.verification_rejections + 1})...")
        headers = {"Content-Type": "application/json"}
        v_api_key = getattr(_cfg, "VERIFIER_API_KEY", "") or ""
        if v_api_key and v_api_url != _cfg.VLLM_API_URL:
            headers["Authorization"] = f"Bearer {v_api_key}"
        resp = requests.post(f"{v_api_url}/chat/completions", json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            raw_text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()

            # Parse suspicious claims flagged by the external model
            suspicious_claims = []
            sc_json = _extract_xml_tag(raw_text, "suspicious_claims")
            if sc_json:
                try:
                    sc_parsed = json.loads(sc_json)
                    if isinstance(sc_parsed, list):
                        suspicious_claims = sc_parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            if suspicious_claims and state.verbose:
                print(f"       \U0001f50d Verifier: flagged {len(suspicious_claims)} suspicious claim(s)")

            text = _extract_xml_tag(raw_text, "verdict")
            clean = text.replace("*", "").upper()
            if "VERDICT: REVISION_NEEDED" in clean or "VERDICT:REVISION_NEEDED" in clean:
                state.verification_rejections += 1
                feedback = re.sub(
                    r'^.*?\*{0,2}Verdict\*{0,2}:\*{0,2}\s*REVISION_NEEDED\s*',
                    '', text, count=1, flags=re.IGNORECASE | re.DOTALL,
                ).strip()
                if not feedback:
                    feedback = text
                if state.verbose:
                    print(f"       \u26a0\ufe0f  Verifier: REVISION_NEEDED "
                          f"(rejection {state.verification_rejections}/{state.MAX_VERIFICATION_REJECTIONS})")
                return False, feedback, text, suspicious_claims
            else:
                if state.verbose:
                    print(f"       \u2705 Verifier: APPROVED")
                return True, "", text, suspicious_claims
        else:
            if state.verbose:
                err_body = ""
                try:
                    err_body = resp.json().get("error", {}).get("message", "")[:200]
                except Exception:
                    err_body = resp.text[:200]
                print(f"       \u26a0\ufe0f  Verifier API error ({resp.status_code}): {err_body}")
            return True, "", "", []
    except Exception as e:
        if state.verbose:
            print(f"       \u26a0\ufe0f  Verifier error: {e}, skipping verification")
        return True, "", "", []


def _resolve_numbered_citations(draft: str) -> Tuple[str, dict]:
    """Parse ## Sources section and inline-expand [N] → [N](URL) throughout the draft.

    Returns (expanded_draft, citation_map) where citation_map is {int: url_string}.
    This lets the extraction LLM see the URL right next to each claim instead of
    having to cross-reference against a Sources section 50+ lines away.
    """
    citation_map: dict[int, str] = {}

    # Find the ## Sources section (case-insensitive, also handles Chinese variants)
    sources_match = re.search(
        r'^##\s+(?:Sources|来源|参考|引用)\s*\n(.*)',
        draft, re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if not sources_match:
        return draft, citation_map

    sources_block = sources_match.group(1)
    # Parse lines like: [1] Source Title: https://... or [1] Source Title — https://...
    for line in sources_block.splitlines():
        m = re.match(r'\s*\[(\d+)\]\s*(.+)', line.strip())
        if m:
            num = int(m.group(1))
            rest = m.group(2)
            # Extract URL from the rest of the line
            url_m = re.search(r'(https?://\S+)', rest)
            if url_m:
                citation_map[num] = url_m.group(1).rstrip('.,;)')
            else:
                # No URL found — still record the source name for context
                citation_map[num] = rest.strip()

    if not citation_map:
        return draft, citation_map

    # Inline-expand [N] references in the body (everything BEFORE ## Sources)
    body = draft[:sources_match.start()]

    def _expand_citation(match):
        num = int(match.group(1))
        url = citation_map.get(num)
        if url and url.startswith("http"):
            return f"[{num}]({url})"
        return match.group(0)  # leave unchanged if no URL

    # Match [N] but not already expanded [N](url) — negative lookahead for '('
    expanded_body = re.sub(r'\[(\d+)\](?!\()', _expand_citation, body)

    # Also handle 【N】 (fullwidth brackets sometimes used in CJK text)
    def _expand_fullwidth(match):
        num = int(match.group(1))
        url = citation_map.get(num)
        if url and url.startswith("http"):
            return f"[{num}]({url})"
        return match.group(0)

    expanded_body = re.sub(r'【(\d+)】', _expand_fullwidth, expanded_body)

    return expanded_body + draft[sources_match.start():], citation_map


def _linkify_citations(draft: str) -> str:
    """Post-process: convert [N] → [[N]](url) in the body using Sources map.

    Only applied for report format.  Leaves the Sources section itself
    untouched.  Already-linked citations [N](url) are skipped.
    """
    if _cfg.DRAFT_FORMAT != "report":
        return draft

    _, citation_map = _resolve_numbered_citations(draft)
    if not citation_map:
        return draft

    # Split at ## Sources so we only transform the body
    sources_match = re.search(
        r'^##\s+(?:Sources|来源|参考|引用)\s*\n',
        draft, re.MULTILINE | re.IGNORECASE,
    )
    if not sources_match:
        return draft

    body = draft[:sources_match.start()]
    sources_section = draft[sources_match.start():]

    def _linkify(match):
        num = int(match.group(1))
        url = citation_map.get(num)
        if url and url.startswith("http"):
            return f"[[{num}]]({url})"
        return match.group(0)  # leave unchanged if no URL

    # Match [N] but NOT already-linked [N](url) — negative lookahead for '('
    linked_body = re.sub(r'\[(\d+)\](?!\()', _linkify, body)

    # Also handle fullwidth brackets 【N】
    def _linkify_fw(match):
        num = int(match.group(1))
        url = citation_map.get(num)
        if url and url.startswith("http"):
            return f"[[{num}]]({url})"
        return match.group(0)

    linked_body = re.sub(r'【(\d+)】', _linkify_fw, linked_body)

    return linked_body + sources_section


def _section_at_position(text: str, pos: int) -> str:
    """Return the nearest ## heading active at `pos` in the text."""
    best = ""
    for m in re.finditer(r'^##\s+(.+?)(?:\s*$)', text[:pos], re.MULTILINE):
        best = m.group(1).strip()
    return best


def _run_spot_check(state: AgentState, draft_content: str,
                    suspicious_claims: list | None = None) -> Tuple[bool, str, str, dict]:
    """Run Stage-2 spot-check — deterministic citation verification + Stage 1 suspicious claims.

    Instead of LLM-mediated claim extraction (fragile, arbitrary), this
    deterministically extracts ALL cited claims via _extract_citation_pairs()
    and merges any suspicious claims flagged by the external Stage 1 verifier.

    Returns (should_publish, feedback, raw_text, metadata_dict).
    """
    meta: dict = {}
    if (
        not _cfg.SPOT_CHECK_ENABLED
        or not _cfg.SPOTCHECK_COMPARE_PROMPT
        or state.spot_check_rejections >= state.MAX_SPOT_CHECK_REJECTIONS
    ):
        if state.spot_check_rejections >= state.MAX_SPOT_CHECK_REJECTIONS and state.verbose:
            print(f"       \u23e9 Spot-check: max rejections reached, skipping")
        return True, "", "", meta

    try:
        # ── Step 1: Deterministic citation extraction ─────────────────
        # Extract ALL [N] citation-claim pairs with their source URLs.
        result = _extract_citation_pairs(draft_content)
        citation_map, pairs = result if result else ({}, [])
        meta["citation_map"] = {str(k): v for k, v in citation_map.items()}

        # Convert citation pairs to claim objects for _spot_check_claims
        sc_claims: list[dict] = []
        for pair in pairs:
            clean_fact = re.sub(r'\[\d+\]', '', pair["attributed_fact"]).strip()
            # Detect which report section this citation is in
            pos = draft_content.find(pair["attributed_fact"][:50])
            section = _section_at_position(draft_content, pos) if pos >= 0 else ""
            sc_claims.append({
                "claim": pair["attributed_fact"],
                "search_query": clean_fact[:200],
                "source_url": pair["url"],
                "section": section,
            })

        # ── Step 2: Add Stage 1 suspicious claims ────────────────────
        if suspicious_claims:
            for sc in suspicious_claims:
                claim_text = sc.get("claim", "")
                if claim_text and not any(c["claim"] == claim_text for c in sc_claims):
                    sc_claims.append({
                        "claim": claim_text,
                        "search_query": claim_text[:200],
                        "source_url": "",
                        "section": "",
                    })

        meta["cited_claims_extracted"] = len(pairs)
        meta["stage1_suspicious_claims"] = len(suspicious_claims or [])
        meta["total_claims"] = len(sc_claims)

        if state.verbose:
            print(f"       \U0001f50e Spot-check: {len(pairs)} cited claim(s) + "
                  f"{len(suspicious_claims or [])} suspicious \u2192 "
                  f"{len(sc_claims)} total to verify")

        if sc_claims:
            return _spot_check_claims(state, draft_content, sc_claims, meta)

        # No checkable claims — check if it's a refusal
        return _spot_check_refusal(state, draft_content, meta)

    except Exception as e:
        if state.verbose:
            print(f"       \u26a0\ufe0f  Spot-check error: {e}, skipping")
        return True, "", "", meta


def _find_relevant_memory_keys(state: AgentState, claim_text: str, max_keys: int = 3) -> List[str]:
    """Find memory keys whose content is likely relevant to a claim.

    Uses simple keyword overlap: tokenise the claim into meaningful words,
    then score each memory entry by how many claim-words appear in the
    entry's key + first 500 chars of content.  Returns the top-scoring
    keys (up to *max_keys*), or all conduct_research keys if nothing
    matches.
    """
    if not state.memory:
        return []

    # Tokenise claim into lowercase keyword set
    _STOP = {"the", "and", "for", "that", "this", "with", "from", "are",
             "was", "were", "has", "have", "been", "its", "not", "but",
             "than", "which", "about", "into", "after", "before", "more",
             "most", "over", "also", "between", "through", "each", "any"}
    claim_words = set(
        w for w in re.findall(r"[a-z0-9]{3,}", claim_text.lower())
        if w not in _STOP
    )
    if not claim_words:
        # Fall back: return all conduct_research keys
        return [e.key for e in state.memory.entries
                if e.source_tool == "conduct_research"][:max_keys]

    scored: list[tuple[int, str]] = []
    for entry in state.memory.entries:
        # Match against key + content preview
        haystack = (entry.key + " " + entry.content[:500]).lower()
        hits = sum(1 for w in claim_words if w in haystack)
        if hits > 0:
            scored.append((hits, entry.key))

    if not scored:
        # No overlap at all — give all conduct_research entries
        return [e.key for e in state.memory.entries
                if e.source_tool == "conduct_research"][:max_keys]

    scored.sort(reverse=True)
    return [key for _, key in scored[:max_keys]]


def _spot_check_claims(
    state: AgentState,
    draft_content: str,
    sc_claims: list,
    meta: dict,
) -> Tuple[bool, str, str, dict]:
    """Verify extracted claims via search + LLM comparison."""
    if state.verbose:
        print(f"       \U0001f50e Spot-check: verifying {len(sc_claims)} claim(s)...")

    # ── Verification sub-agent budget ───────────────────────────────
    # 3 turns is enough: search → think+fetch → final_answer.
    # Sub-agents can search, fetch pages, read PDFs, use Wikipedia,
    # and think about what they find — far more expressive than the
    # old mechanical pipeline, and they don’t pollute the main context.
    _VERIFY_TURN_BUDGET = 3

    def _verify_one_claim(claim_obj):
        _q = claim_obj.get("search_query", "")
        _c = claim_obj.get("claim", "")
        _source_url = claim_obj.get("source_url", "")
        _depends_on = claim_obj.get("depends_on")  # chain dependency (int or None)
        _section = claim_obj.get("section", "")    # section attribution
        if not _q and not _source_url:
            return None

        # ── Build a focused verification task for a sub-agent ─────────
        # The sub-agent gets the full tool suite (search, fetch, read_pdf,
        # wikipedia_lookup, execute_code, etc.) and can reason about
        # what to check.
        _prior_evidence = claim_obj.get("_prior_evidence", "")
        task_parts = [
            "VERIFY this claim by gathering evidence.\n",
            f"CLAIM: {_c}\n",
        ]
        if _prior_evidence:
            task_parts.append(f"\n{_prior_evidence}\n")
        if _source_url:
            # Determine if this is an API/data endpoint vs a regular page
            _is_api = any(p in _source_url.lower() for p in [
                "/api/", "/resource/", ".json", ".csv", ".xml",
                "$select", "$where", "query=", "format=json",
            ])
            if _is_api:
                task_parts.append(
                    f"The draft used this API/data endpoint: {_source_url}\n"
                    f"Fetch it with fetch_url and use execute_code to parse the "
                    f"response (JSON/CSV/XML). Verify the claimed values appear in the data.\n"
                )
            else:
                task_parts.append(
                    f"The draft CITES this source: {_source_url}\n"
                    f"Fetch this page FIRST to check whether the claim actually appears there.\n"
                )
        if _q:
            task_parts.append(f"SUGGESTED SEARCH: {_q}\n")
        task_parts.append(
            "\nINSTRUCTIONS:\n"
            "1. If a source_url was provided, fetch it FIRST (use fetch_url for web pages/APIs, "
            "read_pdf for PDFs, wikipedia_lookup for Wikipedia articles)\n"
            "2. If the source returns structured data (JSON/CSV), use execute_code to parse "
            "and verify the specific values claimed\n"
            "3. Search the web for corroborating evidence if the source is unavailable or insufficient\n"
            "4. Prioritise primary sources: data portals, .gov/.edu sites, PDFs, Wikipedia\n"
            "5. If initial search results are weak, try different queries\n"
            "6. Fetch and READ the most relevant pages — don’t rely on snippets alone\n"
            "7. VERSION MATCH: If the claim references a specific edition, year, or data "
            "snapshot (e.g. 'Guardian 2024 rankings', '2020 Census'), verify against THAT "
            "exact version — not a different year's edition. If you can only find a different "
            "version, note the mismatch explicitly in your assessment.\n"
            "8. ACCESS FAILURES: If you cannot access a source (paywall, 403/404, login-wall, "
            "rate-limit, geo-block, API auth error):\n"
            "   a) ALWAYS try the original URL first — paywalls aren't always enforced and "
            "APIs sometimes allow limited unauthenticated access\n"
            "   b) If blocked, state clearly: 'ACCESS FAILED: [url] — [reason]'\n"
            "   c) Do NOT guess what the page contains — never fabricate evidence\n"
            "   d) Pivot quickly to alternatives: open-access mirrors (Semantic Scholar, "
            "PubMed Central, Internet Archive/Wayback Machine), cached versions, or "
            "different search queries — do not retry the same blocked URL\n"
            "   e) If you get partial access (e.g. abstract-only for a paper), report what you "
            "CAN see and note it is partial\n"
            "   f) If no alternative works, report INSUFFICIENT with the access failure details\n"
            "\nIn your final_answer, report:\n"
            "  • SOURCES CHECKED: [url] — [what it says about the claim]\n"
            "  • ASSESSMENT: SUPPORTED / CONTRADICTED / INSUFFICIENT\n"
            "  • If CONTRADICTED, state what the evidence says instead\n"
            "Be concise but include the key data points from each source."
        )
        task_prompt = "".join(task_parts)

        try:
            output, _child = dispatch_tool_call(
                "spawn_agent",
                {"task": task_prompt, "turn_length": _VERIFY_TURN_BUDGET},
                _depth=state.depth,
                model=state.model,
                reasoning_effort=state.reasoning_effort,
            )

            # Parse sub-agent JSON response to get the evidence report
            evidence_report = output
            try:
                parsed = json.loads(output)
                evidence_report = parsed.get("response", output)
            except (json.JSONDecodeError, TypeError):
                pass

            if state.verbose:
                print(f"       🔍 Spot-check sub-agent: claim verified ({len(evidence_report)} chars)")

            return {
                "claim": _c,
                "section": _section,
                "search_query": _q,
                "source_url": _source_url,
                "depends_on": _depends_on,
                "evidence_report": evidence_report,
            }
        except Exception as e:
            if state.verbose:
                print(f"       ⚠️  Spot-check sub-agent error: {e}")
            return {
                "claim": _c,
                "section": _section,
                "search_query": _q,
                "source_url": _source_url,
                "depends_on": _depends_on,
                "evidence_report": f"(verification failed: {e})",
            }

    # ── Tiered execution: respect chain dependencies ─────────────
    # Partition claims into dependency tiers.  Tier 0 = no depends_on;
    # tier N = depends on a claim in tier N-1.  Verify each tier in
    # parallel, but tiers execute sequentially so a dependent claim
    # can receive the verified evidence from its predecessor.
    claim_by_idx: dict[int, dict] = {}
    for i, c in enumerate(sc_claims, 1):
        claim_by_idx[i] = c
    tier_results: dict[int, dict] = {}  # claim-index → evidence dict

    def _build_tiers(claims_list):
        """Return list of tiers; each tier is a list of (index, claim)."""
        tiers: list[list[tuple[int, dict]]] = []
        placed = set()
        remaining = [(i, c) for i, c in enumerate(claims_list, 1)]
        while remaining:
            tier = []
            still_remaining = []
            for idx, c in remaining:
                dep = c.get("depends_on")
                if dep is None or dep in placed:
                    tier.append((idx, c))
                else:
                    still_remaining.append((idx, c))
            if not tier:
                # Circular or broken deps — flush everything into one tier
                tier = still_remaining
                still_remaining = []
            for idx, _ in tier:
                placed.add(idx)
            tiers.append(tier)
            remaining = still_remaining
        return tiers

    tiers = _build_tiers(sc_claims)
    sc_evidence: list[dict] = []

    for tier_num, tier in enumerate(tiers):
        if state.verbose and len(tiers) > 1:
            print(f"       ⛓ Spot-check tier {tier_num + 1}/{len(tiers)}: "
                  f"{len(tier)} claim(s)")

        # Inject prior-tier evidence into dependent claims so the
        # verifier knows what the predecessor actually resolved to.
        updated_tier = []
        for idx, claim_obj in tier:
            dep = claim_obj.get("depends_on")
            if dep is not None and dep in tier_results:
                prior_ev = tier_results[dep]
                prior_summary = prior_ev.get("evidence_report", "")[:500]
                claim_obj = {**claim_obj}  # shallow copy
                claim_obj["_prior_evidence"] = (
                    f"PRIOR STEP (Claim {dep}) verified result:\n"
                    f"{prior_summary}\n"
                    f"Use this to check whether THIS claim's entity/value "
                    f"is consistent with the verified prior step."
                )
            updated_tier.append((idx, claim_obj))

        max_workers = min(len(updated_tier), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_verify_one_claim, c): (i, c) for i, c in updated_tier}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    idx = futures[fut][0]
                    tier_results[idx] = result
                    sc_evidence.append(result)

    meta["claims_checked"] = len(sc_evidence)

    # Short-circuit: if >50% degraded evidence, auto-PASS.
    # A claim is degraded if the sub-agent failed or returned very little.
    if sc_evidence:
        degraded = sum(
            1 for ev in sc_evidence
            if ev.get("evidence_report", "").startswith("(verification failed")
            or len(ev.get("evidence_report", "").strip()) < 100
        )
        if degraded > len(sc_evidence) / 2:
            meta["spot_check_verdict"] = "PASSED"
            meta["spot_check_skipped_reason"] = "insufficient_evidence"
            meta["degraded_claims"] = degraded
            if state.verbose:
                print(f"       \u23e9 Spot-check: {degraded}/{len(sc_evidence)} claims have degraded evidence, auto-PASS")
            return True, "", "", meta

    if not sc_evidence or "spot_check_skipped_reason" in meta:
        return True, "", "", meta

    # Step 3: Compare claims against evidence
    # Preserve original claim ordering for chain coherence analysis.
    # Claims were verified in parallel so sc_evidence may be unordered;
    # re-sort by the original claim list position.
    _claim_texts = [c.get("claim", "") for c in sc_claims]
    sc_evidence.sort(key=lambda ev: (
        _claim_texts.index(ev["claim"]) if ev["claim"] in _claim_texts else 999
    ))

    has_chain = any(ev.get("depends_on") is not None for ev in sc_evidence)
    evidence_parts = []
    for i, ev in enumerate(sc_evidence, 1):
        part = f"CLAIM {i}: {ev['claim']}\n"
        if ev.get("section"):
            part += f"SECTION: {ev['section']}\n"
        if ev.get("depends_on") is not None:
            part += f"DEPENDS ON: Claim {ev['depends_on']}\n"
        report = ev.get("evidence_report", "(no evidence gathered)")
        part += f"\nVERIFICATION EVIDENCE:\n{report}"
        evidence_parts.append(part)
    evidence_text = "\n\n".join(evidence_parts)

    chain_note = ""
    if has_chain:
        # Build a concrete chain summary showing resolved values so the
        # compare LLM can check entity flow between steps.
        _chain_summary_parts = []
        if state.chain_plan and state.chain_plan.has_chain:
            for cs in state.chain_plan.chain_steps:
                rv = cs.resolved_value or "(unresolved)"
                _chain_summary_parts.append(
                    f"  Step {cs.step}: {cs.lookup} → {rv}"
                )
        _chain_body = "\n".join(_chain_summary_parts) if _chain_summary_parts else ""
        chain_note = (
            "\n\nCHAIN COHERENCE CHECK:\n"
            "The draft answers a chain question. The pre-computed chain resolved as:\n"
            f"{_chain_body}\n\n"
            "Verify:\n"
            "1. Does the VERIFIED evidence for each step match the resolved value "
            "used in the next step? (e.g. if Step 1 resolved to 'Cincinnati' but "
            "Step 2's evidence checked 'Columbus', the chain is BROKEN.)\n"
            "2. If any intermediate step's evidence CONTRADICTS its resolved value, "
            "flag it — all downstream steps are then suspect.\n"
            "3. Check that the FINAL answer follows logically from the verified chain."
        )

    compare_msgs = [
        {"role": "system", "content": _cfg.SPOTCHECK_COMPARE_PROMPT},
        {"role": "user", "content": (
            f"QUESTION:\n{state.user_input}\n\n"
            f"DRAFT ANSWER (excerpt of checked claims):\n\n"
            f"{evidence_text}"
            f"{chain_note}"
        )},
    ]
    compare_resp = requests.post(
        f"{_cfg.VLLM_API_URL}/chat/completions",
        json={
            "model": state.model,
            "messages": compare_msgs,
            "temperature": state.temperature,
            "max_tokens": state.max_tokens,
            "reasoning_effort": "high",
        },
    )
    if compare_resp.status_code == 200:
        compare_raw = (
            compare_resp.json().get("choices", [{}])[0]
            .get("message", {}).get("content", "")
        ).strip()
        meta["compare_response"] = compare_raw
        compare_text = _extract_xml_tag(compare_raw, "verdict")
        clean = compare_text.replace("*", "").upper()
        if "VERDICT: SPOT_CHECK_FAILED" in clean or "VERDICT:SPOT_CHECK_FAILED" in clean:
            state.spot_check_rejections += 1
            meta["spot_check_verdict"] = "FAILED"
            feedback = re.sub(
                r'^.*?\*{0,2}Verdict\*{0,2}:\*{0,2}\s*SPOT_CHECK_FAILED\s*',
                '', compare_text, count=1, flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            if not feedback:
                feedback = compare_text
            raw_text = f"[SPOT-CHECK] {compare_text}"
            # ── Chain contestation: clear contradicted resolved values ──
            # If the failure feedback mentions a specific chain step being
            # wrong, clear its resolved_value so the model doesn't repeat
            # the same error on the next attempt.
            if state.chain_plan and state.chain_plan.has_chain:
                contest_records = _contest_chain_steps(state.chain_plan, feedback)
                if contest_records:
                    meta["chain_contested"] = contest_records

            if state.verbose:
                print(f"       \u274c Spot-check: FAILED \u2014 factual issues found "
                      f"(rejection {state.spot_check_rejections}/{state.MAX_SPOT_CHECK_REJECTIONS})")
            return False, feedback, raw_text, meta
        else:
            meta["spot_check_verdict"] = "PASSED"
            if state.verbose:
                print(f"       \u2705 Spot-check: PASSED")
    else:
        if state.verbose:
            print(f"       \u26a0\ufe0f  Spot-check: compare API error ({compare_resp.status_code}), skipping")
    return True, "", "", meta


def _contest_chain_steps(chain_plan: "ChainPlan", feedback: str) -> List[dict]:
    """Clear resolved_value on chain steps contradicted by spot-check feedback.

    Scans the compare LLM's failure feedback for references to chain
    step entities.  If a step's resolved_value appears near words like
    "contradicted", "incorrect", "wrong", "actually", "instead", we
    clear it (and all downstream steps via ``contest_step``) so the
    model doesn't blindly re-use the bad intermediate on revision.

    Returns a list of dicts describing each contested step (for trace
    metadata), e.g. [{"step": 2, "old_value": "Cincinnati", "reason": "direct"}].
    """
    if not chain_plan.chain_steps:
        return []

    feedback_lower = feedback.lower()

    # Heuristic: for each resolved step, check if the feedback explicitly
    # mentions its resolved value near contradiction language.
    _CONTRADICT_PATTERNS = re.compile(
        r"(contradict|incorrect|wrong|actually|instead|not\s+\w+\s+but|"
        r"evidence\s+says|should\s+be|was\s+actually|misidentif|"
        r"chain\s+step\s+\d+\s+contradicted)",
        re.IGNORECASE,
    )

    contested: set[int] = set()

    for step in chain_plan.chain_steps:
        if not step.is_resolved or not step.resolved_value:
            continue
        rv = step.resolved_value.lower()
        # Skip very short values that would false-match everywhere
        if len(rv) < 4:
            continue
        # Check if the resolved value appears within 200 chars of
        # contradiction language in the feedback.
        for m in re.finditer(re.escape(rv), feedback_lower):
            window_start = max(0, m.start() - 200)
            window_end = min(len(feedback_lower), m.end() + 200)
            window = feedback_lower[window_start:window_end]
            if _CONTRADICT_PATTERNS.search(window):
                contested.add(step.step)
                break

    # Also check for explicit "CHAIN STEP N CONTRADICTED" patterns
    for m in re.finditer(r"chain\s+step\s+(\d+)\s+contradict", feedback_lower):
        contested.add(int(m.group(1)))

    if not contested:
        return []

    # Snapshot old values before clearing
    contest_records: list[dict] = []
    all_cleared: list[int] = []
    for step_num in contested:
        step = chain_plan.get_step(step_num)
        old_val = step.resolved_value if step else None
        cleared = chain_plan.contest_step(step_num)
        for sn in cleared:
            s_obj = chain_plan.get_step(sn)
            contest_records.append({
                "step": sn,
                "old_value": old_val if sn == step_num else "(cascade)",
                "reason": "direct" if sn == step_num else f"depends_on_step_{step_num}",
            })
        all_cleared.extend(cleared)

    if all_cleared:
        logger.info(
            "Spot-check contested chain step(s): %s (cleared: %s)",
            sorted(contested), sorted(set(all_cleared)),
        )

    return contest_records


def _spot_check_refusal(state: AgentState, draft_content: str, meta: dict) -> Tuple[bool, str, str, dict]:
    """If the draft looks like a refusal, challenge it with independent search."""
    refusal_patterns = [
        r"couldn.t find", r"could not find",
        r"unable to (find|locate|determine|identify)",
        r"no (publicly available|reliable|specific) (data|information|evidence|sources)",
        r"data (is|was) (not|un)available",
        r"not publicly (available|accessible)",
        r"information (is|was) not (available|found|accessible)",
        r"does not appear to be (publicly )?available",
        r"no (data|information|results|evidence) (was|were) found",
        r"cannot (be|find|locate|determine)",
    ]
    draft_lower = draft_content.lower()
    is_refusal = any(re.search(p, draft_lower) for p in refusal_patterns)

    if not is_refusal or not _cfg.SPOTCHECK_REFUSAL_PROMPT:
        if state.verbose and not is_refusal:
            print(f"       \u2139\ufe0f  Spot-check: no checkable claims extracted, skipping")
        return True, "", "", meta

    if state.verbose:
        print(f"       \U0001f50e Spot-check: draft looks like a refusal \u2014 launching refusal challenge...")

    # For chain questions, search for Step 1's lookup text instead of the
    # full composite question — the full question is often unanswerable
    # as-is, but Step 1 should be independently verifiable.
    rc_query = state.user_input[:200]
    if state.chain_plan and state.chain_plan.has_chain and state.chain_plan.chain_steps:
        _step1 = state.chain_plan.chain_steps[0]
        rc_query = _step1.lookup[:200]

    # ── Spawn a sub-agent to independently investigate the question ──
    # This is more expressive than the old search+fetch pipeline — the
    # sub-agent can try multiple queries, read PDFs, use Wikipedia, etc.
    try:
        rc_task = (
            f"Find evidence that answers this question:\n"
            f"QUESTION: {state.user_input[:500]}\n\n"
            f"SUGGESTED SEARCH: {rc_query}\n\n"
            f"Search the web, fetch relevant pages, read PDFs, try Wikipedia. "
            f"If initial results are weak, try different search queries. "
            f"Report what you find — even partial information is valuable."
        )
        rc_output, _ = dispatch_tool_call(
            "spawn_agent",
            {"task": rc_task, "turn_length": 3},
            _depth=state.depth,
            model=state.model,
            reasoning_effort=state.reasoning_effort,
        )
        rc_evidence_text = rc_output
        try:
            rc_parsed = json.loads(rc_output)
            rc_evidence_text = rc_parsed.get("response", rc_output)
        except (json.JSONDecodeError, TypeError):
            pass

        if state.verbose:
            print(f"       \U0001f50d Refusal challenge sub-agent: gathered {len(rc_evidence_text)} chars of evidence")

        rc_msgs = [
            {"role": "system", "content": _cfg.SPOTCHECK_REFUSAL_PROMPT},
            {"role": "user", "content": (
                f"QUESTION:\n{state.user_input}\n\n"
                f"DRAFT ANSWER (REFUSAL):\n{draft_content[:2000]}\n\n"
                f"INDEPENDENT EVIDENCE:\n{rc_evidence_text}"
            )},
        ]
        rc_resp = requests.post(
            f"{_cfg.VLLM_API_URL}/chat/completions",
            json={
                "model": state.model,
                "messages": rc_msgs,
                "temperature": state.temperature,
                "max_tokens": state.max_tokens,
                "reasoning_effort": "high",
            },
        )
        if rc_resp.status_code == 200:
            rc_raw = (
                rc_resp.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "")
            ).strip()
            meta["refusal_challenge_response"] = rc_raw
            rc_text = _extract_xml_tag(rc_raw, "verdict")
            rc_clean = rc_text.replace("*", "").upper()
            if "VERDICT: REFUSAL_CHALLENGED" in rc_clean or "VERDICT:REFUSAL_CHALLENGED" in rc_clean:
                state.spot_check_rejections += 1
                feedback = re.sub(
                    r'^.*?\*{0,2}Verdict\*{0,2}:\*{0,2}\s*REFUSAL_CHALLENGED\s*',
                    '', rc_text, count=1, flags=re.IGNORECASE | re.DOTALL,
                ).strip()
                if not feedback:
                    feedback = rc_text
                raw_text = f"[SPOT-CHECK] Refusal challenged: {rc_text}"
                meta["spot_check_verdict"] = "REFUSAL_CHALLENGED"
                if state.verbose:
                    print(f"       \u274c Refusal challenge: evidence found \u2014 refusal was premature "
                          f"(rejection {state.spot_check_rejections}/{state.MAX_SPOT_CHECK_REJECTIONS})")
                return False, feedback, raw_text, meta
            else:
                meta["spot_check_verdict"] = "REFUSAL_JUSTIFIED"
                if state.verbose:
                    print(f"       \u2705 Refusal challenge: refusal appears justified")
        else:
            if state.verbose:
                print(f"       \u26a0\ufe0f  Refusal challenge: API error ({rc_resp.status_code}), skipping")
    except Exception as e:
        if state.verbose:
            print(f"       \u26a0\ufe0f  Refusal challenge error: {e}, skipping")

    return True, "", "", meta


# ═══════════════════════════════════════════════════════════════════════
# CITATION AUDIT  (Stage 3 — citation faithfulness)
# ═══════════════════════════════════════════════════════════════════════

def _extract_citation_pairs(draft_content: str) -> Tuple[dict, list]:
    """Parse draft to extract (citation_map, citation_pairs).

    citation_map: {int: url_string}
    citation_pairs: list of dicts with keys:
        citation_num, url, attributed_fact (the sentence/clause where [N] appears)
    """
    # Re-use the citation map parser from _resolve_numbered_citations
    _, citation_map = _resolve_numbered_citations(draft_content)
    if not citation_map:
        return {}, []

    # Split draft into body (before ## Sources) and find each [N] in context
    sources_match = re.search(
        r'^##\s+(?:Sources|来源|参考|引用)\s*\n',
        draft_content, re.MULTILINE | re.IGNORECASE,
    )
    body = draft_content[:sources_match.start()] if sources_match else draft_content

    # Split body into sentences (rough heuristic — period/newline boundaries)
    # We want the sentence containing each [N] reference.
    pairs: list[dict] = []
    seen: set[tuple] = set()  # (citation_num, sentence_hash) dedup

    for m in re.finditer(r'\[(\d+)\]', body):
        num = int(m.group(1))
        if num not in citation_map:
            continue
        url = citation_map[num]
        if not url.startswith("http"):
            continue

        # Extract ~200 chars of surrounding context
        start = max(0, m.start() - 150)
        end = min(len(body), m.end() + 150)
        # Expand to sentence boundaries (nearest period/newline)
        while start > 0 and body[start] not in '.!?\n':
            start -= 1
            if m.start() - start > 300:
                break
        if start > 0:
            start += 1  # skip the boundary char itself
        while end < len(body) and body[end] not in '.!?\n':
            end += 1
            if end - m.end() > 300:
                break

        context = body[start:end].strip()
        dedup_key = (num, hash(context))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        pairs.append({
            "citation_num": num,
            "url": url,
            "attributed_fact": context,
        })

    return citation_map, pairs


def _run_citation_audit(
    state: AgentState,
    draft_content: str,
) -> Tuple[bool, str, str, dict]:
    """Run Stage-3 citation audit — verify each [N] URL supports its attributed fact.

    Returns (should_publish, feedback, raw_text, metadata_dict).
    """
    meta: dict = {}

    if (
        not _cfg.CITATION_AUDIT_ENABLED
        or not _cfg.CITATION_AUDIT_PROMPT
    ):
        return True, "", "", meta

    if state.verbose:
        print(f"       📑 Citation audit: extracting citation pairs...")

    citation_map, pairs = _extract_citation_pairs(draft_content)
    meta["citation_map"] = {str(k): v for k, v in citation_map.items()}
    meta["pairs_found"] = len(pairs)

    if not pairs:
        if state.verbose:
            print(f"       ℹ️  Citation audit: no verifiable citation pairs found, skipping")
        return True, "", "", meta

    if state.verbose:
        print(f"       📑 Citation audit: {len(pairs)} citation-claim pair(s) to check, fetching URLs...")

    # ── Fetch each cited URL (cache-first via fetch_url) ─────────
    # Group by URL to avoid duplicate fetches
    urls_to_fetch: dict[str, str] = {}  # url → content
    unique_urls = {p["url"] for p in pairs}

    def _fetch_one(url: str) -> Tuple[str, str]:
        result = fetch_url(url, max_chars=100_000)
        if result.get("ok"):
            content = result["content"]
            if len(content.strip()) < 50:
                return url, "(page fetched but content too short — possibly paywalled)"
            # Truncate to keep LLM context manageable
            return url, content[:8000]
        return url, f"(fetch failed: {result.get('reason', 'unknown')})"

    with ThreadPoolExecutor(max_workers=min(len(unique_urls), 6)) as pool:
        futures = {pool.submit(_fetch_one, u): u for u in unique_urls}
        for fut in as_completed(futures):
            url, content = fut.result()
            urls_to_fetch[url] = content

    # ── Build the LLM audit prompt ───────────────────────────────
    audit_parts = []
    for i, pair in enumerate(pairs, 1):
        url_content = urls_to_fetch.get(pair["url"], "(not fetched)")
        audit_parts.append(
            f"─── Citation [{pair['citation_num']}] (pair {i}) ───\n"
            f"URL: {pair['url']}\n"
            f"ATTRIBUTED FACT (text around this citation):\n"
            f"  \"{pair['attributed_fact']}\"\n\n"
            f"PAGE CONTENT (first 8000 chars):\n"
            f"{url_content}\n"
        )
    audit_text = "\n\n".join(audit_parts)

    audit_msgs = [
        {"role": "system", "content": _cfg.CITATION_AUDIT_PROMPT},
        {"role": "user", "content": (
            f"QUESTION:\n{state.user_input}\n\n"
            f"NUMBER OF CITATIONS TO CHECK: {len(pairs)}\n\n"
            f"{audit_text}"
        )},
    ]

    try:
        if state.verbose:
            print(f"       📑 Citation audit: sending {len(pairs)} pairs to LLM...")
        # Route through external verifier when configured for Stage 3
        use_external_s3 = (
            getattr(_cfg, "VERIFIER_STAGE3_PROVIDER", "self") == "external"
            and _cfg.VERIFIER_MODEL
        )
        ca_model = _cfg.VERIFIER_MODEL if use_external_s3 else state.model
        ca_api_url = (_cfg.VERIFIER_API_URL or _cfg.VLLM_API_URL) if use_external_s3 else _cfg.VLLM_API_URL
        ca_temp = (_cfg.VERIFIER_TEMPERATURE if _cfg.VERIFIER_TEMPERATURE is not None else state.temperature)
        ca_headers = {"Content-Type": "application/json"}
        ca_api_key = getattr(_cfg, "VERIFIER_API_KEY", "") or ""
        if ca_api_key and use_external_s3:
            ca_headers["Authorization"] = f"Bearer {ca_api_key}"

        if state.verbose and use_external_s3:
            print(f"       📑 Citation audit: using external model ({ca_model})")

        audit_payload = {
                "model": ca_model,
                "messages": audit_msgs,
                "reasoning_effort": "high",
            }
        # OpenAI newer models require max_completion_tokens; vLLM uses max_tokens.
        # External reasoning models only support temperature=1 and have lower token ceilings.
        if use_external_s3:
            audit_payload["max_completion_tokens"] = min(state.max_tokens, 100_000)
        else:
            audit_payload["max_tokens"] = state.max_tokens
            audit_payload["temperature"] = ca_temp

        audit_resp = requests.post(
            f"{ca_api_url}/chat/completions",
            json=audit_payload,
            headers=ca_headers,
            timeout=90,
        )
        if audit_resp.status_code == 200:
            audit_raw = (
                audit_resp.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "")
            ).strip()
            meta["audit_response"] = audit_raw
            audit_verdict_text = _extract_xml_tag(audit_raw, "audit")
            clean = audit_verdict_text.replace("*", "").upper()

            if "VERDICT: CITATION_FAIL" in clean or "VERDICT:CITATION_FAIL" in clean:
                meta["citation_audit_verdict"] = "FAILED"
                # Extract feedback: everything after the verdict line
                feedback = re.sub(
                    r'^.*?\*{0,2}Verdict\*{0,2}:\*{0,2}\s*CITATION_FAIL\s*',
                    '', audit_verdict_text, count=1,
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip()
                if not feedback:
                    # Fall back to full text
                    feedback = audit_verdict_text
                # Prepend the failing citations summary
                feedback = (
                    "Citation faithfulness check found issues:\n\n"
                    + feedback + "\n\n"
                    "Fix: for each UNSUPPORTED citation, either (a) find the "
                    "correct source URL that supports the claim, (b) remove "
                    "the citation, or (c) correct the attributed fact. Then "
                    "call refine_draft() and research_complete() again."
                )
                raw_text = f"[CITATION-AUDIT] {audit_verdict_text}"
                if state.verbose:
                    print(f"       ❌ Citation audit: FAILED — unfaithful citations found")
                return False, feedback, raw_text, meta
            else:
                meta["citation_audit_verdict"] = "PASSED"
                if state.verbose:
                    print(f"       ✅ Citation audit: PASSED")
        else:
            if state.verbose:
                print(f"       ⚠️  Citation audit: API error ({audit_resp.status_code}), skipping")
    except Exception as e:
        if state.verbose:
            print(f"       ⚠️  Citation audit error: {e}, skipping")

    return True, "", "", meta


def handle_research_complete(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Publish the draft — runs verification + spot-check + citation-audit pipeline."""
    if state.depth != 0:
        return _CONTINUE

    # Read draft from file
    draft_content = ""
    if state.draft_path and os.path.exists(state.draft_path):
        try:
            with open(state.draft_path, "r", encoding="utf-8") as f:
                raw = f.read()
            draft_content = re.sub(r'^<!--.*?-->\n?', '', raw, count=1).strip()
        except Exception as e:
            if state.verbose:
                print(f"       \u26a0\ufe0f  Failed to read draft file: {e}")

    if len(draft_content.strip()) < 50:
        reject_msg = (
            "Cannot publish \u2014 your draft is empty or too short. "
            "Call refine_draft(content='...') first with a complete answer, "
            "then call research_complete() to publish it."
        )
        tc_record = ToolCallRecord(
            tool_name="research_complete", tool_args=tool_args,
            tool_call_id=tool_call["id"], output=reject_msg,
            duration_s=0, child_trace=None,
        )
        turn_record.tool_calls.append(tc_record)
        state.messages.append({
            "role": "tool", "tool_call_id": tool_call["id"], "content": reject_msg,
        })
        if state.verbose:
            print(f"       \u26a0\ufe0f  research_complete rejected \u2014 draft is empty")
        return _CONTINUE

    # ── Format compliance check ──────────────────────────────────────
    format_issues = []
    if _cfg.DRAFT_FORMAT == "report":
        # Bench / research-report format: # Title → ## Executive Summary → ## Sections → ## Sources
        has_exec_summary = ("## Executive Summary" in draft_content
                            or "## executive summary" in draft_content.lower())
        if not has_exec_summary:
            format_issues.append(
                "Missing '## Executive Summary' section. Your report MUST include "
                "an Executive Summary that directly answers the question."
            )
        has_sources = ("## Sources" in draft_content or "## sources" in draft_content.lower()
                       or "## 参考" in draft_content or "## 来源" in draft_content
                       or "## 引用" in draft_content)
        if not has_sources:
            format_issues.append(
                "Missing '## Sources' section. Include source URLs at the end of the report."
            )
    else:
        # Default QA format: **Final Answer:** → **Sources:** → **Details:**
        if "**Final Answer:**" not in draft_content and "**Final Answer**" not in draft_content:
            format_issues.append(
                "Missing '**Final Answer:**' section. Your draft MUST start with "
                "'**Final Answer:**' followed by the direct answer."
            )
        if "**Sources:**" not in draft_content and "**Sources**" not in draft_content:
            format_issues.append(
                "Missing '**Sources:**' section. Include at least one source URL."
            )
    if format_issues:
        if _cfg.DRAFT_FORMAT == "report":
            fmt_msg = (
                "Cannot publish \u2014 your report does not follow the required format.\n\n"
                "ISSUES:\n- " + "\n- ".join(format_issues) + "\n\n"
                "Required structure:\n"
                "  # [Report Title]\n\n"
                "  ## Executive Summary\n  [direct answer to the question]\n\n"
                "  ## [Section Title]\n  [analytical content]\n\n"
                "  ## Sources\n  - [Source](URL)\n\n"
                "Fix with refine_draft(), then call research_complete() again."
            )
        else:
            fmt_msg = (
                "Cannot publish \u2014 your draft does not follow the required format.\n\n"
                "ISSUES:\n- " + "\n- ".join(format_issues) + "\n\n"
                "Required structure:\n"
                "  **Final Answer:**\n  [direct answer]\n\n"
                "  **Sources:**\n  - [Source](URL)\n\n"
                "  **Details:** (optional)\n  [supporting context]\n\n"
                "Fix with refine_draft(), then call research_complete() again."
            )
        tc_record = ToolCallRecord(
            tool_name="research_complete", tool_args=tool_args,
            tool_call_id=tool_call["id"], output=fmt_msg,
            duration_s=0, child_trace=None,
            metadata={"format_check": "failed", "issues": format_issues},
        )
        turn_record.tool_calls.append(tc_record)
        state.messages.append({
            "role": "tool", "tool_call_id": tool_call["id"], "content": fmt_msg,
        })
        if state.verbose:
            print(f"       \u26a0\ufe0f  Format check failed: {format_issues}")
        return _CONTINUE

    # ── Stage 1: LLM verification ────────────────────────────────────
    should_publish, verify_feedback, verify_text_raw, suspicious_claims = _run_verification(state, draft_content)

    # ── Stage 2: Spot-check (deterministic citation verification + suspicious claims) ──
    spot_check_meta: dict = {}
    if should_publish:
        should_publish, sc_feedback, sc_raw, spot_check_meta = _run_spot_check(
            state, draft_content, suspicious_claims=suspicious_claims)
        if not should_publish:
            verify_feedback = sc_feedback
            verify_text_raw = sc_raw

    # ── Stage 3: Citation audit (only when spot-check is disabled) ───
    # When spot-check is enabled, citation verification is already done
    # deterministically in Stage 2 via _extract_citation_pairs().
    citation_audit_meta: dict = {}
    if should_publish and not _cfg.SPOT_CHECK_ENABLED:
        should_publish, ca_feedback, ca_raw, citation_audit_meta = _run_citation_audit(state, draft_content)
        if not should_publish:
            verify_feedback = ca_feedback
            verify_text_raw = ca_raw

    # ── Publish or reject ────────────────────────────────────────────
    if should_publish:
        # ── Inline URL linkification: [N] → [[N]](url) ──────────
        published_content = _linkify_citations(draft_content)

        verify_meta: dict = {}
        if verify_text_raw:
            verify_meta["verification_verdict"] = "APPROVED"
            verify_meta["verification_response"] = verify_text_raw
        verify_meta["verification_attempt"] = state.verification_rejections + 1
        if spot_check_meta:
            verify_meta["spot_check"] = spot_check_meta
        if citation_audit_meta:
            verify_meta["citation_audit"] = citation_audit_meta
        tc_record = ToolCallRecord(
            tool_name="research_complete", tool_args=tool_args,
            tool_call_id=tool_call["id"], output=published_content,
            duration_s=0, child_trace=None,
            metadata=verify_meta if verify_meta else None,
        )
        turn_record.tool_calls.append(tc_record)
        if state.verbose:
            print(f"   ✅ research_complete — publishing draft ({len(published_content):,} chars)")
        return _finalize(published_content)
    else:
        is_citation_reject = verify_text_raw.startswith("[CITATION-AUDIT]")
        is_spot_check_reject = verify_text_raw.startswith("[SPOT-CHECK]")
        if is_citation_reject:
            reject_msg = (
                f"Citation faithfulness audit found issues in your draft:\n\n"
                f"{verify_feedback}\n\n"
                f"Please fix the cited sources with refine_draft(), then call "
                f"research_complete() again."
            )
        elif is_spot_check_reject:
            reject_msg = (
                f"Spot-check found factual issues in your draft that should be corrected:\n\n"
                f"{verify_feedback}\n\n"
                f"Please correct these specific claims with refine_draft(), then call "
                f"research_complete() again. "
                # f"(Spot-check rejection {state.spot_check_rejections}/{state.MAX_SPOT_CHECK_REJECTIONS} \u2014 "
                # f"{'next attempt will force-publish' if state.spot_check_rejections >= state.MAX_SPOT_CHECK_REJECTIONS else f'{state.MAX_SPOT_CHECK_REJECTIONS - state.spot_check_rejections} correction(s) remaining'})"
            )
        else:
            reject_msg = (
                f"Draft review found issues that should be addressed before publishing:\n\n"
                f"{verify_feedback}\n\n"
                f"Please fix these issues with refine_draft(), then call research_complete() again. "
                # f"(Attempt {state.verification_rejections}/{state.MAX_VERIFICATION_REJECTIONS} \u2014 "
                # f"{'next attempt will force-publish' if state.verification_rejections >= state.MAX_VERIFICATION_REJECTIONS else f'{state.MAX_VERIFICATION_REJECTIONS - state.verification_rejections} revision(s) remaining'})"
            )
        reject_meta = {
            "verification_verdict": (
                "CITATION_AUDIT_FAILED" if is_citation_reject
                else "SPOT_CHECK_FAILED" if is_spot_check_reject
                else "REVISION_NEEDED"
            ),
            "verification_response": verify_text_raw or verify_feedback,
            "verification_attempt": state.verification_rejections,
            "spot_check_attempt": state.spot_check_rejections,
        }
        if spot_check_meta:
            reject_meta["spot_check"] = spot_check_meta
        if citation_audit_meta:
            reject_meta["citation_audit"] = citation_audit_meta
        tc_record = ToolCallRecord(
            tool_name="research_complete", tool_args=tool_args,
            tool_call_id=tool_call["id"], output=reject_msg,
            duration_s=0, child_trace=None,
            metadata=reject_meta,
        )
        turn_record.tool_calls.append(tc_record)
        state.messages.append({
            "role": "tool", "tool_call_id": tool_call["id"], "content": reject_msg,
        })
        state.draft_revised_since_rejection = False
        # Store feedback keyed by draft version for read_draft retrieval
        _fb_ver = len(state.draft_versions)
        state.draft_feedback[_fb_ver] = verify_feedback
        return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# CONDUCT_RESEARCH  (root only — delegates to sub-agent)
# ═══════════════════════════════════════════════════════════════════════

def _match_chain_step(
    chain_plan: "ChainPlan",
    task_text: str,
    output: str,
) -> Optional["ChainStep"]:
    """Match a conduct_research task to an unresolved chain step.

    Uses Jaccard word-overlap between the task and each unresolved
    chain step's lookup text. Also checks if the task explicitly
    references the step placeholder.

    Returns the best-matching unresolved ChainStep, or None.
    """
    from .chain import ChainPlan, ChainStep  # lazy to avoid circular

    stopwords = frozenset({
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "has", "have", "been", "will", "but", "not", "all", "can", "about",
        "how", "what", "which", "their", "there", "each", "who", "more",
        "find", "search", "look", "research", "investigate",
    })

    task_words = set(re.findall(r"\w{3,}", task_text.lower())) - stopwords
    if not task_words:
        return None

    best_score = 0.0
    best_step = None

    for step in chain_plan.chain_steps:
        if step.is_resolved:
            continue
        # Check explicit placeholder reference first
        if step.placeholder and step.placeholder in task_text:
            # Task explicitly references this step's placeholder — poor match
            # (it's trying to USE this step, not resolve it)
            continue
        lookup = step.lookup_resolved(chain_plan)
        step_words = set(re.findall(r"\w{3,}", lookup.lower())) - stopwords
        if not step_words:
            continue
        jaccard = len(task_words & step_words) / len(task_words | step_words)
        if jaccard > best_score:
            best_score = jaccard
            best_step = step

    if best_score >= 0.35 and best_step is not None:
        return best_step
    return None


def _extract_chain_answer(output: str) -> str:
    """Extract a concise factual answer from sub-agent output.

    Used to fill chain step resolved_value. Tries to pull out the
    key factual answer rather than the full verbose output.
    """
    text = output.strip()

    # If it's JSON-wrapped (spawn_agent response), unwrap
    if text.startswith("{"):
        import json as _json
        try:
            parsed = _json.loads(text)
            resp = parsed.get("response", "")
            if resp:
                text = resp.strip()
        except (ValueError, KeyError):
            pass

    # Take first meaningful line (skip empty lines and markers)
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "---", "```", "[", "Source")):
            # Cap at reasonable length for a placeholder value
            if len(stripped) > 200:
                stripped = stripped[:200].rsplit(" ", 1)[0] + "..."
            return stripped

    # Fallback: first 200 chars
    return text[:200].strip()


def _find_similar_prior_research(state: AgentState, new_task: str) -> Optional[str]:
    """Check memory for prior conduct_research results with overlapping tasks.

    Uses Jaccard word-overlap on the task description.  Returns a summary
    note to inject into the new sub-agent's context, or None.
    """
    if not new_task or not state.memory.entries:
        return None

    # Tokenise the new task into 3+ letter words (lowered)
    stopwords = frozenset({
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "has", "have", "been", "will", "but", "not", "all", "can", "about",
        "how", "what", "which", "their", "there", "each", "who", "more",
    })
    new_words = set(re.findall(r"\w{3,}", new_task.lower())) - stopwords
    if len(new_words) < 2:
        return None

    best_score = 0.0
    best_entry = None

    for entry in state.memory.entries:
        if entry.source_tool != "conduct_research":
            continue
        # The key slug contains the task description words
        entry_words = set(re.findall(r"\w{3,}", entry.key.lower())) - stopwords
        if not entry_words:
            continue
        jaccard = len(new_words & entry_words) / len(new_words | entry_words)
        if jaccard > best_score:
            best_score = jaccard
            best_entry = entry

    if best_score >= 0.40 and best_entry is not None:
        prior = best_entry.content[:2000]
        if len(best_entry.content) > 2000:
            prior += "..."
        return (
            "[PRIOR RESEARCH NOTE: A previous sub-agent researched a very "
            f"similar task ({best_entry.key}). Their findings are below — "
            "build on these rather than re-doing the same work. If URLs they "
            "mention failed, do not retry them; search for alternatives.]\n\n"
            + prior
        )
    return None


# ── Evidence extraction from child traces ─────────────────────────────
# Walk a sub-agent's EpisodeTrace to build a compact evidence block that
# gives the verifier actual data values (URLs fetched, table rows, search
# snippets) rather than just the sub-agent's final_answer JSON wrapper.

_EVIDENCE_TOOLS = {"fetch_url", "extract_tables", "search_web", "wikipedia_lookup", "execute_code"}
_EVIDENCE_CAP = 4000       # total chars budget for evidence block
_PER_SNIPPET_CAP = 800     # max chars per individual tool output snippet

def _extract_child_evidence(child_trace, task_desc: str = "", mem_key: str | None = None) -> str | None:
    """Extract key data snippets from a child trace for the verifier.

    Returns a compact evidence string, or None if nothing useful found.
    """
    if child_trace is None:
        return None

    snippets: list[str] = []
    total_len = 0

    for turn in child_trace.turns:
        for tc in turn.tool_calls:
            if tc.tool_name not in _EVIDENCE_TOOLS:
                continue
            raw = (tc.output or "").strip()
            if not raw or raw.startswith("ERROR:") or raw.startswith("\u26d4"):
                continue
            # Skip very short outputs (not useful as evidence)
            if len(raw) < 60:
                continue

            # Build label
            label_parts = [tc.tool_name]
            if tc.tool_name == "fetch_url":
                url = (tc.tool_args or {}).get("url", "")
                if url:
                    label_parts.append(url[:120])
            elif tc.tool_name == "search_web":
                query = (tc.tool_args or {}).get("query", "")
                if query:
                    label_parts.append(f'q="{query[:80]}"')
            elif tc.tool_name == "execute_code":
                label_parts.append("(code output)")

            label = " | ".join(label_parts)

            # Truncate the output to a useful snippet
            # Prefer the start — contains headers/first rows/first results
            snippet = raw[:_PER_SNIPPET_CAP]
            if len(raw) > _PER_SNIPPET_CAP:
                # Try to cut at a newline boundary
                last_nl = snippet.rfind("\n")
                if last_nl > _PER_SNIPPET_CAP // 2:
                    snippet = snippet[:last_nl]
                snippet += f"\n  ... ({len(raw):,} chars total)"

            entry = f"  [{label}]: {snippet}"
            if total_len + len(entry) > _EVIDENCE_CAP:
                remaining = len([tc2 for t in child_trace.turns for tc2 in t.tool_calls
                                 if tc2.tool_name in _EVIDENCE_TOOLS])
                snippets.append(f"  ... (evidence cap reached, {remaining} evidence calls total)")
                break
            snippets.append(entry)
            total_len += len(entry)
        else:
            continue
        break  # break outer loop if inner loop broke

    if not snippets:
        return None

    header = f"[EVIDENCE from sub-agent"
    if task_desc:
        header += f': "{task_desc[:80]}"'
    header += "]"
    if mem_key:
        header += f" (full data in memory key: {mem_key})"
    return header + "\n" + "\n".join(snippets)


def handle_conduct_research(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Dispatch a sub-agent research task (root only)."""
    if state.depth != 0:
        return _CONTINUE

    new_task = str(tool_args.get("task", ""))

    # ── Chain enforcement gate ────────────────────────────────────────
    # If a causal chain is active, block tasks that reference unresolved
    # placeholders (the model is trying to skip ahead).
    if state.chain_plan is not None and state.chain_plan.has_chain:
        missing = state.chain_plan.unresolved_dependencies_for(new_task)
        if missing:
            missing_desc = "; ".join(
                f"step {s.step} ({s.lookup})" for s in missing
            )
            block_msg = (
                f"⛓ BLOCKED: This task references unresolved chain dependencies: "
                f"{missing_desc}. You must resolve earlier chain steps first. "
                f"Next step to resolve: "
            )
            next_step = state.chain_plan.next_unlocked_step()
            if next_step:
                resolved_lookup = next_step.lookup_resolved(state.chain_plan)
                block_msg += (
                    f"Step {next_step.step} — {resolved_lookup}"
                )
            else:
                block_msg += "(all preceding steps appear resolved — check your task text)"
            state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": block_msg,
            })
            tc_record = ToolCallRecord(
                tool_name="conduct_research", tool_args=tool_args,
                tool_call_id=tool_call["id"], output=block_msg,
                duration_s=0.0, child_trace=None,
            )
            turn_record.tool_calls.append(tc_record)
            if state.verbose:
                print(f"       ⛓ Chain gate: blocked out-of-order dispatch")
            return _CONTINUE

    # ── Task-level deduplication: inject prior findings if similar ─────
    prior_note = _find_similar_prior_research(state, new_task)
    if prior_note:
        existing_ctx = tool_args.get("context", "") or ""
        tool_args = {**tool_args, "context": (existing_ctx + "\n\n" + prior_note).strip()}
        if state.verbose:
            print(f"       ♻  Injected prior research note into sub-agent context")

    tc_start = time.time()
    child_trace = None
    try:
        output, child_trace = dispatch_tool_call(
            "spawn_agent", tool_args,
            _depth=state.depth, model=state.model,
            reasoning_effort=state.reasoning_effort,
            _sandbox_files=state.sandbox_files,
            _memory_store=state.memory,
        )
        if state.verbose and len(output) < 200:
            print(f"       \u2192 {output}")
        elif state.verbose:
            print(f"       \u2192 {output[:200]}...")
    except Exception as e:
        output = f"ERROR: {str(e)}"
        if state.verbose:
            print(f"       \u2192 \u274c {output}")
    tc_duration = round(time.time() - tc_start, 3)

    tc_record = ToolCallRecord(
        tool_name="conduct_research", tool_args=tool_args,
        tool_call_id=tool_call["id"], output=output,
        duration_s=tc_duration, child_trace=child_trace,
    )
    turn_record.tool_calls.append(tc_record)

    # Track in findings and memory
    mem_key = None
    if not output.startswith("ERROR:"):
        state.conduct_research_count += 1
        if state.draft_versions:
            state.research_after_first_draft += 1
        state.findings.append(f"[conduct_research] {output.strip()}")
        desc = str(tool_args.get("task", ""))[:60]
        mem_key = state.memory.add(
            tool_name="conduct_research", turn=state.turn,
            content=output, description=desc,
        )
        if state.plan is not None:
            state.plan.record_tool_call(
                tool_name="conduct_research", tool_args=tool_args,
                output=output, memory_key=mem_key, is_error=False,
            )
    else:
        if state.plan is not None:
            state.plan.record_tool_call(
                tool_name="conduct_research", tool_args=tool_args,
                output=output, memory_key=None, is_error=True,
            )

    # ── Chain step resolution ─────────────────────────────────────────
    # If a chain is active and the task looks like it matches a chain step,
    # resolve it with a concise summary of the output.
    if (state.chain_plan is not None and state.chain_plan.has_chain
            and not output.startswith("ERROR:")):
        matched_step = _match_chain_step(state.chain_plan, new_task, output)
        if matched_step is not None:
            # Extract a concise answer from the output (first meaningful line)
            answer_value = _extract_chain_answer(output)
            state.chain_plan.resolve_step(matched_step.step, answer_value)
            # Also update the linked PlanTask if it exists
            if state.plan is not None:
                plan_task = state.plan.find_chain_task(matched_step.step)
                if plan_task is not None:
                    plan_task.status = "done"
                    plan_task.result_summary = answer_value[:200]
            if state.verbose:
                print(f"       ⛓ Resolved chain step {matched_step.step} → {answer_value[:80]}")

    # Build the message for context
    msg_output = re.sub(
        r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
        r"\1[file content saved to trace]\2",
        output, flags=re.DOTALL,
    )
    if _cfg.SYMBOLIC_REFERENCES and mem_key is not None and len(msg_output) > _cfg.SYMBOLIC_THRESHOLD:
        msg_output = make_symbolic(
            tool_name="conduct_research", tool_args=tool_args,
            output=msg_output, memory_key=mem_key,
        )
        if state.verbose:
            print(f"       \U0001f4ce Symbolic ref: {len(output):,} \u2192 {len(msg_output):,} chars [{mem_key}]")
    state.messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": str(msg_output),
    })
    return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# SUMMARIZE_WEBPAGE  (root only — lightweight fetch + summarize)
# ═══════════════════════════════════════════════════════════════════════

def handle_summarize_webpage(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Fetch a URL and produce a focused LLM summary (root only)."""
    if state.depth != 0:
        return _CONTINUE

    tc_start = time.time()
    sw_url = tool_args.get("url", "")
    sw_focus = tool_args.get("focus", "")

    if not sw_url:
        sw_output = "ERROR: url is required."
    else:
        try:
            sw_output = _focused_page_summary(
                url=sw_url, focus=sw_focus, model=state.model,
                vllm_url=_cfg.VLLM_API_URL,
                temperature=state.temperature,
                max_tokens=state.max_tokens,
            )
            if sw_output.startswith("Failed to fetch") or sw_output.startswith("Page fetched but"):
                sw_output += (
                    " Try conduct_research instead \u2014 a sub-agent can "
                    "retry with different strategies."
                )
        except Exception as e:
            sw_output = f"ERROR: {str(e)}"

    tc_duration = round(time.time() - tc_start, 3)
    tc_record = ToolCallRecord(
        tool_name="summarize_webpage", tool_args=tool_args,
        tool_call_id=tool_call["id"], output=sw_output,
        duration_s=tc_duration, child_trace=None,
    )
    turn_record.tool_calls.append(tc_record)

    # Track in findings and memory
    mem_key = None
    if not sw_output.startswith("ERROR:"):
        if state.draft_versions:
            state.research_after_first_draft += 1
        state.findings.append(f"[summarize_webpage] {sw_output[:1500].strip()}")
        sw_desc = f"summary of {sw_url[:50]}"
        if sw_focus:
            sw_desc += f" (focus: {sw_focus[:30]})"
        mem_key = state.memory.add(
            tool_name="summarize_webpage", turn=state.turn,
            content=sw_output, description=sw_desc,
        )
        if state.plan is not None:
            state.plan.record_tool_call(
                tool_name="summarize_webpage", tool_args=tool_args,
                output=sw_output, memory_key=mem_key, is_error=False,
            )
    else:
        if state.plan is not None:
            state.plan.record_tool_call(
                tool_name="summarize_webpage", tool_args=tool_args,
                output=sw_output, memory_key=None, is_error=True,
            )

    # Symbolic compression
    msg_output = sw_output
    if _cfg.SYMBOLIC_REFERENCES and mem_key is not None and len(msg_output) > _cfg.SYMBOLIC_THRESHOLD:
        msg_output = make_symbolic(
            tool_name="summarize_webpage", tool_args=tool_args,
            output=msg_output, memory_key=mem_key,
        )
        if state.verbose:
            print(f"       \U0001f4ce Symbolic ref: {len(sw_output):,} \u2192 {len(msg_output):,} chars [{mem_key}]")
    state.messages.append({
        "role": "tool", "tool_call_id": tool_call["id"], "content": str(msg_output),
    })
    if state.verbose:
        preview = sw_output[:200].replace('\n', ' ')
        print(f"       \U0001f310 Summarized {sw_url[:60]} ({tc_duration:.1f}s): {preview}...")
    return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# COMPRESS_FINDINGS  (root only — LLM-based findings compression)
# ═══════════════════════════════════════════════════════════════════════

def handle_compress_findings(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Compress all MemoryStore entries into a structured summary."""
    if state.depth != 0:
        return _CONTINUE

    tc_start = time.time()
    cf_focus = tool_args.get("focus", "")
    cf_entries = state.memory.entries

    if not cf_entries:
        cf_output = (
            "No research findings to compress yet. "
            "Use conduct_research first to gather data, then call "
            "compress_findings to organize it."
        )
    else:
        try:
            cf_data_parts = []
            cf_total_chars = 0
            for entry in cf_entries:
                entry_content = entry.content
                if len(entry_content) > 5000:
                    entry_content = entry_content[:5000] + "\n... [truncated]"
                cf_data_parts.append(
                    f"=== [{entry.key}] "
                    f"(source: {entry.source_tool}, turn {entry.turn}) ===\n"
                    f"{entry_content}"
                )
                cf_total_chars += len(entry_content)

            cf_data_block = "\n\n".join(cf_data_parts)
            if len(cf_data_block) > 60000:
                cf_data_block = cf_data_block[:60000] + "\n\n... [remaining entries truncated]"

            focus_instruction = f"\n\nPRIORITY FOCUS: {cf_focus}" if cf_focus else ""

            compress_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant that has conducted research on a "
                        "topic by calling several tools and web searches. Your job is to "
                        "clean up and compress the findings, but preserve ALL relevant "
                        "statements, facts, and information that the researcher gathered.\n\n"
                        f"Today's date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
                        "OUTPUT FORMAT:\n"
                        "Organize findings into a structured summary:\n\n"
                        "## Key Facts & Data\n"
                        "  - Specific numbers, dates, statistics with their sources\n\n"
                        "## Themes & Findings\n"
                        "  - Group related findings under descriptive subheadings\n"
                        "  - Cite which research entry each finding came from\n\n"
                        "## Contradictions & Conflicts\n"
                        "  - Note where sources disagree and which seems more reliable\n\n"
                        "## Gaps & Missing Information\n"
                        "  - What questions remain unanswered?\n"
                        "  - What data would strengthen the analysis?\n\n"
                        "RULES:\n"
                        "  - NEVER drop quantitative data (numbers, percentages, dates)\n"
                        "  - Preserve source attributions\n"
                        "  - Be concise but complete \u2014 compress prose, keep facts\n"
                        "  - If a finding is well-supported by multiple sources, note that\n\n"
                        "OUTPUT WRAPPING:\n"
                        "  Wrap your ENTIRE structured summary inside <compressed> and </compressed> XML tags.\n"
                        "  You may think/reason before the opening tag, but the summary MUST be inside the tags."
                        f"{focus_instruction}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Compress and organize the following {len(cf_entries)} research "
                        f"entries ({cf_total_chars:,} total characters):\n\n"
                        f"{cf_data_block}"
                    ),
                },
            ]
            cf_payload = {
                "model": state.model,
                "messages": compress_messages,
                "temperature": state.temperature,
                "max_tokens": state.max_tokens,
                "reasoning_effort": "high",
            }
            cf_resp = requests.post(f"{_cfg.VLLM_API_URL}/chat/completions", json=cf_payload)
            if cf_resp.status_code == 200:
                cf_result = cf_resp.json()
                cf_choices = cf_result.get("choices", [])
                if cf_choices:
                    cf_raw = cf_choices[0].get("message", {}).get("content", "")
                    cf_output = _extract_xml_tag(cf_raw, "compressed") if cf_raw else ""
                    if not cf_output or not cf_output.strip():
                        cf_output = (
                            f"LLM returned empty compression. "
                            f"You have {len(cf_entries)} entries in memory. "
                            f"Try refine_draft directly using the findings summaries."
                        )
                else:
                    cf_output = "LLM returned no choices for compression."
            else:
                error_detail = ""
                try:
                    error_detail = cf_resp.json().get("error", {}).get("message", "")[:200]
                except Exception:
                    pass
                cf_output = (
                    f"Compression LLM call failed (HTTP {cf_resp.status_code}). "
                    f"{error_detail}"
                )
        except Exception as e:
            cf_output = f"ERROR: {str(e)}"

    tc_duration = round(time.time() - tc_start, 3)
    tc_record = ToolCallRecord(
        tool_name="compress_findings", tool_args=tool_args,
        tool_call_id=tool_call["id"], output=cf_output,
        duration_s=tc_duration, child_trace=None,
    )
    turn_record.tool_calls.append(tc_record)

    if not cf_output.startswith("ERROR:") and not cf_output.startswith("No research"):
        cf_desc = f"compressed findings ({len(cf_entries)} entries)"
        if cf_focus:
            cf_desc += f" [focus: {cf_focus[:30]}]"
        state.memory.upsert(
            key="compressed_findings", tool_name="compress_findings",
            turn=state.turn, content=cf_output, description=cf_desc,
        )
        state.findings.append(f"[compress_findings] {cf_output[:1500].strip()}")

    state.messages.append({
        "role": "tool", "tool_call_id": tool_call["id"], "content": cf_output,
    })
    if state.verbose:
        print(f"       \U0001f4e6 Compressed {len(cf_entries)} findings ({tc_duration:.1f}s, {len(cf_output):,} chars output)")
    return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# FINAL_ANSWER  (all depths — terminal for sub-agents, redirect for root)
# ═══════════════════════════════════════════════════════════════════════

def handle_final_answer(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    *,
    args_were_malformed: bool = False,
    **kw,
) -> NodeResult:
    """Handle final_answer — terminal for sub-agents only.

    Root calls should be blocked by runner.py before reaching here.
    If one slips through, hard-block it (no draft saving, no redirect).
    """

    # ── Root: hard block (safety net — runner.py should catch first) ──
    if state.depth == 0:
        error_msg = (
            "⛔ final_answer is NOT available at root level. "
            "Use refine_draft(content='...') then research_complete()."
        )
        tc_record = ToolCallRecord(
            tool_name="final_answer", tool_args=tool_args,
            tool_call_id=tool_call["id"], output=error_msg,
            duration_s=0, child_trace=None,
        )
        turn_record.tool_calls.append(tc_record)
        state.messages.append({
            "role": "tool", "tool_call_id": tool_call["id"], "content": error_msg,
        })
        if state.verbose:
            print(f"   ⛔ Root called final_answer — hard-blocked")
        return _CONTINUE

    # ── Sub-agent / synthesizer path ──────────────────────────────────
    final_content = tool_args.get("answer", "")

    # Reject empty answer from malformed args if turns remain
    if len(final_content.strip()) < 20 and args_were_malformed:
        has_turns_left = state.turn_length is None or state.turn < state.turn_length
        if has_turns_left:
            reject_msg = (
                "ERROR: Your final_answer call had malformed JSON arguments and "
                "the answer could not be recovered. Please call final_answer again "
                "with your complete answer as a simple string. Make sure to properly "
                "escape any quotes or special characters in the answer text."
            )
            state.messages.append({
                "role": "tool", "tool_call_id": tool_call["id"], "content": reject_msg,
            })
            tc_record = ToolCallRecord(
                tool_name="final_answer", tool_args=tool_args,
                tool_call_id=tool_call["id"], output=reject_msg,
                duration_s=0, child_trace=None,
            )
            turn_record.tool_calls.append(tc_record)
            if state.verbose:
                print(f"   \U0001f504 Rejected malformed final_answer \u2014 asking model to retry")
            return _CONTINUE

    # Backfill empty answer from prior assistant content
    if len(final_content.strip()) < 20:
        _NON_ANSWER_PREFIXES = (
            "i should", "i'll", "i need", "i will", "let me",
            "okay", "ok,", "ok ", "sure", "now i",
            "let's", "alright",
        )
        _STRUCTURE_SIGNALS = ("|", "#", "- ", "1.", "1)", "{", "**")
        for msg in reversed(state.messages):
            if msg.get("role") != "assistant":
                continue
            candidate = (msg.get("content") or "").strip()
            if len(candidate) < 100:
                continue
            lower = candidate[:40].lower()
            if any(lower.startswith(p) for p in _NON_ANSWER_PREFIXES):
                continue
            has_structure = any(s in candidate[:500] for s in _STRUCTURE_SIGNALS)
            if has_structure or len(candidate) > 200:
                final_content = candidate
                if state.verbose:
                    print(f"   \u21a9\ufe0f  Backfilled empty final_answer from prior response ({len(candidate)} chars)")
                break

        # Still empty — break to synthesis
        if len(final_content.strip()) < 20 and (state.findings or state.memory):
            if state.verbose:
                print(f"   \U0001f504 Empty final_answer with {len(state.findings)} findings \u2014 breaking to synthesis pipeline")
            tc_record = ToolCallRecord(
                tool_name="final_answer", tool_args=tool_args,
                tool_call_id=tool_call["id"],
                output="[answer lost \u2014 routing to synthesis]",
                duration_s=0, child_trace=None,
            )
            turn_record.tool_calls.append(tc_record)
            state.degenerated = True
            return _BREAK

    tc_record = ToolCallRecord(
        tool_name="final_answer", tool_args=tool_args,
        tool_call_id=tool_call["id"], output=final_content,
        duration_s=0, child_trace=None,
    )
    turn_record.tool_calls.append(tc_record)
    if state.verbose:
        print(f"   \u2705 final_answer received")
    return _finalize(final_content)


# ═══════════════════════════════════════════════════════════════════════
# GENERIC TOOL EXECUTION  (catch-all for search_web, fetch_url, etc.)
# ═══════════════════════════════════════════════════════════════════════

def handle_generic_tool(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Execute any tool via dispatch_tool_call and apply post-processing."""
    tool_name = tool_call["function"]["name"]
    if "<|" in tool_name:
        tool_name = tool_name.split("<|")[0]

    # ── Pre-dispatch: hard-block search when think() is required ──────
    # If the consecutive-search limit was already hit, block the call
    # BEFORE execution so we don't waste API calls or tokens.
    _SEARCH_TOOLS = ("search_web", "fetch_url", "read_pdf", "extract_tables", "fetch_cached", "wikipedia_lookup")
    if (state.depth > 0
            and tool_name in _SEARCH_TOOLS
            and state.consecutive_search_count >= state.MAX_CONSECUTIVE_SEARCHES):
        state.consecutive_search_count += 1
        block_msg = (
            f"\u26d4 SEARCH BLOCKED ({state.consecutive_search_count} consecutive search/fetch calls). "
            "Call NOT executed. You MUST call think() now to reflect on:\n"
            "  1. What data you have gathered so far\n"
            "  2. What is still missing\n"
            "  3. Your plan for the next steps\n"
            "After think(), you can resume searching. "
            "Do NOT call another search tool — it will also be blocked."
        )
        tc_record = ToolCallRecord(
            tool_name=tool_name, tool_args=tool_args,
            tool_call_id=tool_call["id"], output=block_msg,
            duration_s=0.0, child_trace=None,
        )
        turn_record.tool_calls.append(tc_record)
        state.messages.append({
            "role": "tool", "tool_call_id": tool_call["id"], "content": block_msg,
        })
        if state.verbose:
            print(f"       \u26d4  Search #{state.consecutive_search_count} hard-blocked (must think())")
        return _CONTINUE

    tc_start = time.time()
    child_trace = None
    try:
        output, child_trace = dispatch_tool_call(
            tool_name, tool_args,
            _depth=state.depth, model=state.model,
            reasoning_effort=state.reasoning_effort,
            _sandbox_files=state.sandbox_files,
            _memory_store=state.memory,
        )
        if state.verbose and len(output) < 200:
            print(f"       \u2192 {output}")
        elif state.verbose:
            print(f"       \u2192 {output[:200]}...")
    except Exception as e:
        output = f"ERROR: {str(e)}"
        if state.verbose:
            print(f"       \u2192 \u274c {output}")
    tc_duration = round(time.time() - tc_start, 3)

    # ── Consecutive error tracking ────────────────────────────────────
    if output.startswith("ERROR:"):
        error_sig = f"{tool_name}:{output[:200]}"
        if error_sig == state.last_error_signature:
            state.consecutive_error_count += 1
        else:
            state.consecutive_error_count = 1
            state.last_error_signature = error_sig
        if state.consecutive_error_count >= state.MAX_CONSECUTIVE_ERRORS:
            output += (
                f"\n\nFATAL: This same error has occurred {state.consecutive_error_count} times in a row. "
                "STOP retrying. Either try a completely different approach, simplify your code, "
                "or call final_answer with what you have so far."
            )
            if state.verbose:
                print(f"       \u26a0\ufe0f  Degenerate retry loop detected ({state.consecutive_error_count}x)")
    else:
        state.consecutive_error_count = 0
        state.last_error_signature = None

    # ── Snapshot raw tool output before any nudge annotations ─────────
    # All detection gates below must classify _raw_output (the tool's
    # actual return value), not `output` which accumulates nudge text
    # from earlier gates and can false-trigger later ones (e.g. the word
    # "blocked" in the consecutive-search warning triggering is_blocked).
    _raw_output = output

    # ── Consecutive search tracking & warning (sub-agents only) ───────
    # The hard-block fires above (pre-dispatch) once the limit is hit.
    # Here we track the count for calls that ARE executed and issue the
    # approaching-limit warning.
    if state.depth > 0 and tool_name in _SEARCH_TOOLS:
        state.consecutive_search_count += 1
        if state.consecutive_search_count == state.CONSECUTIVE_SEARCH_WARNING:
            output += (
                f"\n\n\u26a0\ufe0f WARNING: {state.consecutive_search_count} consecutive search/fetch calls. "
                "Call think() to reflect on what you have before your next "
                "search \u2014 the next call will be blocked until you do."
            )
    elif tool_name != "think":
        state.consecutive_search_count = 0

    # ── Empty/low-quality result detection (sub-agents) ───────────────
    if state.depth > 0 and not output.startswith(("ERROR:", "\u26d4")):
        empty_nudge = ""
        if tool_name == "search_web":
            has_urls = bool(re.search(r'URL:\s*https?://', _raw_output))
            if not has_urls or len(_raw_output.strip()) < 100:
                empty_nudge = (
                    "\n\n\u26a0\ufe0f EMPTY/POOR RESULTS: This search returned little or no useful data. "
                    "Before retrying, try:\n"
                    "  1. SIMPLIFY your query \u2014 remove quotes, dates, or specific terms\n"
                    "  2. Try SYNONYMS or alternate phrasing\n"
                    "  3. Try a DIFFERENT source type (Wikipedia, government site, etc.)\n"
                    "  4. If the data may not exist publicly, note it as a gap\n"
                    "Do NOT repeat the same query. Call think() to plan a different approach."
                )
        elif tool_name in ("fetch_url", "extract_tables", "wikipedia_lookup"):
            content_len = len(_raw_output.strip())
            # Only check for block markers in the first 500 chars (error
            # headers), NOT the page body.  Body text like "403rd Wing"
            # or an article mentioning "blocked" caused false positives
            # on perfectly valid 100K+ responses.
            _head_for_block = _raw_output[:500].lower()
            _starts_with_error = _raw_output.startswith("FETCH FAILED:")
            is_blocked = _starts_with_error or any(
                p in _head_for_block for p in [
                    "blocked", "access denied", "403", "captcha",
                    "cloudflare", "do not retry this url",
                ]
            )
            # Skip the nudge entirely when the response is large — a
            # 2 KB+ page is clearly not "blocked" regardless of
            # incidental keyword matches in its content.
            _has_real_content = content_len > 2000
            if (is_blocked or content_len < 150) and not _has_real_content:
                tool_url = tool_args.get("url", "")
                is_archive = "web.archive.org" in tool_url
                is_jina = "r.jina.ai" in tool_url or "s.jina.ai" in tool_url
                if is_archive or is_jina:
                    empty_nudge = (
                        "\n\n\u26a0\ufe0f BLOCKED/EMPTY: This is already an archived/rendered URL and it failed. "
                        "Do NOT retry it. Instead:\n"
                        "  1. Search for the same information from a COMPLETELY DIFFERENT source\n"
                        "  2. Try wikipedia_lookup() if the topic has a Wikipedia article\n"
                        "  3. Accept the gap and note it in your final_answer"
                    )
                else:
                    empty_nudge = (
                        "\n\n\u26a0\ufe0f LOW-QUALITY/BLOCKED CONTENT: This page returned very little usable data. "
                        "Do NOT retry the same URL. Instead:\n"
                        "  1. Search for the same information from a DIFFERENT source\n"
                        "  2. Try fetch_cached() for a Wayback Machine copy\n"
                        "  3. Try wikipedia_lookup() if the topic has a Wikipedia article\n"
                        "  4. Accept the gap and note it in your final_answer"
                    )
        # Don't stack LOW-QUALITY with more specific nudges that fire below
        if empty_nudge and tool_name in ("fetch_url", "extract_tables", "wikipedia_lookup"):
            _skip = []
            if "[STRUCTURED DATA:" in _raw_output:
                _skip.append("structured-data")
            if any(m in _raw_output for m in ("[PDF TRUNCATED:", "[ARTICLE TRUNCATED:", "[PAGE TRUNCATED:")):
                _skip.append("truncated")
            if "[DATA FILE DOWNLOADED:" in _raw_output:
                _skip.append("data-downloaded")
            if _skip:
                empty_nudge = ""
                if state.verbose:
                    print(f"       ↳  Skipped LOW-QUALITY nudge ({', '.join(_skip)} nudge takes priority)")
        if empty_nudge:
            output += empty_nudge
            if state.verbose:
                print(f"       \u26a0\ufe0f  Empty/low-quality result detected \u2014 nudge injected")

    # ── Shared marker list (reused by struct-data and truncation checks) ──
    _truncation_markers = ("[PDF TRUNCATED:", "[ARTICLE TRUNCATED:", "[PAGE TRUNCATED:")

    # ── Data-file sandbox injection (when _try_direct_data_download handled it) ──
    # The download function cached the text in _page_text_cache but doesn't
    # have access to state.sandbox_files.  Inject here so execute_code can
    # actually open the file.
    if (
        tool_name == "fetch_url"
        and "[DATA FILE DOWNLOADED:" in _raw_output
    ):
        _dl_url = str(tool_args.get("url", ""))
        try:
            from .tool_store import _page_text_cache, _page_text_cache_lock, _URL_CACHE_TTL
            import base64 as _b64dl
            import hashlib as _hldl
            import re as _re_dl
            with _page_text_cache_lock:
                _dlcache = _page_text_cache.get(_dl_url)
            if _dlcache and (time.time() - _dlcache[1] < _URL_CACHE_TTL):
                _dl_data = _dlcache[0]
                _dl_hash = _hldl.md5(_dl_url.encode()).hexdigest()[:8]
                # Extract the sandbox filename from output (e.g. data_abc123.csv)
                _fname_match = _re_dl.search(r"'(data_[0-9a-f]+\.\w+)'", output)
                _dl_sandbox = _fname_match.group(1) if _fname_match else f"data_{_dl_hash}.txt"
                # Try encoding as UTF-8 text first; fall back to raw (already base64 for binary)
                try:
                    _dl_enc = _b64dl.b64encode(_dl_data.encode("utf-8")).decode("ascii")
                except (UnicodeDecodeError, AttributeError):
                    _dl_enc = _dl_data  # already base64 for binary files
                if state.sandbox_files is None:
                    state.sandbox_files = {}
                state.sandbox_files[_dl_sandbox] = _dl_enc
                if state.verbose:
                    print(f"       📂  Injected data download as {_dl_sandbox} into sandbox_files")
        except Exception:
            pass

    # ── Structured data nudge (sub-agents): use code, not eyeballing ──
    # Skip if the direct-download path already handled the file, or if a
    # truncation nudge will fire below (it injects the sandbox file too).
    if (
        state.depth > 0
        and tool_name == "fetch_url"
        and "[STRUCTURED DATA:" in _raw_output
        and "[DATA FILE DOWNLOADED:" not in _raw_output
        and not any(m in output for m in _truncation_markers)
    ):
        # Check if we already injected a sandbox file from the truncation nudge below;
        # if so, reference that file name.  Otherwise, try to inject now.
        _struct_sandbox_file = None
        _struct_url = str(tool_args.get("url", ""))
        try:
            from .tool_store import _page_text_cache, _page_text_cache_lock, _URL_CACHE_TTL
            import base64 as _b64s
            import hashlib as _hls
            with _page_text_cache_lock:
                _sptc = _page_text_cache.get(_struct_url)
            if _sptc and (time.time() - _sptc[1] < _URL_CACHE_TTL):
                _sfull = _sptc[0]
                _shash = _hls.md5(_struct_url.encode()).hexdigest()[:8]
                _struct_sandbox_file = f"data_{_shash}.txt"
                _senc = _b64s.b64encode(_sfull.encode("utf-8")).decode("ascii")
                if state.sandbox_files is None:
                    state.sandbox_files = {}
                state.sandbox_files[_struct_sandbox_file] = _senc
        except Exception:
            pass

        if _struct_sandbox_file:
            output += (
                f"\n\n\u26a0\ufe0f STRUCTURED DATA — USE CODE:\n"
                f"  Full data ({len(_sfull):,} chars) is auto-loaded as '{_struct_sandbox_file}'.\n"
                f"  Do NOT read it manually. Use execute_code instead:\n"
                f"  execute_code(code=\"\"\"```python\n"
                f"  import json\n"
                f"  data = json.loads(open('{_struct_sandbox_file}').read())\n"
                f"  # Filter, aggregate, extract what you need\n"
                f"  print(json.dumps(data[:5], indent=2))  # preview first 5 entries\n"
                f"  ```\"\"\")\n"
            )
        else:
            output += (
                "\n\n\u26a0\ufe0f STRUCTURED DATA: The response is machine-readable data (JSON/CSV/XML). "
                "Do NOT try to read it manually. Instead:\n"
                "  1. Use execute_code() to load and filter with pandas/json\n"
                "  2. Parse, filter, and extract only the values you need\n"
                "  3. For large datasets, use conduct_research with memory_keys to delegate to a sub-agent"
            )
        if state.verbose:
            print(f"       \U0001f4ca  Structured data detected \u2014 code nudge injected")

    # ── Truncation nudge: auto-inject full text into sandbox + steer to code ──
    if (
        tool_name in ("read_pdf", "fetch_cached", "wikipedia_lookup", "fetch_url")
        and any(m in output for m in _truncation_markers)
    ):
        # Auto-inject full cached text into state.sandbox_files so the model's
        # execute_code calls can open() it directly in the sandbox.
        _trunc_url = str(tool_args.get("url", tool_args.get("title", "")))
        _sandbox_filename = None
        try:
            from .tool_store import _page_text_cache, _page_text_cache_lock, _URL_CACHE_TTL
            import base64 as _b64
            import hashlib as _hl
            with _page_text_cache_lock:
                _ptc = _page_text_cache.get(_trunc_url)
                # Also try wikipedia: prefix for wikipedia_lookup
                if not _ptc and tool_name == "wikipedia_lookup":
                    _ptc = _page_text_cache.get(f"wikipedia:{tool_args.get('title', '')}")
            if _ptc and (time.time() - _ptc[1] < _URL_CACHE_TTL):
                _full_text = _ptc[0]
                # Generate a short stable filename from the URL
                _url_hash = _hl.md5(_trunc_url.encode()).hexdigest()[:8]
                _sandbox_filename = f"page_{_url_hash}.txt"
                _encoded = _b64.b64encode(_full_text.encode("utf-8")).decode("ascii")
                if state.sandbox_files is None:
                    state.sandbox_files = {}
                state.sandbox_files[_sandbox_filename] = _encoded
                if state.verbose:
                    print(f"       📂  Injected {len(_full_text):,} chars as {_sandbox_filename} into sandbox_files")
        except Exception:
            pass

        # For sub-agents, add a concrete code nudge referencing the injected file
        if state.depth > 0 and _sandbox_filename:
            output += (
                f"\n\n⚠️ TRUNCATED — FULL TEXT AVAILABLE IN SANDBOX:\n"
                f"  The complete text ({len(_full_text):,} chars) has been auto-loaded as '{_sandbox_filename}'.\n"
                f"  Search it with execute_code instead of reading page-by-page:\n"
                f"  execute_code(code=\"\"\"```python\n"
                f"  import re\n"
                f"  text = open('{_sandbox_filename}').read()\n"
                f"  # Find what you need:\n"
                f"  matches = re.findall(r'your_pattern', text, re.I)\n"
                f"  print(matches)\n"
                f"  # Or search line by line:\n"
                f"  for i, line in enumerate(text.split('\\n')):\n"
                f"      if 'keyword' in line.lower(): print(f'Line {{i}}: {{line.strip()}}')\n"
                f"  ```\"\"\")\n"
            )
            if state.verbose:
                print(f"       📄  Truncated content — code strategy injected with {_sandbox_filename}")
        elif state.depth > 0:
            # Fallback nudge if injection failed
            output += (
                "\n\n⚠️ TIP: Use read_page(url=..., offset=0, max_chars=50000) to get more "
                "of the text, then use execute_code to regex/filter through it."
            )

    # ── Record in trace ───────────────────────────────────────────────
    tc_record = ToolCallRecord(
        tool_name=tool_name, tool_args=tool_args,
        tool_call_id=tool_call["id"], output=output,
        duration_s=tc_duration, child_trace=child_trace,
    )
    turn_record.tool_calls.append(tc_record)

    # ── Findings tracking + memory store ──────────────────────────────
    is_error = output.startswith("ERROR:")
    mem_key = None
    if not is_error and tool_name not in ("search_available_tools",):
        snippet = output[:1500].strip()
        if snippet:
            state.findings.append(f"[{tool_name}] {snippet}")

        desc = ""
        memory_content = output
        if tool_name == "search_web":
            desc = str(tool_args.get("q", ""))[:60]
        elif tool_name == "fetch_url":
            desc = str(tool_args.get("url", ""))[:60]
            try:
                from .tool_store import _page_text_cache, _page_text_cache_lock, _URL_CACHE_TTL
                fetch_url_str = str(tool_args.get("url", ""))
                with _page_text_cache_lock:
                    ptc = _page_text_cache.get(fetch_url_str)
                if ptc and (time.time() - ptc[1] < _URL_CACHE_TTL):
                    memory_content = ptc[0]
            except Exception:
                pass
        elif tool_name == "read_pdf":
            desc = str(tool_args.get("url", ""))[:60]
            # Store full PDF text from cache (same pattern as fetch_url)
            try:
                from .tool_store import _page_text_cache, _page_text_cache_lock, _URL_CACHE_TTL
                pdf_url_str = str(tool_args.get("url", ""))
                with _page_text_cache_lock:
                    ptc = _page_text_cache.get(pdf_url_str)
                if ptc and (time.time() - ptc[1] < _URL_CACHE_TTL):
                    memory_content = ptc[0]
            except Exception:
                pass
        elif tool_name == "wikipedia_lookup":
            desc = str(tool_args.get("title", ""))[:60]
            # Store full Wikipedia text from cache
            try:
                from .tool_store import _page_text_cache, _page_text_cache_lock, _URL_CACHE_TTL
                wiki_key = f"wikipedia:{tool_args.get('title', '')}"
                with _page_text_cache_lock:
                    ptc = _page_text_cache.get(wiki_key)
                if ptc and (time.time() - ptc[1] < _URL_CACHE_TTL):
                    memory_content = ptc[0]
            except Exception:
                pass
        elif tool_name == "fetch_cached":
            desc = str(tool_args.get("url", ""))[:60]
            # Store full Wayback text from cache
            try:
                from .tool_store import _page_text_cache, _page_text_cache_lock, _URL_CACHE_TTL
                cached_url_str = str(tool_args.get("url", ""))
                with _page_text_cache_lock:
                    ptc = _page_text_cache.get(cached_url_str)
                if ptc and (time.time() - ptc[1] < _URL_CACHE_TTL):
                    memory_content = ptc[0]
            except Exception:
                pass
        elif tool_name == "spawn_agent":
            desc = str(tool_args.get("task", ""))[:60]
        mem_key = state.memory.add(
            tool_name=tool_name, turn=state.turn,
            content=memory_content, description=desc,
        )

    # ── Plan tracking (root only) ─────────────────────────────────────
    if state.plan is not None and tool_name not in ("search_available_tools",):
        state.plan.record_tool_call(
            tool_name=tool_name, tool_args=tool_args,
            output=output, memory_key=mem_key, is_error=is_error,
        )

    # ── Build message (strip base64, apply symbolic compression) ──────
    msg_output = re.sub(
        r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
        r"\1[file content saved to trace]\2",
        output, flags=re.DOTALL,
    )
    if (
        _cfg.SYMBOLIC_REFERENCES
        and state.depth == 0
        and mem_key is not None
        and len(msg_output) > _cfg.SYMBOLIC_THRESHOLD
    ):
        msg_output = make_symbolic(
            tool_name=tool_name, tool_args=tool_args,
            output=msg_output, memory_key=mem_key,
        )
        if state.verbose:
            print(f"       \U0001f4ce Symbolic ref: {len(output):,} \u2192 {len(msg_output):,} chars [{mem_key}]")

    state.messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": str(msg_output),
    })
    return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# DISPATCH TABLE
# ═══════════════════════════════════════════════════════════════════════

TOOL_HANDLERS.update({
    "think":                    handle_think,
    "search_available_tools":   handle_search_available_tools,
    "refine_draft":             handle_refine_draft,
    "read_draft":               handle_read_draft,
    "research_complete":        handle_research_complete,
    "conduct_research":         handle_conduct_research,
    "summarize_webpage":        handle_summarize_webpage,
    "compress_findings":        handle_compress_findings,
    "final_answer":             handle_final_answer,
})
