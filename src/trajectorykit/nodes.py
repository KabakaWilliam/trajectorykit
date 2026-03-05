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
from typing import Any, Dict, Optional, Tuple

from .agent_state import AgentState
from .config import (
    VLLM_API_URL,
    SYMBOLIC_REFERENCES,
    SYMBOLIC_THRESHOLD,
    VERIFY_BEFORE_PUBLISH,
    VERIFIER_PROMPT,
    SPOT_CHECK_ENABLED,
    SPOTCHECK_EXTRACT_PROMPT,
    SPOTCHECK_COMPARE_PROMPT,
    SPOTCHECK_REFUSAL_PROMPT,
)
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
# RESEARCH_COMPLETE  (root only — verification + spot-check pipeline)
# ═══════════════════════════════════════════════════════════════════════

def _focused_page_summary(
    url: str,
    focus: str,
    model: str,
    vllm_url: str,
    max_fetch_chars: int = 200_000,
    max_summary_tokens: int = 4096,
) -> str:
    """Fetch a URL and produce an LLM-generated summary.

    Shared by summarize_webpage and the spot-check evidence pipeline.
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
                "  - Be concise but never drop quantitative data"
                f"{focus_instruction}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Summarize the following webpage content from: {url}\n\n"
                f"--- PAGE CONTENT ---\n{page_content}\n--- END ---\n\n"
                "Produce a structured summary with the key information."
            ),
        },
    ]
    try:
        resp = requests.post(
            f"{vllm_url}/chat/completions",
            json={
                "model": model,
                "messages": msgs,
                "temperature": 0.3,
                "max_tokens": max_summary_tokens,
            },
        )
        if resp.status_code == 200:
            choices = resp.json().get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                if text and text.strip():
                    return text.strip()
                return f"LLM returned empty summary. Raw page excerpt:\n{page_content[:3000]}"
            return f"LLM returned no choices. Raw page excerpt:\n{page_content[:3000]}"
        return (
            f"Summarization LLM call failed (HTTP {resp.status_code}). "
            f"Raw page excerpt:\n{page_content[:3000]}"
        )
    except Exception as e:
        return f"Summarization error: {e}. Raw page excerpt:\n{page_content[:3000]}"


def _run_verification(state: AgentState, draft_content: str) -> Tuple[bool, str, str]:
    """Run Stage-1 LLM verifier.  Returns (should_publish, feedback, raw_text)."""
    if (
        not VERIFY_BEFORE_PUBLISH
        or not VERIFIER_PROMPT
        or state.verification_rejections >= state.MAX_VERIFICATION_REJECTIONS
    ):
        if state.verification_rejections >= state.MAX_VERIFICATION_REJECTIONS and state.verbose:
            print(f"       \u23e9 Verifier: max rejections reached, force-publishing")
        return True, "", ""

    try:
        # Build findings summary for cross-checking
        findings_section = ""
        if state.findings:
            lines = []
            total = 0
            for i, f in enumerate(state.findings, 1):
                entry = f"{i}. {f}"
                if total + len(entry) > 15000:
                    lines.append(f"... ({len(state.findings) - i + 1} more entries truncated)")
                    break
                lines.append(entry)
                total += len(entry)
            findings_section = (
                "\n\nRESEARCH FINDINGS (raw worker outputs for cross-checking):\n"
                + "\n".join(lines)
            )
        verify_messages = [
            {"role": "system", "content": VERIFIER_PROMPT},
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
            "model": state.model,
            "messages": verify_messages,
            "temperature": state.temperature,
            "max_tokens": state.max_tokens,
        }
        if state.verbose:
            print(f"       \U0001f50d Verifier: auditing draft (attempt {state.verification_rejections + 1})...")
        resp = requests.post(f"{VLLM_API_URL}/chat/completions", json=payload)
        if resp.status_code == 200:
            result = resp.json()
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()
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
                return False, feedback, text
            else:
                if state.verbose:
                    print(f"       \u2705 Verifier: APPROVED")
                return True, "", text
        else:
            if state.verbose:
                print(f"       \u26a0\ufe0f  Verifier API error ({resp.status_code}), skipping verification")
            return True, "", ""
    except Exception as e:
        if state.verbose:
            print(f"       \u26a0\ufe0f  Verifier error: {e}, skipping verification")
        return True, "", ""


def _run_spot_check(state: AgentState, draft_content: str) -> Tuple[bool, str, str, dict]:
    """Run Stage-2 spot-check (search-verify key claims).

    Returns (should_publish, feedback, raw_text, metadata_dict).
    """
    meta: dict = {}
    if (
        not SPOT_CHECK_ENABLED
        or not SPOTCHECK_EXTRACT_PROMPT
        or not SPOTCHECK_COMPARE_PROMPT
        or state.spot_check_rejections >= state.MAX_SPOT_CHECK_REJECTIONS
    ):
        if state.spot_check_rejections >= state.MAX_SPOT_CHECK_REJECTIONS and state.verbose:
            print(f"       \u23e9 Spot-check: max rejections reached, skipping")
        return True, "", "", meta

    try:
        if state.verbose:
            print(f"       \U0001f50e Spot-check: extracting checkable claims...")

        # Step 1: Extract claims from draft
        sc_extract_msgs = [
            {"role": "system", "content": SPOTCHECK_EXTRACT_PROMPT},
            {"role": "user", "content": (
                f"QUESTION:\n{state.user_input}\n\n"
                f"DRAFT ANSWER:\n{draft_content}"
            )},
        ]
        sc_extract_resp = requests.post(
            f"{VLLM_API_URL}/chat/completions",
            json={
                "model": state.model,
                "messages": sc_extract_msgs,
                "temperature": 0.3,
                "max_tokens": 1024,
            },
        )
        sc_claims = []
        if sc_extract_resp.status_code == 200:
            sc_extract_text = (
                sc_extract_resp.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "")
            ).strip()
            meta["extract_response"] = sc_extract_text
            sc_json_text = re.sub(r'^```(?:json)?\s*', '', sc_extract_text, flags=re.MULTILINE)
            sc_json_text = re.sub(r'```\s*$', '', sc_json_text, flags=re.MULTILINE).strip()
            try:
                sc_parsed = json.loads(sc_json_text)
                sc_claims = sc_parsed.get("claims", [])
            except json.JSONDecodeError:
                if state.verbose:
                    print(f"       \u26a0\ufe0f  Spot-check: failed to parse claims JSON, skipping")
        else:
            if state.verbose:
                print(f"       \u26a0\ufe0f  Spot-check: extract API error ({sc_extract_resp.status_code}), skipping")

        if sc_claims:
            return _spot_check_claims(state, draft_content, sc_claims, meta)

        # No checkable claims — check if it's a refusal
        return _spot_check_refusal(state, draft_content, meta)

    except Exception as e:
        if state.verbose:
            print(f"       \u26a0\ufe0f  Spot-check error: {e}, skipping")
        return True, "", "", meta


def _spot_check_claims(
    state: AgentState,
    draft_content: str,
    sc_claims: list,
    meta: dict,
) -> Tuple[bool, str, str, dict]:
    """Verify extracted claims via search + LLM comparison."""
    if state.verbose:
        print(f"       \U0001f50e Spot-check: verifying {len(sc_claims)} claim(s)...")

    def _verify_one_claim(claim_obj):
        _q = claim_obj.get("search_query", "")
        _c = claim_obj.get("claim", "")
        if not _q:
            return None
        try:
            _sr, _ = dispatch_tool_call("search_web", {"q": _q, "num_results": 5}, _depth=0)
            _ps = ""
            _um = re.search(r'URL:\s*(https?://\S+)', _sr)
            if _um:
                _top_url = _um.group(1)
                try:
                    _ps = _focused_page_summary(
                        url=_top_url,
                        focus=f"Verify this claim: {_c}",
                        model=state.model,
                        vllm_url=VLLM_API_URL,
                        max_summary_tokens=1024,
                    )
                    if _ps.startswith(("Failed to fetch", "Page fetched but", "Summarization error")):
                        _ps = ""
                    elif state.verbose:
                        print(f"       \U0001f4c4 Spot-check: summarized {_top_url[:60]} ({len(_ps)} chars)")
                except Exception:
                    pass
            return {
                "claim": _c,
                "search_query": _q,
                "search_results": _sr[:3000],
                "page_content": _ps,
            }
        except Exception as e:
            if state.verbose:
                print(f"       \u26a0\ufe0f  Spot-check search error: {e}")
            return {
                "claim": _c,
                "search_query": _q,
                "search_results": f"(search failed: {e})",
                "page_content": "",
            }

    max_workers = min(len(sc_claims), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_verify_one_claim, c): c for c in sc_claims}
        sc_evidence = []
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                sc_evidence.append(result)

    meta["claims_checked"] = len(sc_evidence)

    # Short-circuit: if >50% degraded evidence, auto-PASS
    if sc_evidence:
        degraded = sum(
            1 for ev in sc_evidence
            if ev["search_results"].startswith("(search failed")
            or (not ev["page_content"] and len(ev["search_results"]) < 200)
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
    evidence_parts = []
    for i, ev in enumerate(sc_evidence, 1):
        part = (
            f"CLAIM {i}: {ev['claim']}\n"
            f"SEARCH QUERY: {ev['search_query']}\n"
            f"SEARCH RESULTS:\n{ev['search_results']}"
        )
        if ev.get("page_content"):
            part += f"\n\nSUMMARY OF TOP RESULT:\n{ev['page_content']}"
        evidence_parts.append(part)
    evidence_text = "\n\n".join(evidence_parts)

    compare_msgs = [
        {"role": "system", "content": SPOTCHECK_COMPARE_PROMPT},
        {"role": "user", "content": (
            f"QUESTION:\n{state.user_input}\n\n"
            f"DRAFT ANSWER (excerpt of checked claims):\n\n"
            f"{evidence_text}"
        )},
    ]
    compare_resp = requests.post(
        f"{VLLM_API_URL}/chat/completions",
        json={
            "model": state.model,
            "messages": compare_msgs,
            "temperature": 0.3,
            "max_tokens": 2048,
        },
    )
    if compare_resp.status_code == 200:
        compare_text = (
            compare_resp.json().get("choices", [{}])[0]
            .get("message", {}).get("content", "")
        ).strip()
        meta["compare_response"] = compare_text
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

    if not is_refusal or not SPOTCHECK_REFUSAL_PROMPT:
        if state.verbose and not is_refusal:
            print(f"       \u2139\ufe0f  Spot-check: no checkable claims extracted, skipping")
        return True, "", "", meta

    if state.verbose:
        print(f"       \U0001f50e Spot-check: draft looks like a refusal \u2014 launching refusal challenge...")

    rc_query = state.user_input[:200]
    try:
        rc_search_result, _ = dispatch_tool_call("search_web", {"q": rc_query, "num_results": 5}, _depth=0)
        rc_page_summary = ""
        rc_url_match = re.search(r'URL:\s*(https?://\S+)', rc_search_result)
        if rc_url_match:
            rc_top_url = rc_url_match.group(1)
            try:
                rc_page_summary = _focused_page_summary(
                    url=rc_top_url,
                    focus=f"Find information relevant to: {state.user_input[:300]}",
                    model=state.model,
                    vllm_url=VLLM_API_URL,
                    max_summary_tokens=1024,
                )
                if rc_page_summary.startswith(("Failed to fetch", "Page fetched but", "Summarization error")):
                    rc_page_summary = ""
            except Exception:
                pass

        rc_evidence_text = f"SEARCH RESULTS:\n{rc_search_result[:4000]}"
        if rc_page_summary:
            rc_evidence_text += f"\n\nSUMMARY OF TOP RESULT:\n{rc_page_summary}"

        rc_msgs = [
            {"role": "system", "content": SPOTCHECK_REFUSAL_PROMPT},
            {"role": "user", "content": (
                f"QUESTION:\n{state.user_input}\n\n"
                f"DRAFT ANSWER (REFUSAL):\n{draft_content[:2000]}\n\n"
                f"INDEPENDENT EVIDENCE:\n{rc_evidence_text}"
            )},
        ]
        rc_resp = requests.post(
            f"{VLLM_API_URL}/chat/completions",
            json={
                "model": state.model,
                "messages": rc_msgs,
                "temperature": 0.3,
                "max_tokens": 1024,
            },
        )
        if rc_resp.status_code == 200:
            rc_text = (
                rc_resp.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "")
            ).strip()
            meta["refusal_challenge_response"] = rc_text
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


def handle_research_complete(
    state: AgentState,
    tool_call: dict,
    tool_args: dict,
    turn_record: TurnRecord,
    **kw,
) -> NodeResult:
    """Publish the draft — runs verification + spot-check pipeline."""
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
    should_publish, verify_feedback, verify_text_raw = _run_verification(state, draft_content)

    # ── Stage 2: Spot-check ──────────────────────────────────────────
    spot_check_meta: dict = {}
    if should_publish:
        should_publish, sc_feedback, sc_raw, spot_check_meta = _run_spot_check(state, draft_content)
        if not should_publish:
            verify_feedback = sc_feedback
            verify_text_raw = sc_raw

    # ── Publish or reject ────────────────────────────────────────────
    if should_publish:
        verify_meta: dict = {}
        if verify_text_raw:
            verify_meta["verification_verdict"] = "APPROVED"
            verify_meta["verification_response"] = verify_text_raw
        verify_meta["verification_attempt"] = state.verification_rejections + 1
        if spot_check_meta:
            verify_meta["spot_check"] = spot_check_meta
        tc_record = ToolCallRecord(
            tool_name="research_complete", tool_args=tool_args,
            tool_call_id=tool_call["id"], output=draft_content,
            duration_s=0, child_trace=None,
            metadata=verify_meta if verify_meta else None,
        )
        turn_record.tool_calls.append(tc_record)
        if state.verbose:
            print(f"   \u2705 research_complete \u2014 publishing draft ({len(draft_content):,} chars)")
        return _finalize(draft_content)
    else:
        is_spot_check_reject = verify_text_raw.startswith("[SPOT-CHECK]")
        if is_spot_check_reject:
            reject_msg = (
                f"Spot-check found factual issues in your draft that should be corrected:\n\n"
                f"{verify_feedback}\n\n"
                f"Please correct these specific claims with refine_draft(), then call "
                f"research_complete() again. "
                f"(Spot-check rejection {state.spot_check_rejections}/{state.MAX_SPOT_CHECK_REJECTIONS} \u2014 "
                f"{'next attempt will force-publish' if state.spot_check_rejections >= state.MAX_SPOT_CHECK_REJECTIONS else f'{state.MAX_SPOT_CHECK_REJECTIONS - state.spot_check_rejections} correction(s) remaining'})"
            )
        else:
            reject_msg = (
                f"Draft review found issues that should be addressed before publishing:\n\n"
                f"{verify_feedback}\n\n"
                f"Please fix these issues with refine_draft(), then call research_complete() again. "
                f"(Attempt {state.verification_rejections}/{state.MAX_VERIFICATION_REJECTIONS} \u2014 "
                f"{'next attempt will force-publish' if state.verification_rejections >= state.MAX_VERIFICATION_REJECTIONS else f'{state.MAX_VERIFICATION_REJECTIONS - state.verification_rejections} revision(s) remaining'})"
            )
        reject_meta = {
            "verification_verdict": "SPOT_CHECK_FAILED" if is_spot_check_reject else "REVISION_NEEDED",
            "verification_response": verify_text_raw or verify_feedback,
            "verification_attempt": state.verification_rejections,
            "spot_check_attempt": state.spot_check_rejections,
        }
        if spot_check_meta:
            reject_meta["spot_check"] = spot_check_meta
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
        return _CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# CONDUCT_RESEARCH  (root only — delegates to sub-agent)
# ═══════════════════════════════════════════════════════════════════════

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

    # ── Task-level deduplication: inject prior findings if similar ─────
    new_task = str(tool_args.get("task", ""))
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
        state.findings.append(f"[conduct_research] {output[:1500].strip()}")
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

    # Build the message for context
    msg_output = re.sub(
        r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
        r"\1[file content saved to trace]\2",
        output, flags=re.DOTALL,
    )
    if SYMBOLIC_REFERENCES and mem_key is not None and len(msg_output) > SYMBOLIC_THRESHOLD:
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
                vllm_url=VLLM_API_URL,
                max_summary_tokens=min(4096, state.max_tokens or 4096),
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
    if SYMBOLIC_REFERENCES and mem_key is not None and len(msg_output) > SYMBOLIC_THRESHOLD:
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
                        "  - If a finding is well-supported by multiple sources, note that"
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
                "temperature": 0.3,
                "max_tokens": min(8192, state.max_tokens),
            }
            cf_resp = requests.post(f"{VLLM_API_URL}/chat/completions", json=cf_payload)
            if cf_resp.status_code == 200:
                cf_result = cf_resp.json()
                cf_choices = cf_result.get("choices", [])
                if cf_choices:
                    cf_output = cf_choices[0].get("message", {}).get("content", "")
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
    """Handle final_answer — terminal for sub-agents, redirect for root."""

    # ── Root: redirect to refine_draft + research_complete flow ──
    if state.depth == 0:
        fa_content = tool_args.get("answer", "").strip()
        if fa_content and len(fa_content) >= 20:
            state.draft_versions.append((state.turn, fa_content))
            ver = len(state.draft_versions)
            if state.draft_path:
                try:
                    with open(state.draft_path, "w", encoding="utf-8") as f:
                        f.write(f"<!-- Draft v{ver} | turn {state.turn} -->\n")
                        f.write(fa_content)
                except Exception:
                    pass
            state.memory.upsert(
                key="draft_latest", tool_name="refine_draft",
                turn=state.turn, content=fa_content,
                description="latest draft answer",
            )
            redirect_msg = (
                f"\u2705 Draft v{ver} saved ({len(fa_content):,} chars).\n\n"
                "NOTE: You called final_answer, but you should use "
                "research_complete() to publish. This runs a quality "
                "review before publishing. Call research_complete() now."
            )
        else:
            redirect_msg = (
                "ERROR: final_answer is not available at root level. "
                "Use refine_draft(content='...') to write your answer, "
                "then research_complete() to publish it."
            )
        tc_record = ToolCallRecord(
            tool_name="final_answer", tool_args=tool_args,
            tool_call_id=tool_call["id"], output=redirect_msg,
            duration_s=0, child_trace=None,
        )
        turn_record.tool_calls.append(tc_record)
        state.messages.append({
            "role": "tool", "tool_call_id": tool_call["id"], "content": redirect_msg,
        })
        if state.verbose:
            print(f"   \u21a9\ufe0f  Root called final_answer \u2014 redirected to draft v{len(state.draft_versions)}")
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

    # ── Consecutive search enforcement (sub-agents only) ──────────────
    _SEARCH_TOOLS = ("search_web", "fetch_url", "read_pdf", "extract_tables", "fetch_cached", "wikipedia_lookup")
    if state.depth > 0 and tool_name in _SEARCH_TOOLS:
        state.consecutive_search_count += 1
        if state.consecutive_search_count > state.MAX_CONSECUTIVE_SEARCHES:
            output = (
                f"\u26d4 SEARCH LIMIT: {state.consecutive_search_count} consecutive search/fetch calls. "
                "Output suppressed. You MUST call think() now to reflect on:\n"
                "  1. What data you have gathered so far\n"
                "  2. What is still missing\n"
                "  3. Your plan for the next steps\n"
                "After think(), you can resume searching."
            )
            if state.verbose:
                print(f"       \u26d4  Consecutive search #{state.consecutive_search_count} \u2014 blocked, must think()")
        elif state.consecutive_search_count == state.CONSECUTIVE_SEARCH_WARNING:
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
            has_urls = bool(re.search(r'URL:\s*https?://', output))
            if not has_urls or len(output.strip()) < 100:
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
            content_len = len(output.strip())
            is_blocked = any(p in output.lower() for p in [
                "blocked", "access denied", "403", "captcha",
                "cloudflare", "do not retry this url",
            ])
            if is_blocked or content_len < 150:
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
        if empty_nudge:
            output += empty_nudge
            if state.verbose:
                print(f"       \u26a0\ufe0f  Empty/low-quality result detected \u2014 nudge injected")

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
        SYMBOLIC_REFERENCES
        and state.depth == 0
        and mem_key is not None
        and len(msg_output) > SYMBOLIC_THRESHOLD
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
    "research_complete":        handle_research_complete,
    "conduct_research":         handle_conduct_research,
    "summarize_webpage":        handle_summarize_webpage,
    "compress_findings":        handle_compress_findings,
    "final_answer":             handle_final_answer,
})
