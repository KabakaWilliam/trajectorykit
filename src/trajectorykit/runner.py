"""
Agent turn loop and post-loop synthesis pipeline.

Replaces the while-loop body and post-loop code from the old monolithic
dispatch(). The public entry point is ``run_agent_loop(state) -> dict``.
"""

from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .agent_state import AgentState
from . import config as _cfg
from .nodes import (
    TOOL_HANDLERS,
    handle_generic_tool,
    _CONTINUE,
)
from .tool_store import TOOLS
from .tracing import TurnRecord, ToolCallRecord

import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# TOOL NAME SANITIZATION (hallucination cleanup)
# ═══════════════════════════════════════════════════════════════════════

# Common suffixes the model hallucinates onto tool names
_HALLUCINATED_SUFFIXES = ("commentaryjson", "commentary", "json")
# Noise characters the model appends to tool names
_STRIP_CHARS = "][)(}{'\"\n\r\t .…"


def _sanitize_tool_name(raw_name: str, state: AgentState) -> str:
    """Strip common hallucinated suffixes from tool names.

    Models sometimes append 'json', 'commentary', brackets, etc.
    to tool names. This fuzzy-matches them back to valid names.
    Returns the cleaned name if a match is found, otherwise the
    original (stripped of trailing noise characters).
    """
    name = raw_name.strip(_STRIP_CHARS)

    # Build valid name set from available tools + dispatch table
    valid_names = {t["function"]["name"] for t in state.available_tools}
    valid_names.update(TOOL_HANDLERS.keys())

    if name in valid_names:
        return name

    for suffix in _HALLUCINATED_SUFFIXES:
        if name.endswith(suffix):
            candidate = name[: -len(suffix)]
            if candidate in valid_names:
                return candidate

    return name


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (moved here from old agent.py)
# ═══════════════════════════════════════════════════════════════════════

def _recover_final_answer_from_raw(raw_args: str) -> str | None:
    """Try to extract the answer text from a malformed JSON arguments string.

    The model often produces *almost* valid JSON for final_answer — e.g. a
    trailing quote is missing, or there's an unescaped newline.  We try
    several regex strategies to pull out the answer value.

    Returns the recovered answer string, or None if nothing useful found.
    """
    # Strategy 1: everything after "answer": " up to the last quote-like boundary
    m = re.search(r'"answer"\s*:\s*"(.*)', raw_args, re.DOTALL)
    if m:
        candidate = m.group(1)
        candidate = re.sub(r'"\s*\}?\s*$', '', candidate)
        for esc, repl in [('\\n', '\n'), ('\\t', '\t'), ('\\"', '"'), ('\\\\', '\\')]:
            candidate = candidate.replace(esc, repl)
        if len(candidate.strip()) >= 20:
            return candidate.strip()

    # Strategy 2: the whole raw_args might just be the answer text (no JSON wrapper)
    stripped = raw_args.strip().strip('"').strip()
    if len(stripped) >= 50 and '{' not in stripped[:5]:
        return stripped

    return None


def _sanitize_raw_args(raw: str) -> dict | None:
    """Try to recover valid JSON tool args from common model noise.

    Handles patterns produced by models that wrap arguments in function-call
    syntax (e.g. ``({...})`` instead of ``{...}``) or emit trailing garbage
    after the closing brace.

    Strategies (tried in order):
      1. Strip trailing parens / whitespace and re-parse.
      2. Find the outermost balanced ``{ ... }`` and parse that.

    Returns the parsed dict on success, ``None`` on failure.
    """
    # Strategy 1: strip trailing noise  ─  covers  {…})  {…})\n  {…}) )
    stripped = raw.rstrip(") \t\r\n")
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract the outermost balanced { … } block
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    end = -1
    for i in range(start, len(raw)):
        c = raw[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end > start:
        try:
            return json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _extract_final_answer(assistant_msg: dict) -> str:
    """Extract the answer from a final_answer tool call in an assistant message.

    Handles cases where tool_calls is None/empty, arguments is malformed, etc.
    Falls back to assistant content if no tool call found.
    """
    tool_calls = assistant_msg.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        if "<|" in name:
            name = name.split("<|")[0]
        if name == "final_answer":
            raw_args = func.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
            answer = args.get("answer", "")
            if answer:
                return answer
    # Fallback: check content field
    return assistant_msg.get("content", "") or ""


def _extract_tool_results(messages: list, max_chars: int = 4000) -> str:
    """Extract recent tool results from message history for context."""
    tool_results = []
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))[:500]
            if content.strip() and content != "None":
                tool_results.append(content)
            if sum(len(r) for r in tool_results) > max_chars:
                break
    tool_results.reverse()
    return "\n---\n".join(tool_results)


def _build_fallback_response(messages: list, findings: Optional[list] = None) -> str:
    """Build a response from message history when synthesis fails.

    If findings are provided (from state.findings), prefer them over
    scanning messages — they contain the actual research snippets
    gathered during the turn loop.
    """
    # First: check if any assistant message contains an actual final_answer
    # tool call with content (strict — no content fallback here).
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            func = tc.get("function", {})
            name = func.get("name", "").split("<|")[0]
            if name == "final_answer":
                try:
                    args = json.loads(func.get("arguments", "{}") or "{}")
                except json.JSONDecodeError:
                    args = {}
                answer = args.get("answer", "")
                if answer.strip():
                    return answer

    # Second: use accumulated findings if available — these contain
    # the real research data gathered by search/fetch/wikipedia tools.
    # Always preferred over assistant content (which is often monologue).
    if findings:
        findings_text = "\n".join(findings[-20:])  # most recent 20
        if len(findings_text) > 6000:
            findings_text = findings_text[:6000] + "\n... (truncated)"
        if findings_text.strip():
            return (
                "[Auto-condensed from research findings]\n\n"
                + findings_text
            )

    # Third: check assistant content
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            if content.strip():
                return content

    # Fourth: concatenate the last few tool results as raw findings
    tool_results = _extract_tool_results(messages, max_chars=3000)
    if tool_results.strip():
        return f"[Auto-extracted from research — synthesis failed]\n\n{tool_results}"

    return ""


# ═══════════════════════════════════════════════════════════════════════
# API CALL HELPER
# ═══════════════════════════════════════════════════════════════════════

def call_api(
    state: AgentState,
    effective_max_tokens: int,
    tools_override: Optional[List[dict]] = None,
) -> requests.Response:
    """Build payload and call the chat completions API."""
    payload = {
        "model": state.model,
        "messages": state.messages,
        "tools": tools_override if tools_override is not None else state.available_tools,
        "tool_choice": "required",
        "temperature": state.temperature,
        "max_tokens": effective_max_tokens,
    }
    if state.reasoning_effort and state.profile.get("supports_reasoning_effort"):
        payload["reasoning_effort"] = state.reasoning_effort
    chat_template_kwargs = state.profile.get("chat_template_kwargs")
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    return requests.post(f"{_cfg.VLLM_API_URL}/chat/completions", json=payload)


# ═══════════════════════════════════════════════════════════════════════
# FINALIZE HELPER
# ═══════════════════════════════════════════════════════════════════════

def _finalize(state: AgentState, final_content: str) -> Dict[str, Any]:
    """Build the return dict and finalize the trace."""
    state.episode.final_response = final_content
    state.episode.total_turns = state.turn
    state.episode.total_tool_calls = state.total_tool_calls
    state.episode.ended_at = datetime.now().isoformat()
    state.episode.duration_s = round(time.time() - state.episode_start, 3)
    state.episode.compute_recursive_stats()
    # Attach chain plan snapshot to the trace
    if state.chain_plan is not None:
        state.episode.chain_plan = state.chain_plan.to_dict()
    return {
        "final_response": final_content,
        "turns": state.turn,
        "tool_calls": state.total_tool_calls,
        "messages": state.messages,
        "trace": state.episode,
    }


# ═══════════════════════════════════════════════════════════════════════
# HISTORY COMPACTION  (evict stale messages once draft absorbs them)
# Settings loaded from config: HISTORY_COMPACTION_*
# ═══════════════════════════════════════════════════════════════════════


def _compact_history(state: AgentState) -> None:
    """Replace old messages with a compact summary, keeping recent turns.

    Preserves:
      - msg[0] (system prompt) and msg[1] (user question)
      - A synthetic summary of evicted messages
      - The last ``_RECENT_TURNS_KEEP`` turn-groups (assistant + tool msgs)

    Everything else living on ``state`` (chain_plan, memory, findings,
    draft_versions) is untouched — they live on Python objects, not in
    the message list.  The chain plan and draft are re-injected every
    turn by ``_inject_pre_turn``, so evicting their old injections is
    safe.
    """
    msgs = state.messages
    if len(msgs) < _cfg.HISTORY_COMPACTION_MSG_THRESHOLD:
        return  # nothing to do

    # ── Find the "recent window" boundary ─────────────────────────────
    #  Walk backwards counting assistant messages (each one starts a
    #  turn-group of assistant + subsequent tool/system messages).
    keep_from = len(msgs)  # index from which we keep everything
    asst_seen = 0
    recent_keep = _cfg.HISTORY_COMPACTION_RECENT_TURNS
    for i in range(len(msgs) - 1, 1, -1):  # skip msg[0] and msg[1]
        if msgs[i].get("role") == "assistant":
            asst_seen += 1
            if asst_seen >= recent_keep:
                keep_from = i
                break
    # Safety: never evict if we can't find enough turns
    if keep_from <= 2:
        return

    evicted = msgs[2:keep_from]  # everything between header and recent
    if not evicted:
        return

    # ── Build the synthetic summary ───────────────────────────────────
    tool_counts: dict[str, int] = {}
    mem_keys: list[str] = []
    evicted_turns = set()
    for m in evicted:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "assistant":
            for tc in m.get("tool_calls", []):
                tname = tc.get("function", {}).get("name", "?")
                tool_counts[tname] = tool_counts.get(tname, 0) + 1
        elif role == "tool":
            # Capture any [Stored → key] references
            import re as _re_compact
            for match in _re_compact.finditer(r'\[Stored → ([^\]]+)\]', content):
                mem_keys.append(match.group(1))

    # Tool call summary line
    tool_parts = [f"{cnt}x {name}" for name, cnt in sorted(tool_counts.items(), key=lambda x: -x[1])]
    tool_line = ", ".join(tool_parts) if tool_parts else "(none)"

    # Memory keys line
    mem_line = ", ".join(mem_keys[:15]) if mem_keys else "(none stored)"
    if len(mem_keys) > 15:
        mem_line += f" … and {len(mem_keys) - 15} more"

    # Chain status (rendered from live object, NOT from old messages)
    chain_line = ""
    if state.chain_plan is not None and state.chain_plan.has_chain:
        parts = []
        for s in state.chain_plan.chain_steps:
            if s.is_resolved and s.resolved_value:
                parts.append(f"Step {s.step} ✅ \"{s.resolved_value[:40]}\"")
            else:
                parts.append(f"Step {s.step} {'🔓' if s.is_unlocked else '🔒'}")
        chain_line = f"\nChain progress: {' → '.join(parts)}"

    # Draft status
    draft_line = ""
    if state.draft_versions:
        _dv = len(state.draft_versions)
        _dt, _dd = state.draft_versions[-1]
        draft_line = (
            f"\nDraft status: v{_dv} ({len(_dd):,} chars, saved turn {_dt}) "
            f"already incorporates findings from evicted history."
        )

    # Research log digest (what was actually found, not just tool counts)
    research_log_line = ""
    if state.research_log:
        research_log_line = "\n\n" + _render_research_log(state)

    summary = (
        f"[HISTORY COMPRESSED — earlier messages archived]\n"
        f"Tool calls in archived turns: {tool_line}\n"
        f"Memory keys available: {mem_line}"
        f"{chain_line}"
        f"{draft_line}"
        f"{research_log_line}\n\n"
        f"All research data remains accessible via memory_keys in "
        f"conduct_research(). Your draft is the canonical summary of "
        f"prior findings — focus on remaining gaps."
    )

    # ── Splice ────────────────────────────────────────────────────────
    recent = msgs[keep_from:]
    state.messages = [
        msgs[0],                                        # system prompt
        msgs[1],                                        # user question
        {"role": "system", "content": summary},          # compressed history
    ] + recent

    if state.verbose:
        evicted_chars = sum(len(m.get("content", "") or "") for m in evicted)
        print(
            f"📦  History compacted: {len(evicted)} messages ({evicted_chars:,} chars) "
            f"→ 1 summary ({len(summary):,} chars). "
            f"Keeping {len(recent)} recent messages."
        )


# ═══════════════════════════════════════════════════════════════════════
# PRE-TURN INJECTIONS  (plan, draft, reminders, budget warnings)
# ═══════════════════════════════════════════════════════════════════════


def _render_research_log(state: AgentState) -> str:
    """Render the research log as a compact string for prompt injection."""
    if not state.research_log:
        return ""
    lines = [f"📋 RESEARCH LOG ({len(state.research_log)} tasks completed):"]
    for entry in state.research_log:
        key_tag = f" → {entry['mem_key']}" if entry.get("mem_key") else ""
        finding = entry.get("finding", "")
        # First line of finding only, truncated
        first_line = finding.split("\n")[0][:120]
        lines.append(
            f"  T{entry['turn']}  {entry['tool']} \"{entry['task']}\"{key_tag}"
        )
        if first_line:
            lines.append(f"      → {first_line}")
    # Append available memory keys summary
    mem_keys = state.memory.keys()
    if mem_keys:
        keys_str = ", ".join(mem_keys[:15])
        if len(mem_keys) > 15:
            keys_str += f" … +{len(mem_keys) - 15} more"
        lines.append(f"\nAvailable memory keys: {keys_str}")
    return "\n".join(lines)


def _inject_pre_turn(state: AgentState) -> Optional[List[dict]]:
    """Inject periodic plan, draft, reminders, and budget warnings.

    Returns a tools_override list if tools should be restricted this turn,
    or None to use the full available_tools.
    """
    tools_for_turn: Optional[List[dict]] = None

    # ── Chain plan injection (root, early turns) ──────────────────────
    if (state.depth == 0 and state.chain_plan is not None
            and state.chain_plan.has_chain and state.turn <= 2):
        chain_msg = state.chain_plan.render()
        if chain_msg:
            state.messages.append({"role": "system", "content": chain_msg})
            if state.verbose:
                print(f"⛓  Injected chain plan (turn {state.turn})")

    # ── Periodic plan state / question reminder ────────────────────────
    _REMINDER_INTERVAL = 8
    if state.plan is not None and state.plan.should_inject(state.turn):
        plan_msg = state.plan.render(state.turn, state.turn_length)
        # Append chain status if active
        if (state.chain_plan is not None and state.chain_plan.has_chain
                and not state.chain_plan.all_resolved()):
            plan_msg += "\n\n" + state.chain_plan.render()
        state.messages.append({"role": "system", "content": plan_msg})
        if state.verbose:
            print(f"\U0001f4cb  Injected research plan (turn {state.turn})")
    elif state.plan is None and state.turn > 1 and state.turn % _REMINDER_INTERVAL == 0:
        q_snippet = state.user_input[:300] + ("\u2026" if len(state.user_input) > 300 else "")
        state.messages.append({"role": "system", "content": (
            f"\U0001f50e REMINDER \u2014 You are answering this question:\n"
            f"{q_snippet}\n\n"
            "Stay focused on gathering information relevant to this question. "
            "If you already have enough, call final_answer."
        )})
        if state.verbose:
            print(f"\U0001f50e  Injected periodic question reminder (turn {state.turn})")

    # ── Research log injection (root only, periodic) ──────────────────
    # Inject a compact log of what research found so far, on the same
    # cadence as plan injection.  Survives compaction via state object.
    if (state.depth == 0
            and state.research_log
            and state.turn > 1
            and (state.turn % _REMINDER_INTERVAL == 0
                 or (state.plan is not None and state.plan.should_inject(state.turn)))):
        _log_text = _render_research_log(state)
        if _log_text:
            state.messages.append({"role": "system", "content": _log_text})
            if state.verbose:
                print(f"📋  Injected research log ({len(state.research_log)} entries)")

    # ── Draft injection (root only, every turn after first draft) ────
    # Inject the FULL current draft.  Avg draft is ~1.6K chars (p95 ~3K),
    # so this is affordable.  After history compaction evicts old turns,
    # this is the only copy in the context — must be complete.
    if state.depth == 0 and state.draft_versions:
        _draft_turn, _draft_text = state.draft_versions[-1]
        _ver = len(state.draft_versions)
        _prev_note = ""
        if _ver > 1:
            _prev_note = f"  ({_ver - 1} previous version{'s' if _ver > 2 else ''} available via read_draft)\n"
        state.messages.append({"role": "system", "content": (
            f"\U0001f4c4 YOUR CURRENT DRAFT (v{_ver}, saved turn {_draft_turn}, "
            f"{len(_draft_text):,} chars):\n"
            f"{_draft_text}\n\n"
            f"{_prev_note}"
            "Review your draft. Are there gaps? Delegate more research with "
            "conduct_research. When satisfied, call research_complete to publish."
        )})
        if state.verbose:
            print(f"\U0001f4c4  Injected full draft v{_ver} ({len(_draft_text):,} chars)")

    # ── History compaction (root only, after draft exists) ────────────
    # Once a draft absorbs research findings, old messages are redundant.
    # Replace them with a compact summary to reclaim context window.
    if (state.depth == 0
            and _cfg.HISTORY_COMPACTION_ENABLED
            and state.draft_versions
            and len(state.messages) > _cfg.HISTORY_COMPACTION_MSG_THRESHOLD
            and state.turn - state._last_truncation_turn >= _cfg.HISTORY_COMPACTION_MIN_INTERVAL):
        _compact_history(state)
        state._last_truncation_turn = state.turn

    # ── Tool gates (root only) ────────────────────────────────────────
    #
    # We always tell the model WHY a tool is hidden so it doesn't
    # hallucinate calls or try to game the unlock sequence.
    #
    _hidden_tools: set = set()
    _hidden_reasons: list = []

    if state.depth == 0:
        # Gate 1: research_complete + read_draft hidden until a draft exists
        if not state.draft_versions:
            _hidden_tools.add("research_complete")
            _hidden_tools.add("read_draft")
            _hidden_reasons.append(
                "research_complete and read_draft are not available yet — you "
                "must write a draft first with refine_draft(content='...')."
            )

        # Gate 2: research_complete hidden after rejection until draft revised
        if state.draft_versions and not state.draft_revised_since_rejection:
            _hidden_tools.add("research_complete")
            _hidden_reasons.append(
                "research_complete is locked — your last submission was "
                "rejected. You must revise your draft with refine_draft() "
                "before you can call research_complete again."
            )

        # Gate 3: refine_draft hidden until sufficient research
        _min_research = 2 if _cfg.DRAFT_FORMAT == "report" else 1
        if state.conduct_research_count < _min_research:
            _hidden_tools.add("refine_draft")
            _hidden_reasons.append(
                f"refine_draft is not available yet — you must do at least "
                f"{_min_research} round(s) of research first. Call conduct_research(task='...') "
                f"to gather information, then refine_draft to write your answer. "
                f"({state.conduct_research_count}/{_min_research} done)"
            )

        # Gate 5: research_complete hidden until post-draft research done (report mode)
        if (_cfg.DRAFT_FORMAT == "report"
                and state.draft_versions
                and state.research_after_first_draft == 0):
            _hidden_tools.add("research_complete")
            _hidden_reasons.append(
                "research_complete is locked — you must do at least one "
                "round of deepening research (conduct_research or "
                "summarize_webpage) AFTER your first draft before publishing. "
                "Review your draft for gaps and shallow sections, then research."
            )

        # Gate 4: think hidden at root when plan is active
        # The ResearchPlan auto-tracks progress mechanically — think()
        # at root is just a stalling tool.  Sub-agents keep it.
        if state.plan is not None:
            _hidden_tools.add("think")
            # No reason message needed — the model never had a habit of
            # calling think at root, and explaining its absence would just
            # draw attention to a tool that doesn't help here.

    if _hidden_tools:
        tools_for_turn = [
            t for t in (tools_for_turn or state.available_tools)
            if t["function"]["name"] not in _hidden_tools
        ]
        # Only inject gate message if there are reasons to explain
        # (silent hides like think don't need a message)
        if _hidden_reasons:
            _gate_msg = (
                "\U0001f512 TOOL AVAILABILITY:\n- "
                + "\n- ".join(_hidden_reasons)
                + "\n\nAll other tools are available normally."
            )
            state.messages.append({"role": "system", "content": _gate_msg})
        if state.verbose:
            print(f"\U0001f512  Hidden tools: {_hidden_tools}")

    # ── Budget-aware tool restriction (sub-agents only) ───────────────
    if state.depth > 0 and state.turn_length is not None:
        remaining = state.turn_length - state.turn

        # Short-budget sub-agents (verifiers, refusal challengers) are
        # purpose-built with tight budgets.  Injecting countdown messages
        # leaks orchestration internals and distracts from the task.
        # Only inject checkpoints / warnings for longer-running sub-agents.
        _SHORT_BUDGET_THRESHOLD = 5
        if state.turn_length > _SHORT_BUDGET_THRESHOLD:
            q_echo = state.user_input[:500] + ("\u2026" if len(state.user_input) > 500 else "")
            _ckpt_interval = max(state.turn_length // 3, 2)

            # Periodic checkpoint
            if state.turn > 0 and state.turn % _ckpt_interval == 0 and remaining >= _ckpt_interval:
                state.messages.append({"role": "system", "content": (
                    f"\U0001f4ca CHECKPOINT (turn {state.turn}/{state.turn_length}): "
                    "Pause and assess your progress.\n"
                    "  1. What key facts have you found so far?\n"
                    "  2. Do you have enough to answer the question?\n"
                    "  3. If YES \u2192 call final_answer now with what you have.\n"
                    "  4. If NO \u2192 what ONE specific thing is still missing?\n\n"
                    f"Question: {q_echo}\n\n"
                    "A good answer with 3 sources beats a perfect answer with 0. "
                    "Call think() to assess, then act."
                )})
                if state.verbose:
                    print(f"\U0001f4ca  Checkpoint injected (turn {state.turn}/{state.turn_length}, interval={_ckpt_interval})")

            # Escalating warnings — inject findings condensation to help synthesis
            if remaining == 3:
                # Build a quick findings recap so the model has a "cheat sheet"
                findings_recap = ""
                if state.findings:
                    recent = state.findings[-10:]
                    findings_recap = (
                        "\n\nYour research findings so far:\n"
                        + "\n".join(f"  {i+1}. {f[:300]}" for i, f in enumerate(recent))
                        + "\n"
                    )
                state.messages.append({"role": "system", "content": (
                    "\u26a0\ufe0f BUDGET WARNING: You have 4 turns remaining (including this one). "
                    "Start synthesizing NOW. Call final_answer with everything you have gathered.\n"
                    f"{findings_recap}\n"
                    f"Question: {q_echo}\n\n"
                    "Combine these findings into a clear answer. A partial answer with "
                    "real data is better than no answer."
                )})
            elif remaining == 2:
                state.messages.append({"role": "system", "content": (
                    "\U0001f6a8 FINAL WARNING: 3 turns left. Call final_answer THIS TURN "
                    "or NEXT TURN with your research. Do NOT start new searches."
                )})
            elif remaining == 1:
                # Last chance — inject findings directly so forced synthesis
                # has them even if the model fails to synthesize
                findings_dump = ""
                if state.findings:
                    recent = state.findings[-10:]
                    findings_dump = (
                        "\n\nHere are your findings — include them in final_answer:\n"
                        + "\n".join(f"  - {f[:400]}" for f in recent)
                        + "\n"
                    )
                state.messages.append({"role": "system", "content": (
                    "\U0001f6a8 LAST CHANCE: 2 turns remaining. Call final_answer NOW "
                    "with a comprehensive answer using everything below."
                    f"{findings_dump}"
                )})

        # Hard stop: restrict to terminal tool on final turn (all sub-agents)
        if remaining == 0:
            tools_for_turn = [state.terminal_tool]
            if state.verbose:
                print(f"\U0001f512 Restricting to {state.terminal_tool_name} only (last turn)")

    return tools_for_turn


# ═══════════════════════════════════════════════════════════════════════
# POST-TURN PROCESSING  (auto-reflection, degeneration check)
# ═══════════════════════════════════════════════════════════════════════

def _post_turn(state: AgentState, turn_record: TurnRecord, final_answer_result: Optional[str]) -> None:
    """Inject auto-reflection for search-heavy turns (sub-agents)."""
    _SEARCH_TOOLS = {
        "search_web", "fetch_url", "read_pdf",
        "extract_tables", "fetch_cached", "wikipedia_lookup",
    }
    if state.depth > 0 and not state.degenerated and final_answer_result is None:
        _turn_tool_names = [tc.tool_name for tc in turn_record.tool_calls]
        _did_search = any(t in _SEARCH_TOOLS for t in _turn_tool_names)
        _did_think = "think" in _turn_tool_names
        if _did_search and not _did_think:
            _search_summaries = []
            for tc in turn_record.tool_calls:
                if tc.tool_name in _SEARCH_TOOLS:
                    _out_preview = (tc.output or "")[:200].strip()
                    if _out_preview:
                        _search_summaries.append(f"  [{tc.tool_name}]: {_out_preview}")
            _summary_block = "\n".join(_search_summaries[:3])
            state.messages.append({
                "role": "system",
                "content": (
                    "\U0001f50d REFLECT before your next action. Call think(thought='...') to assess:\n"
                    "  - What key information did you just find?\n"
                    "  - What specific data is still missing?\n"
                    "  - Do you have enough to answer, or should you search more?\n"
                    "  - If searching more: what DIFFERENT query or source should you try?\n"
                    + (f"\nThis turn's results:\n{_summary_block}" if _summary_block else "")
                ),
            })
            if state.verbose:
                print(f"       \U0001f50d Auto-injected reflection prompt")


# ═══════════════════════════════════════════════════════════════════════
# NO-TOOL-CALL HANDLING
# ═══════════════════════════════════════════════════════════════════════

def _handle_no_tool_calls(
    state: AgentState,
    assistant_message: dict,
    turn_record: TurnRecord,
    turn_start: float,
) -> Optional[Dict[str, Any]]:
    """Handle a response with no tool calls (shouldn't happen with required).

    Returns a finalized dict if we should terminate, or None to continue.
    """
    state.consecutive_no_tool_count += 1
    final_content = assistant_message.get("content", "") or ""

    # ── Auto-draft capture (root only) ───────────────────────────────
    if state.depth == 0 and len(final_content.strip()) > 200 and state.draft_path:
        _STRUCTURE_SIGNALS = ("|", "#", "- ", "1.", "1)", "{", "**")
        _has_structure = any(s in final_content[:500] for s in _STRUCTURE_SIGNALS)
        if _has_structure or len(final_content.strip()) > 400:
            state.draft_versions.append((state.turn, final_content.strip()))
            ver = len(state.draft_versions)
            try:
                with open(state.draft_path, "w", encoding="utf-8") as f:
                    f.write(f"<!-- Auto-draft v{ver} | turn {state.turn} -->\n")
                    f.write(final_content.strip())
                state.memory.upsert(
                    key="draft_latest",
                    tool_name="auto_draft",
                    turn=state.turn,
                    content=final_content.strip(),
                    description="auto-captured from plain text",
                )
            except Exception:
                pass
            if state.verbose:
                print(f"\U0001f4dd  Auto-captured plain text as Draft v{ver} ({len(final_content):,} chars)")

    if state.consecutive_no_tool_count >= 3:
        # Model has degenerated
        if state.depth == 0 and state.draft_versions:
            _, latest_draft = state.draft_versions[-1]
            if state.verbose:
                print(f"\u26a0\ufe0f  Model degenerated at root \u2014 using draft v{len(state.draft_versions)}")
            turn_record.duration_s = round(time.time() - turn_start, 3)
            state.episode.turns.append(turn_record)
            return _finalize(state, latest_draft)
        elif len(final_content.strip()) > 200:
            if state.verbose:
                print(f"\u26a0\ufe0f  Model produced text without tool calls {state.consecutive_no_tool_count}x \u2014 accepting as final answer")
            turn_record.duration_s = round(time.time() - turn_start, 3)
            state.episode.turns.append(turn_record)
            return _finalize(state, final_content)
        else:
            if state.verbose:
                print(f"\u26a0\ufe0f  Model degenerated ({state.consecutive_no_tool_count}x no tool calls) \u2014 breaking to synthesis")
            state.degenerated = True
            turn_record.duration_s = round(time.time() - turn_start, 3)
            state.episode.turns.append(turn_record)
            return None  # signal: break out of turn loop
    else:
        # Nudge model to call a tool
        if state.verbose:
            print(f"\u26a0\ufe0f  No tool calls returned (attempt {state.consecutive_no_tool_count}/3) \u2014 nudging")
        content_echo = final_content[:3000] if final_content.strip() else ""
        if state.depth == 0:
            if content_echo:
                nudge = (
                    "Your response was plain text without a tool call. You MUST call a tool.\n\n"
                    "If this text is your answer, call refine_draft(content='...') to save it, "
                    "then research_complete() to publish.\n\n"
                    "If you need more research, call conduct_research(task='...').\n"
                    "If you need to plan, call think(thought='...').\n\n"
                    f"YOUR TEXT (for reference):\n{content_echo}"
                )
            else:
                nudge = (
                    "Your response was empty. Call a tool: "
                    "conduct_research(task='...') to research, "
                    "think(thought='...') to plan, "
                    "or refine_draft(content='...') to write your draft."
                )
        else:
            if content_echo:
                nudge = (
                    "Your response contained text but no tool call. You MUST call a tool.\n\n"
                    "If the text below IS your final answer, call final_answer now "
                    "and COPY YOUR COMPLETE ANSWER into the answer parameter.\n\n"
                    "If you still need to research, call a research tool.\n\n"
                    f"YOUR PREVIOUS RESPONSE:\n{content_echo}"
                )
            else:
                nudge = (
                    "Your response was empty. Call a research tool or final_answer."
                )
        state.messages.append({"role": "user", "content": nudge})
        turn_record.duration_s = round(time.time() - turn_start, 3)
        state.episode.turns.append(turn_record)
        return None  # continue


# ═══════════════════════════════════════════════════════════════════════
# POST-LOOP SYNTHESIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def _run_synthesis(state: AgentState) -> Dict[str, Any]:
    """Post-loop: draft fallback, forced synthesis, sandbox sub-agent, absolute fallback.

    Called when the turn loop exits without a final answer.
    """
    # ── Draft → final answer (root architecture) ─────────────────────
    if state.draft_versions:
        _draft_turn, latest_draft = state.draft_versions[-1]
        if len(latest_draft.strip()) >= 50:
            if state.verbose:
                print(f"\n\U0001f4dd Using draft v{len(state.draft_versions)} (turn {_draft_turn}, "
                      f"{len(latest_draft):,} chars) as final answer")
            return _finalize(state, latest_draft)

    if state.degenerated:
        if state.verbose:
            print(f"\n\u26a0\ufe0f  Model degenerated \u2014 skipping Stage 1, going straight to fresh synthesis sub-agent")
    else:
        if state.verbose:
            print(f"\n\u26a0\ufe0f  Turn limit reached \u2014 forcing final_answer synthesis")

    # Build findings summary
    findings_block = ""
    if state.findings:
        recent = state.findings[-15:]
        findings_text = "\n".join(f"{i+1}. {f}" for i, f in enumerate(recent))
        if len(findings_text) > 3000:
            findings_text = findings_text[:3000] + "\n... (truncated)"
        findings_block = (
            "\n\nHere are your key research findings so far:\n"
            f"{findings_text}\n"
        )

    # Log memory store stats
    if state.memory and state.verbose:
        stats = state.memory.compression_stats()
        print(f"\U0001f4e6 Memory store: {stats['entries']} entries, "
              f"{stats['raw_chars']:,} raw chars \u2192 "
              f"{stats['base64_chars']:,} compressed chars "
              f"({stats['savings_pct']}% savings)")

    q_echo_synth = state.user_input[:500] + ("\u2026" if len(state.user_input) > 500 else "")

    # ── Gap check before synthesis (plan_state only) ──────────────────
    gap_note = ""
    if state.plan is not None:
        gap_msg = state.plan.render_gap_check()
        if gap_msg:
            gap_note = f"\n\n{gap_msg}\n"
            if state.verbose:
                print(f"\U0001f4ca  Gap check injected before synthesis")

    # ── Stage 1: Forced final_answer in current context ──────────────
    _FINAL_ANSWER_SCHEMA = next(
        (t for t in TOOLS if t["function"]["name"] == "final_answer"),
        state.terminal_tool,
    )

    if not state.degenerated:
        state.messages.append({
            "role": "user",
            "content": (
                "You have run out of turns. Call final_answer NOW with your best "
                "response based on everything gathered so far.\n\n"
                f"ORIGINAL QUESTION:\n{q_echo_synth}\n"
                f"{findings_block}\n"
                f"{gap_note}"
                "Synthesize these findings into a clear, well-cited answer to the "
                "question above. Address any noted gaps honestly \u2014 state what is "
                "well-supported and what remains uncertain."
            ),
        })

        approx_input_tokens = sum(len(str(m.get("content", ""))) for m in state.messages) // 4
        synth_max_tokens = max(state.context_window - approx_input_tokens - _cfg.TOKEN_SAFETY_MARGIN, 256)
        synth_max_tokens = min(synth_max_tokens, state.max_tokens)
    else:
        synth_max_tokens = state.max_tokens  # won't be used; loop range is 0

    # Try synthesis up to 2 times (skipped when degenerated)
    for synth_attempt in range(0 if state.degenerated else 2):
        try:
            synth_response = call_api(state, synth_max_tokens, tools_override=[_FINAL_ANSWER_SCHEMA])
            if synth_response.status_code == 200:
                synth_result = synth_response.json()
                synth_choices = synth_result.get("choices")
                if synth_choices and isinstance(synth_choices, list) and len(synth_choices) > 0:
                    synth_msg = synth_choices[0].get("message", {})
                    state.messages.append(synth_msg)
                    # Strict extraction: only accept an actual final_answer
                    # tool call — do NOT fall back to assistant content, which
                    # is often garbage monologue from vLLM when it ignores
                    # tools_override.
                    final_content = ""
                    for tc in (synth_msg.get("tool_calls") or []):
                        func = tc.get("function", {})
                        name = func.get("name", "").split("<|")[0]
                        if name == "final_answer":
                            try:
                                args = json.loads(func.get("arguments", "{}") or "{}")
                            except json.JSONDecodeError:
                                args = {}
                            final_content = args.get("answer", "")
                            break
                    if not final_content:
                        # Fall back to assistant content if present.
                        # Findings-based fallback downstream will catch
                        # cases where this is low-quality.
                        final_content = (synth_msg.get("content", "") or "").strip()
                    if final_content.strip():
                        synth_usage = synth_result.get("usage", {})
                        synth_record = TurnRecord(
                            turn_number=state.turn + 1 + synth_attempt,
                            assistant_content=final_content,
                            raw_assistant_message=synth_msg,
                            prompt_tokens=synth_usage.get("prompt_tokens", 0),
                            completion_tokens=synth_usage.get("completion_tokens", 0),
                            total_tokens=synth_usage.get("total_tokens", 0),
                        )
                        synth_record.duration_s = 0
                        state.episode.turns.append(synth_record)
                        if state.verbose:
                            print(f"\u2705 Synthesis turn produced response (attempt {synth_attempt + 1})")
                            print(f"\n\U0001f4dd Final Response:\n{final_content}")
                        return _finalize(state, final_content)
                    else:
                        if state.verbose:
                            print(f"\u26a0\ufe0f  Synthesis attempt {synth_attempt + 1} returned empty \u2014 retrying")
                else:
                    if state.verbose:
                        print(f"\u274c Synthesis attempt {synth_attempt + 1} returned no choices: "
                              f"{str(synth_result)[:200]}")
            else:
                error_msg = str(synth_response.json().get("error", {}).get("message", ""))
                if "max_tokens" in error_msg or "input tokens" in error_msg:
                    match = re.search(r'has (\d+) input tokens', error_msg)
                    if match:
                        input_tokens = int(match.group(1))
                        synth_max_tokens = max(
                            state.context_window - input_tokens - _cfg.TOKEN_SAFETY_MARGIN, 256
                        )
                        if state.verbose:
                            print(f"\u26a0\ufe0f  Synthesis context overflow, retrying with {synth_max_tokens} max_tokens")
                        continue
                if state.verbose:
                    print(f"\u274c Synthesis turn API error: {synth_response.status_code}")
        except Exception as e:
            if state.verbose:
                print(f"\u274c Synthesis attempt {synth_attempt + 1} failed: {e}")

    # ── Synthesis sub-agent via sandbox file ──────────────────────────
    if state.memory and state.sandbox_files is None:
        memory_json = state.memory.to_json()
        memory_b64 = base64.b64encode(memory_json.encode("utf-8")).decode("ascii")

        if state.verbose:
            stats = state.memory.compression_stats()
            print(f"\U0001f4e6 Uploading {stats['entries']} memory entries to sandbox "
                  f"({len(memory_json):,} bytes as research_data.json)")

        # Include a findings recap in the task so the synthesizer
        # has immediate context before even opening the file
        findings_hint = ""
        if state.findings:
            recent = state.findings[-10:]
            findings_hint = (
                "\n\nKey findings so far (details in research_data.json):\n"
                + "\n".join(f"  - {f[:300]}" for f in recent)
                + "\n"
            )

        synthesis_task = (
            f"Answer this question using the research data in research_data.json:\n\n"
            f"{state.user_input}"
            f"{findings_hint}"
        )
        sandbox_files = {"research_data.json": memory_b64}

        try:
            # Synthesis needs at most 3-5 turns: read data + answer
            synth_turn_budget = min(_cfg.SUB_AGENT_TURN_BUDGET, 5)

            if state.verbose:
                print(f"\U0001f504 Spawning synthesis sub-agent ({synth_turn_budget} turn budget, "
                      f"research_data.json pre-loaded in sandbox)")

            # Import dispatch lazily to avoid circular imports
            from .agent import dispatch
            synth_result = dispatch(
                user_input=synthesis_task,
                turn_length=synth_turn_budget,
                verbose=state.verbose,
                max_tokens=state.max_tokens,
                temperature=state.temperature,
                model=state.model,
                reasoning_effort=state.reasoning_effort,
                example_id=state.example_id,
                _depth=state.depth + 1,
                _sandbox_files=sandbox_files,
                _is_synthesizer=True,
            )

            final_content = synth_result.get("final_response", "")

            # Record the synthesis sub-agent as a turn in the parent trace
            if synth_result.get("trace"):
                synth_child_trace = synth_result["trace"]
                synth_record = TurnRecord(
                    turn_number=state.turn + 3,
                    assistant_content=final_content,
                    raw_assistant_message={"synthesis_sub_agent": True},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )
                tc_record = ToolCallRecord(
                    tool_name="spawn_agent",
                    tool_args={"task": "synthesis via sandbox file"},
                    tool_call_id="synthesis_fallback",
                    output=final_content[:500] if final_content else "",
                    duration_s=synth_result["trace"].duration_s or 0,
                    child_trace=synth_child_trace,
                )
                synth_record.tool_calls.append(tc_record)
                synth_record.duration_s = synth_result["trace"].duration_s or 0
                state.episode.turns.append(synth_record)

            if final_content and final_content.strip() and not final_content.startswith("[Auto-extracted"):
                if state.verbose:
                    print(f"\u2705 Synthesis sub-agent succeeded")
                    print(f"\n\U0001f4dd Final Response:\n{final_content}")
                return _finalize(state, final_content)
            else:
                if state.verbose:
                    print(f"\u26a0\ufe0f  Synthesis sub-agent returned empty/fallback response \u2014 "
                          f"falling through to absolute fallback")
        except Exception as e:
            if state.verbose:
                print(f"\u274c Synthesis sub-agent failed: {e}")

    # ── Absolute fallback ─────────────────────────────────────────────
    final_content = _build_fallback_response(state.messages, findings=state.findings)

    if state.verbose:
        if final_content:
            print(f"\n\U0001f4dd Response (raw fallback from tool results):\n{final_content[:500]}")
        else:
            print(f"\n\u274c No content could be extracted from conversation history")

    return _finalize(state, final_content)


# ═══════════════════════════════════════════════════════════════════════
# MAIN TURN LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_agent_loop(state: AgentState) -> Dict[str, Any]:
    """Execute the agent turn loop and return the final result dict.

    This is the direct replacement for the old dispatch() while-loop plus
    its post-loop synthesis pipeline. The caller creates an AgentState via
    create_state(), then passes it here.
    """

    while True:
        # ── Turn limit ────────────────────────────────────────────────
        if state.turn_length is not None and state.turn >= state.turn_length:
            if state.verbose:
                print(f"\n\u23f9\ufe0f  Reached maximum turns ({state.turn_length})")
            break

        state.turn += 1
        turn_start = time.time()
        if state.verbose:
            print(f"\n{'─' * 70}")
            print(f"TURN {state.turn}" + (f" / {state.turn_length}" if state.turn_length else " (unlimited)"))
            print(f"{'─' * 70}")

        # ── Token budget estimation ───────────────────────────────────
        approx_input_tokens = sum(len(str(m.get("content", ""))) for m in state.messages) // 4
        effective_max_tokens = max(state.context_window - approx_input_tokens - _cfg.TOKEN_SAFETY_MARGIN, 256)
        effective_max_tokens = min(effective_max_tokens, state.max_tokens)
        max_completion = state.profile.get("max_completion_tokens")
        if max_completion:
            effective_max_tokens = min(effective_max_tokens, max_completion)

        # ── Pre-turn injections ───────────────────────────────────────
        tools_for_turn = _inject_pre_turn(state)

        # ── API call ──────────────────────────────────────────────────
        response = call_api(state, effective_max_tokens, tools_override=tools_for_turn)
        result = response.json()

        # ── Error handling ────────────────────────────────────────────
        if response.status_code != 200:
            error_msg = str(result.get("error", {}).get("message", ""))

            # Context overflow — retry with reduced max_tokens
            if "max_tokens" in error_msg or "max_completion_tokens" in error_msg:
                match = re.search(r'has (\d+) input tokens', error_msg)
                if match:
                    input_tokens = int(match.group(1))
                    effective_max_tokens = state.context_window - input_tokens - _cfg.TOKEN_SAFETY_MARGIN
                    if effective_max_tokens >= 1:
                        if state.verbose:
                            print(f"\u26a0\ufe0f  max_tokens too large, retrying with {effective_max_tokens}")
                        response = call_api(state, effective_max_tokens, tools_override=tools_for_turn)
                        result = response.json()
                        if response.status_code != 200:
                            if state.verbose:
                                print(f"\u274c API Error: {result}")
                            return _finalize(state, f"Error: {result}")
                    else:
                        if state.verbose:
                            print(f"\u274c Context window exhausted ({input_tokens} input tokens, "
                                  f"{state.context_window} max)")
                        return _finalize(state, f"Error: Context window exhausted. "
                                         f"Input too long ({input_tokens} tokens).")
                else:
                    if state.verbose:
                        print(f"\u274c API Error: {result}")
                    return _finalize(state, f"Error: {result}")

            # vLLM tool-call parser error (recoverable)
            elif ("unexpected tokens" in error_msg
                  or "tool_call" in error_msg.lower()
                  or result.get("error", {}).get("code") == 400):
                if state.verbose:
                    print(f"\u26a0\ufe0f  vLLM parser error (recoverable): {error_msg[:150]}")
                state.messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed. "
                        "Call your tool again with correct formatting. "
                        "Use exact tool names from your available tools."
                    )
                })
                turn_record = TurnRecord(
                    turn_number=state.turn,
                    assistant_content=None,
                    raw_assistant_message={"error": error_msg},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )
                turn_record.duration_s = round(time.time() - turn_start, 3)
                state.episode.turns.append(turn_record)
                continue

            else:
                if state.verbose:
                    print(f"\u274c API Error: {result}")
                return _finalize(state, f"Error: {result}")

        # ── Malformed 200 response ────────────────────────────────────
        if result.get("error"):
            err_detail = result["error"]
            if state.verbose:
                print(f"\u26a0\ufe0f  vLLM returned 200 with error body: {str(err_detail)[:200]}")
            state.messages.append({
                "role": "user",
                "content": (
                    "The server encountered an internal error processing your last response. "
                    "Please try your tool call again. Use a simpler approach if possible."
                )
            })
            turn_record = TurnRecord(
                turn_number=state.turn,
                assistant_content=None,
                raw_assistant_message={"error": str(err_detail)},
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
            )
            turn_record.duration_s = round(time.time() - turn_start, 3)
            state.episode.turns.append(turn_record)
            continue

        choices = result.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            if state.verbose:
                print(f"\u274c Malformed API response (no choices): {str(result)[:300]}")
            return _finalize(state, f"Error: Malformed API response \u2014 no 'choices' returned. "
                             f"Raw: {str(result)[:200]}")

        assistant_message = choices[0].get("message")
        if not assistant_message:
            if state.verbose:
                print(f"\u274c Malformed API response (no message in choice): {str(choices[0])[:300]}")
            return _finalize(state, f"Error: Malformed API response \u2014 no 'message' in choice. "
                             f"Raw: {str(choices[0])[:200]}")

        usage = result.get("usage", {})

        # ── Sanitize assistant message ────────────────────────────────
        clean_msg = {
            "role": assistant_message["role"],
            "content": assistant_message.get("content") or "",
        }
        if assistant_message.get("tool_calls"):
            clean_msg["tool_calls"] = assistant_message["tool_calls"]
        state.messages.append(clean_msg)

        # ── Build turn record ─────────────────────────────────────────
        turn_record = TurnRecord(
            turn_number=state.turn,
            assistant_content=assistant_message.get("content"),
            raw_assistant_message=assistant_message,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

        # ── No tool calls → nudge / degeneration ─────────────────────
        tool_calls_in_msg = assistant_message.get("tool_calls") or []
        if not tool_calls_in_msg:
            result_dict = _handle_no_tool_calls(state, assistant_message, turn_record, turn_start)
            if result_dict is not None:
                return result_dict
            if state.degenerated:
                break
            continue
        else:
            state.consecutive_no_tool_count = 0

        state.total_tool_calls += len(tool_calls_in_msg)
        if state.verbose:
            print(f"\U0001f527 Tool calls: {len(tool_calls_in_msg)}")

        # ── Process each tool call ────────────────────────────────────
        final_answer_result: Optional[str] = None
        for i, tool_call in enumerate(tool_calls_in_msg, 1):
            tool_name = tool_call["function"]["name"]
            if "<|" in tool_name:
                tool_name = tool_name.split("<|")[0]

            # ── Sanitize hallucinated tool names ──────────────────────
            raw_tool_name = tool_name
            tool_name = _sanitize_tool_name(tool_name, state)
            if tool_name != raw_tool_name and state.verbose:
                print(f"   🔧 Sanitized tool name: '{raw_tool_name}' → '{tool_name}'")

            raw_args = tool_call["function"].get("arguments", "")
            args_were_malformed = False
            try:
                tool_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args_were_malformed = True
                # ── General sanitizer: strip common model noise ───────
                _sanitized = _sanitize_raw_args(raw_args)
                if _sanitized is not None:
                    tool_args = _sanitized
                    if state.verbose:
                        print(f"   \U0001f527 Sanitized malformed args for {tool_name} "
                              f"({len(raw_args)} raw chars → {len(tool_args)} keys)")
                elif tool_name in ("final_answer", "refine_draft") and raw_args:
                    _recovered = _recover_final_answer_from_raw(raw_args)
                    if _recovered:
                        _param = "answer" if tool_name == "final_answer" else "content"
                        tool_args = {_param: _recovered}
                        if state.verbose:
                            print(f"   \U0001f527 Recovered {tool_name} from malformed JSON ({len(_recovered)} chars)")
                    else:
                        tool_args = {}
                        if state.verbose:
                            print(f"   \u26a0\ufe0f  Malformed tool arguments for {tool_name}, could not recover")
                else:
                    tool_args = {}
                    if state.verbose:
                        print(f"   \u26a0\ufe0f  Malformed JSON for {tool_name}: {raw_args[:100]}")

            if state.verbose:
                args_preview = str(tool_args)[:100]
                print(f"   [{i}/{len(tool_calls_in_msg)}] {tool_name}({args_preview})")

            # ── Root tool scope enforcement ───────────────────────────
            # Block final_answer at root — must use research_complete.
            if state.depth == 0 and tool_name == "final_answer":
                error_msg = (
                    "⛔ final_answer is NOT available at root level.\n\n"
                    "You are the orchestrator. To publish your answer:\n"
                    "  1. refine_draft(content='<your complete answer>')\n"
                    "  2. research_complete()\n\n"
                    "research_complete() publishes your latest draft after "
                    "quality review. Do NOT call final_answer again."
                )
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": error_msg,
                })
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=error_msg,
                    duration_s=0.0,
                )
                turn_record.tool_calls.append(tc_record)
                if state.verbose:
                    print(f"       🚫 Blocked final_answer at root (must use research_complete)")
                continue

            # The orchestrator (depth==0) may only use tools in
            # TOOL_HANDLERS. Block any worker-only tool and nudge the
            # model toward conduct_research instead.
            if state.depth == 0 and tool_name not in TOOL_HANDLERS:
                error_msg = (
                    f"⚠️ Tool '{tool_name}' is not available to the orchestrator. "
                    "You do NOT have direct access to search, fetch, or code "
                    "execution tools. Delegate research tasks to sub-agents:\n"
                    "  conduct_research(task='<self-contained description of "
                    "what to find>')\n"
                    "The sub-agent can use search_web, fetch_url, execute_code, "
                    "and other research tools on your behalf."
                )
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": error_msg,
                })
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=error_msg,
                    duration_s=0.0,
                )
                turn_record.tool_calls.append(tc_record)
                if state.verbose:
                    print(f"       🚫 Blocked root tool leakage: {tool_name}")
                continue

            # Sub-agents (depth>0) must not call orchestrator-only or
            # recursion-inducing tools.  The tool list already excludes
            # them (agent_state.py), but models sometimes hallucinate
            # tool calls anyway.  Hard-block here as defence-in-depth.
            _SUB_AGENT_BLOCKED = {"spawn_agent", "conduct_research", "recall_memory", "update_draft"}
            if state.depth > 0 and tool_name in _SUB_AGENT_BLOCKED:
                error_msg = (
                    f"⛔ Tool '{tool_name}' is not available to sub-agents. "
                    "You have search_web, fetch_url, execute_code, extract_tables, "
                    "think, and final_answer. Use those tools directly to complete "
                    "your task, then call final_answer with your results."
                )
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": error_msg,
                })
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=error_msg,
                    duration_s=0.0,
                )
                turn_record.tool_calls.append(tc_record)
                if state.verbose:
                    print(f"       🚫 Blocked sub-agent recursion: {tool_name} at depth {state.depth}")
                continue

            # ── Dispatch to handler ───────────────────────────────────
            handler = TOOL_HANDLERS.get(tool_name, handle_generic_tool)
            node_result = handler(
                state, tool_call, tool_args, turn_record,
                args_were_malformed=args_were_malformed,
            )

            fa_content, should_break = node_result
            if fa_content is not None:
                final_answer_result = fa_content
                break
            if should_break:
                state.degenerated = True
                break

        # ── Finalize turn ─────────────────────────────────────────────
        turn_record.duration_s = round(time.time() - turn_start, 3)
        # Snapshot chain plan state so the HTML trace shows per-turn progress
        if (state.depth == 0 and state.chain_plan is not None
                and state.chain_plan.has_chain):
            turn_record.chain_snapshot = [
                {
                    "step": s.step,
                    "lookup": s.lookup,
                    "placeholder": s.placeholder,
                    "resolved_value": s.resolved_value,
                }
                for s in state.chain_plan.chain_steps
            ]
        state.episode.turns.append(turn_record)

        # ── Post-turn processing ──────────────────────────────────────
        _post_turn(state, turn_record, final_answer_result)

        # Degeneration triggered inside tool handler
        if state.degenerated:
            break

        # Final answer received
        if final_answer_result is not None:
            if state.verbose:
                print(f"\n\U0001f4dd Final Response:\n{final_answer_result}")
            return _finalize(state, final_answer_result)

    # ── Post-loop synthesis ───────────────────────────────────────────
    return _run_synthesis(state)
