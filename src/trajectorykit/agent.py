"""
Main agent loop for the trajectorykit system.
"""

import base64
import requests
import json
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any

import os
from .config import MODEL_NAME, VLLM_API_URL, SYSTEM_PROMPT, WORKER_PROMPT, SYNTHESIZER_PROMPT, TOKEN_SAFETY_MARGIN, TRACES_DIR, get_model_profile
from .config import SYMBOLIC_REFERENCES, SYMBOLIC_THRESHOLD, PLAN_STATE, PLAN_INJECT_INTERVAL
from .config import VERIFY_BEFORE_PUBLISH, VERIFIER_PROMPT
from .config import SPOT_CHECK_ENABLED, SPOT_CHECK_CLAIMS, SPOTCHECK_EXTRACT_PROMPT, SPOTCHECK_COMPARE_PROMPT, SPOTCHECK_REFUSAL_PROMPT
from .config import MAX_VERIFICATION_REJECTIONS, MAX_SPOT_CHECK_REJECTIONS
from .memory import MemoryStore
from .plan import ResearchPlan
from .symbolic import make_symbolic
from .tool_store import TOOLS, ROOT_TOOLS, dispatch_tool_call, fetch_url
from .tracing import EpisodeTrace, TurnRecord, ToolCallRecord

import logging
logger = logging.getLogger(__name__)


def _recover_final_answer_from_raw(raw_args: str) -> str | None:
    """Try to extract the answer text from a malformed JSON arguments string.
    
    The model often produces *almost* valid JSON for final_answer — e.g. a
    trailing quote is missing, or there's an unescaped newline.  We try
    several regex strategies to pull out the answer value.
    
    Returns the recovered answer string, or None if nothing useful found.
    """
    # Strategy 1: grab everything after "answer": "  up to the last quote-like boundary
    m = re.search(r'"answer"\s*:\s*"(.*)', raw_args, re.DOTALL)
    if m:
        candidate = m.group(1)
        # Strip trailing incomplete JSON: "}  or just "
        candidate = re.sub(r'"\s*\}?\s*$', '', candidate)
        # Unescape common JSON escapes
        for esc, repl in [('\\n', '\n'), ('\\t', '\t'), ('\\"', '"'), ('\\\\', '\\')]:
            candidate = candidate.replace(esc, repl)
        if len(candidate.strip()) >= 20:
            return candidate.strip()
    
    # Strategy 2: the whole raw_args might just be the answer text (no JSON wrapper)
    stripped = raw_args.strip().strip('"').strip()
    if len(stripped) >= 50 and '{' not in stripped[:5]:
        return stripped
    
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
    """Extract recent tool results from message history for context.
    
    Returns a string summarizing the last N tool results, useful for
    building a fallback response when the model fails to call final_answer.
    """
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


def _build_fallback_response(messages: list) -> str:
    """Build a response from message history when synthesis fails.
    
    With tool_choice='required', assistant messages have content=null,
    so we must look at tool results and any final_answer calls.
    """
    # First: check if any tool call was final_answer with content
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            answer = _extract_final_answer(msg)
            if answer.strip():
                return answer
    
    # Second: check assistant content (for non-tool-choice cases)
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            if content.strip():
                return content
    
    # Third: concatenate the last few tool results as raw findings
    tool_results = _extract_tool_results(messages, max_chars=3000)
    if tool_results.strip():
        return f"[Auto-extracted from research — synthesis failed]\n\n{tool_results}"
    
    return ""


def _focused_page_summary(
    url: str,
    focus: str,
    model: str,
    vllm_url: str,
    max_fetch_chars: int = 200_000,
    max_summary_tokens: int = 4096,
) -> str:
    """Fetch a URL and produce a focused, LLM-generated summary.

    Uses ``fetch_url`` (full safeguard chain: blocklist, Jina-first,
    rate-limiting, CF detection, multi-tier fallback) then a single LLM
    call with an optional *focus* parameter to extract only the relevant
    information from the page.

    Shared by the ``summarize_webpage`` tool intercept and the spot-check
    evidence-gathering step.

    Returns the summary text on success, or an error/empty string on failure.
    """
    result = fetch_url(url, max_chars=max_fetch_chars)
    if not result.get("ok"):
        return (
            f"Failed to fetch {url}: "
            f"{result.get('reason', 'unknown error')}."
        )
    page_content = result["content"]
    if len(page_content.strip()) < 50:
        return (
            f"Page fetched but content is too short ({len(page_content)} chars). "
            f"The page may be behind a paywall or require JavaScript."
        )

    focus_instruction = (
        f"\n\nFOCUS: Pay special attention to: {focus}"
        if focus else ""
    )
    messages = [
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
                "messages": messages,
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


def dispatch(
    user_input: str,
    turn_length: Optional[int] = 5,
    verbose: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    example_id: Optional[str] = None,
    _depth: int = 0,
    _sandbox_files: Optional[Dict[str, str]] = None,
    _is_synthesizer: bool = False,
) -> Dict[str, Any]:
    """
    Run an agentic loop with iterative tool calling and response refinement.
    
    Args:
        user_input: Initial user message describing the task
        turn_length: Maximum number of turns. None for unlimited (runs until completion)
        verbose: Print detailed turn-by-turn output
        max_tokens: Maximum tokens per generation (default: from model profile context_window)
        temperature: Sampling temperature (default: from model profile)
        model: Model name to use (default: config.MODEL_NAME)
        reasoning_effort: Reasoning effort for supported models — "low"/"medium"/"high"
                          (default: from model profile if supported, else None)
        _depth: Current recursion depth (internal — used by spawn_agent)
        _sandbox_files: Files to auto-inject into every execute_code sandbox call.
                        Dict of {filename: base64_content}. Used by synthesis sub-agent
                        to access research data without polluting the LLM context.
        _is_synthesizer: If True, use the synthesizer prompt and restricted tool set.
                         Decoupled from _sandbox_files so regular workers can also
                         receive sandbox files (e.g. via memory_keys on spawn_agent).
    
    Returns:
        Dictionary with:
            - 'final_response': The model's final answer
            - 'turns': Number of turns completed
            - 'tool_calls': Total number of tool calls made
            - 'messages': Full conversation history
            - 'trace': EpisodeTrace capturing the full execution tree
    """

    # Resolve model and its profile
    model = model or MODEL_NAME
    profile = get_model_profile(model)
    context_window = profile["context_window"]

    # Resolve temperature: explicit arg > profile default
    if temperature is None:
        temperature = profile.get("default_temperature", 0.7)

    # Resolve max_tokens: explicit arg > profile context_window
    if max_tokens is None:
        max_tokens = context_window

    # Resolve reasoning_effort: explicit arg > profile default (if model supports it)
    if reasoning_effort is None and profile.get("supports_reasoning_effort"):
        reasoning_effort = profile.get("default_reasoning_effort")

    # Initialize trace
    episode = EpisodeTrace(
        depth=_depth,
        example_id=example_id or "",
        user_input=user_input,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        turn_length=turn_length,
        started_at=datetime.now().isoformat(),
    )
    episode_start = time.time()
    
    # ── Depth-aware prompt & tool selection ──────────────────────────────
    # Root agent (depth 0): orchestrator prompt, ROOT_TOOLS only
    #   (conduct_research, refine_draft, research_complete, think,
    #    summarize_webpage, compress_findings, search_available_tools)
    # Sub-agents (depth >= 1): worker prompt, full TOOLS minus orchestrator-only tools
    # Synthesis sub-agent (_is_synthesizer): synthesizer prompt, code + final_answer only
    if _is_synthesizer:
        system_prompt = SYNTHESIZER_PROMPT
        available_tools = [t for t in TOOLS if t["function"]["name"] in ("execute_code", "final_answer", "search_available_tools")]
    elif _depth == 0:
        system_prompt = SYSTEM_PROMPT
        available_tools = ROOT_TOOLS
    else:
        system_prompt = WORKER_PROMPT
        available_tools = [t for t in TOOLS if t["function"]["name"] not in ("spawn_agent", "recall_memory", "update_draft")]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # Schema for the terminal tool — used when forcing termination.
    # Root uses research_complete; sub-agents/synthesizers use final_answer.
    _terminal_tool_name = "research_complete" if _depth == 0 else "final_answer"
    TERMINAL_TOOL = next(
        (t for t in available_tools if t["function"]["name"] == _terminal_tool_name),
        # Fallback: search the full TOOLS list (ROOT_TOOLS has no final_answer)
        next((t for t in TOOLS if t["function"]["name"] == "final_answer"), available_tools[0])
    )
    
    # ── Draft file path (root only) ──────────────────────────────────
    _draft_path: str | None = None
    _verification_rejections: int = 0   # gatekeeper: how many times verifier rejected
    _MAX_VERIFICATION_REJECTIONS = MAX_VERIFICATION_REJECTIONS  # configurable via YAML
    _spot_check_rejections: int = 0     # Stage 2: how many times spot-check rejected
    _MAX_SPOT_CHECK_REJECTIONS = MAX_SPOT_CHECK_REJECTIONS      # configurable via YAML
    if _depth == 0:
        os.makedirs(TRACES_DIR, exist_ok=True)
        _draft_path = os.path.join(TRACES_DIR, f"draft_{episode.trace_id}.md")
        # Create empty draft file
        with open(_draft_path, "w", encoding="utf-8") as f:
            f.write("")

    def _call_api(effective_max_tokens, tools_override=None):
        """Build payload and call the chat completions API.
        
        Args:
            effective_max_tokens: Max tokens for this call.
            tools_override: If provided, use this list of tool schemas instead
                            of the full TOOLS list. Used to restrict the model
                            to only final_answer on the last turn.
        """
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools_override if tools_override is not None else available_tools,
            "tool_choice": "required",
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if reasoning_effort and profile.get("supports_reasoning_effort"):
            payload["reasoning_effort"] = reasoning_effort
        # Pass chat_template_kwargs (e.g. enable_thinking for Qwen3)
        chat_template_kwargs = profile.get("chat_template_kwargs")
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        return requests.post(f"{VLLM_API_URL}/chat/completions", json=payload)
    
    turn = 0
    total_tool_calls = 0
    consecutive_error_count = 0
    consecutive_no_tool_count = 0  # Tracks successive responses with no tool calls
    last_error_signature = None
    MAX_CONSECUTIVE_ERRORS = 3  # Break retry loops after 3 identical failures
    
    # ── Findings tracker ──────────────────────────────────────────────
    # Accumulate key findings from tool results so we can inject them
    # into the synthesis prompt if the model runs out of turns.
    findings: list[str] = []
    
    # ── Consecutive search tracker ────────────────────────────────────
    # Detect when the agent falls into a search loop instead of
    # pausing to reflect.  After _MAX hits, suppress tool output
    # and force a think() call to break the loop.
    _consecutive_search_count = 0
    _MAX_CONSECUTIVE_SEARCHES = 6  # hard-block after this many
    _CONSECUTIVE_SEARCH_WARNING = 5  # warn at this count
    
    # ── Memory store ──────────────────────────────────────────────────
    # Full-fidelity storage of tool outputs. Compressed at synthesis time
    # and passed to the synthesis sub-agent for programmatic querying.
    memory = MemoryStore()
    
    # ── Research plan (root orchestrator only) ─────────────────────
    # Mechanical progress tracker injected into conversation every N turns.
    plan: ResearchPlan | None = None
    if PLAN_STATE and _depth == 0:
        q_plan = user_input[:300] + ("…" if len(user_input) > 300 else "")
        plan = ResearchPlan(question=q_plan)
        plan.INJECT_EVERY_N_TURNS = PLAN_INJECT_INTERVAL
    
    # ── Degeneration flag ─────────────────────────────────────────────
    # Set when the model can't produce tool calls (3x consecutive failures).
    # When True, the post-loop synthesis skips Stage 1 (same corrupted
    # context) and jumps straight to a fresh synthesis sub-agent.
    _degenerated = False
    
    # ── Draft report (root orchestrator only) ─────────────────────────
    # Stores progressive draft snapshots via the refine_draft virtual tool.
    # Latest draft serves as fallback when final_answer is empty or the
    # model runs out of turns.
    draft_versions: list[tuple[int, str]] = []  # [(turn, content), ...]
    
    def _finalize(final_content: str) -> Dict[str, Any]:
        """Build the return dict and finalize the trace."""
        episode.final_response = final_content
        episode.total_turns = turn
        episode.total_tool_calls = total_tool_calls
        episode.ended_at = datetime.now().isoformat()
        episode.duration_s = round(time.time() - episode_start, 3)
        episode.compute_recursive_stats()
        return {
            'final_response': final_content,
            'turns': turn,
            'tool_calls': total_tool_calls,
            'messages': messages,
            'trace': episode,
        }
    
    while True:
        # Check turn limit
        if turn_length is not None and turn >= turn_length:
            if verbose:
                print(f"\n⏹️  Reached maximum turns ({turn_length})")
            break
        
        turn += 1
        turn_start = time.time()
        if verbose:
            print(f"\n{'─'*70}")
            print(f"TURN {turn}" + (f" / {turn_length}" if turn_length else " (unlimited)"))
            print(f"{'─'*70}")
        
        # Dynamically estimate remaining budget from conversation size
        # tiktoken isn't available, so rough-estimate: 4 chars ≈ 1 token
        approx_input_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        effective_max_tokens = max(context_window - approx_input_tokens - TOKEN_SAFETY_MARGIN, 256)
        # Never exceed the original max_tokens cap
        effective_max_tokens = min(effective_max_tokens, max_tokens)
        # Cap per-turn completion tokens to prevent truncated tool-call JSON
        # (e.g. Qwen3 with 32K context but thinking tokens can eat most of it)
        max_completion = profile.get("max_completion_tokens")
        if max_completion:
            effective_max_tokens = min(effective_max_tokens, max_completion)

        # ── Periodic plan state / question reminder ────────────────────────
        # Root: inject plan + draft every N turns.
        # Sub-agents: simple question reminder every 8 turns.
        _REMINDER_INTERVAL = 8
        if plan is not None and plan.should_inject(turn):
            plan_msg = plan.render(turn, turn_length)
            messages.append({"role": "system", "content": plan_msg})
            if verbose:
                print(f"📋  Injected research plan (turn {turn})")
        elif plan is None and turn > 1 and turn % _REMINDER_INTERVAL == 0:
            q_snippet = user_input[:300] + ("…" if len(user_input) > 300 else "")
            messages.append({"role": "system", "content": (
                f"🔎 REMINDER — You are answering this question:\n"
                f"{q_snippet}\n\n"
                "Stay focused on gathering information relevant to this question. "
                "If you already have enough, call final_answer."
            )})
            if verbose:
                print(f"🔎  Injected periodic question reminder (turn {turn})")

        # ── Draft injection (root only, every turn after first draft) ────
        # The model always sees its current document so it can identify gaps.
        tools_for_turn = None  # None = use full available_tools
        if _depth == 0 and draft_versions:
            _draft_turn, _draft_text = draft_versions[-1]
            _draft_preview = _draft_text[:4000] + ("…" if len(_draft_text) > 4000 else "")
            messages.append({"role": "system", "content": (
                f"📄 YOUR CURRENT DRAFT (v{len(draft_versions)}, saved turn {_draft_turn}, "
                f"{len(_draft_text):,} chars):\n"
                f"{_draft_preview}\n\n"
                "Review your draft. Are there gaps? Delegate more research with "
                "conduct_research. When satisfied, call research_complete to publish."
            )})
            if verbose:
                print(f"📄  Injected draft v{len(draft_versions)} ({len(_draft_text):,} chars)")
        
        # ── research_complete gate (root only) ───────────────────────────
        # research_complete is hidden until a draft exists.
        if _depth == 0 and not draft_versions:
            tools_for_turn = [t for t in available_tools if t["function"]["name"] != "research_complete"]
            if verbose and turn == 1:
                print(f"🔒 research_complete hidden — no draft yet")

        # ── Budget-aware tool restriction (sub-agents only) ───────────────
        # Root has no turn limit. Sub-agents get periodic checkpoints and
        # escalating warnings based on remaining budget.
        #
        # checkpoint_interval = turn_length // 3  (modulo-based)
        #   budget=10 → interval=3  → checkpoints at turns 3, 6
        #   budget=15 → interval=5  → checkpoints at turns 5, 10
        #   budget=12 → interval=4  → checkpoints at turns 4, 8
        # Final-third warnings at remaining=3, 2, 1; force at remaining=0.
        if _depth > 0 and turn_length is not None:
            remaining = turn_length - turn
            q_echo = user_input[:500] + ("…" if len(user_input) > 500 else "")
            _ckpt_interval = max(turn_length // 3, 2)

            # Periodic checkpoint (fires at each interval, skipping the last 3 turns)
            if turn > 0 and turn % _ckpt_interval == 0 and remaining >= _ckpt_interval:
                messages.append({"role": "system", "content": (
                    f"📊 CHECKPOINT (turn {turn}/{turn_length}): "
                    "Pause and assess your progress.\n"
                    "  1. What key facts have you found so far?\n"
                    "  2. Do you have enough to answer the question?\n"
                    "  3. If YES → call final_answer now with what you have.\n"
                    "  4. If NO → what ONE specific thing is still missing?\n\n"
                    f"Question: {q_echo}\n\n"
                    "A good answer with 3 sources beats a perfect answer with 0. "
                    "Call think() to assess, then act."
                )})
                if verbose:
                    print(f"📊  Checkpoint injected (turn {turn}/{turn_length}, interval={_ckpt_interval})")

            # Escalating warnings in the final segment
            if remaining == 3:
                messages.append({"role": "system", "content": (
                    "⚠️ BUDGET WARNING: You have 4 turns remaining (including this one). "
                    "Start synthesizing NOW. Call think() to organize your findings, "
                    "then call final_answer on the next turn.\n"
                    f"Question: {q_echo}"
                )})
            elif remaining == 2:
                messages.append({"role": "system", "content": (
                    "🚨 FINAL WARNING: 3 turns left. Call final_answer THIS TURN "
                    "or NEXT TURN. Do NOT start new searches."
                )})
            elif remaining == 1:
                messages.append({"role": "system", "content": (
                    "🚨 LAST CHANCE: 2 turns remaining. Call final_answer NOW."
                )})
            elif remaining == 0:
                tools_for_turn = [TERMINAL_TOOL]
                if verbose:
                    print(f"🔒 Restricting to {_terminal_tool_name} only (last turn)")

        # Call vLLM with tools
        response = _call_api(effective_max_tokens, tools_override=tools_for_turn)
        
        result = response.json()
        
        if response.status_code != 200:
            error_msg = str(result.get("error", {}).get("message", ""))
            
            # Case 1: Context overflow — retry with reduced max_tokens
            if "max_tokens" in error_msg or "max_completion_tokens" in error_msg:
                import re as _re
                match = _re.search(r'has (\d+) input tokens', error_msg)
                if match:
                    input_tokens = int(match.group(1))
                    effective_max_tokens = context_window - input_tokens - TOKEN_SAFETY_MARGIN
                    if effective_max_tokens >= 1:
                        if verbose:
                            print(f"⚠️  max_tokens too large, retrying with {effective_max_tokens}")
                        response = _call_api(effective_max_tokens, tools_override=tools_for_turn)
                        result = response.json()
                        if response.status_code != 200:
                            if verbose:
                                print(f"❌ API Error: {result}")
                            return _finalize(f"Error: {result}")
                    else:
                        if verbose:
                            print(f"❌ Context window exhausted ({input_tokens} input tokens, {context_window} max)")
                        return _finalize(f"Error: Context window exhausted. Input too long ({input_tokens} tokens).")
                else:
                    if verbose:
                        print(f"❌ API Error: {result}")
                    return _finalize(f"Error: {result}")
            
            # Case 2: vLLM tool-call parser error (recoverable)
            # e.g. "unexpected tokens remaining in message header: Some("to=functions.execute_code")"
            # The model emitted a malformed tool call header. Nudge and retry.
            elif "unexpected tokens" in error_msg or "tool_call" in error_msg.lower() or result.get("error", {}).get("code") == 400:
                if verbose:
                    print(f"⚠️  vLLM parser error (recoverable): {error_msg[:150]}")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed. "
                        "Call your tool again with correct formatting. "
                        "Use exact tool names from your available tools."
                    )
                })
                turn_record = TurnRecord(
                    turn_number=turn,
                    assistant_content=None,
                    raw_assistant_message={"error": error_msg},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )
                turn_record.duration_s = round(time.time() - turn_start, 3)
                episode.turns.append(turn_record)
                continue
            
            else:
                if verbose:
                    print(f"❌ API Error: {result}")
                return _finalize(f"Error: {result}")
        
        # Guard against malformed API responses (200 but missing expected fields)
        # vLLM sometimes returns HTTP 200 with an error body (e.g. generation failures).
        # Detect this and treat it as a recoverable error.
        if result.get("error"):
            err_detail = result["error"]
            if verbose:
                print(f"⚠️  vLLM returned 200 with error body: {str(err_detail)[:200]}")
            messages.append({
                "role": "user",
                "content": (
                    "The server encountered an internal error processing your last response. "
                    "Please try your tool call again. Use a simpler approach if possible."
                )
            })
            turn_record = TurnRecord(
                turn_number=turn,
                assistant_content=None,
                raw_assistant_message={"error": str(err_detail)},
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
            )
            turn_record.duration_s = round(time.time() - turn_start, 3)
            episode.turns.append(turn_record)
            continue
        
        choices = result.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            if verbose:
                print(f"❌ Malformed API response (no choices): {str(result)[:300]}")
            return _finalize(f"Error: Malformed API response — no 'choices' returned. Raw: {str(result)[:200]}")
        
        assistant_message = choices[0].get("message")
        if not assistant_message:
            if verbose:
                print(f"❌ Malformed API response (no message in choice): {str(choices[0])[:300]}")
            return _finalize(f"Error: Malformed API response — no 'message' in choice. Raw: {str(choices[0])[:200]}")
        
        usage = result.get("usage", {})
        
        # Strip non-standard fields from assistant message before appending.
        # vLLM's reasoning parser (deepseek_r1) adds 'reasoning', 'reasoning_content',
        # etc. to responses, but chokes with a 500 if they're sent back as input.
        # Keep only the standard OpenAI fields.
        clean_msg = {
            "role": assistant_message["role"],
            "content": assistant_message.get("content") or "",
        }
        if assistant_message.get("tool_calls"):
            clean_msg["tool_calls"] = assistant_message["tool_calls"]
        messages.append(clean_msg)
        
        # Start building turn record (keep full raw message for trace)
        turn_record = TurnRecord(
            turn_number=turn,
            assistant_content=assistant_message.get("content"),
            raw_assistant_message=assistant_message,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        
        # With tool_choice="required", the model MUST produce tool_calls.
        # Process them, and if any call is research_complete/final_answer → terminate.
        tool_calls_in_msg = assistant_message.get("tool_calls") or []
        if not tool_calls_in_msg:
            # Shouldn't happen with tool_choice="required", but vLLM's tool-call
            # parser sometimes fails to extract calls from the model output,
            # returning empty tool_calls with content text instead.
            consecutive_no_tool_count += 1
            
            final_content = assistant_message.get("content", "") or ""
            
            # ── Auto-draft capture (root only) ───────────────────────
            # If the model wrote substantive text without a tool call,
            # capture it as a draft so it's never lost.
            if _depth == 0 and len(final_content.strip()) > 200 and _draft_path:
                _STRUCTURE_SIGNALS = ("|", "#", "- ", "1.", "1)", "{", "**")
                _has_structure = any(s in final_content[:500] for s in _STRUCTURE_SIGNALS)
                if _has_structure or len(final_content.strip()) > 400:
                    draft_versions.append((turn, final_content.strip()))
                    ver = len(draft_versions)
                    try:
                        with open(_draft_path, "w", encoding="utf-8") as f:
                            f.write(f"<!-- Auto-draft v{ver} | turn {turn} -->\n")
                            f.write(final_content.strip())
                        memory.upsert(
                            key="draft_latest",
                            tool_name="auto_draft",
                            turn=turn,
                            content=final_content.strip(),
                            description="auto-captured from plain text",
                        )
                    except Exception:
                        pass
                    if verbose:
                        print(f"📝  Auto-captured plain text as Draft v{ver} ({len(final_content):,} chars)")
            
            if consecutive_no_tool_count >= 3:
                # Model has degenerated — it can't produce tool calls anymore.
                if _depth == 0 and draft_versions:
                    # Root with a draft — use it
                    _, latest_draft = draft_versions[-1]
                    if verbose:
                        print(f"⚠️  Model degenerated at root — using draft v{len(draft_versions)}")
                    turn_record.duration_s = round(time.time() - turn_start, 3)
                    episode.turns.append(turn_record)
                    return _finalize(latest_draft)
                elif len(final_content.strip()) > 200:
                    # Substantive text — accept as final answer
                    if verbose:
                        print(f"⚠️  Model produced text without tool calls {consecutive_no_tool_count}x — accepting as final answer")
                    turn_record.duration_s = round(time.time() - turn_start, 3)
                    episode.turns.append(turn_record)
                    return _finalize(final_content)
                else:
                    # Empty/garbage — break to synthesis pipeline
                    if verbose:
                        print(f"⚠️  Model degenerated ({consecutive_no_tool_count}x no tool calls) — breaking to synthesis")
                    _degenerated = True
                    turn_record.duration_s = round(time.time() - turn_start, 3)
                    episode.turns.append(turn_record)
                    break
            else:
                # Nudge model to call a tool
                if verbose:
                    print(f"⚠️  No tool calls returned (attempt {consecutive_no_tool_count}/3) — nudging")
                content_echo = final_content[:3000] if final_content.strip() else ""
                if _depth == 0:
                    # Root: nudge toward its tool set
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
                    # Sub-agents: nudge toward final_answer
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
                messages.append({"role": "user", "content": nudge})
                turn_record.duration_s = round(time.time() - turn_start, 3)
                episode.turns.append(turn_record)
                continue
        else:
            # Reset counter on successful tool call
            consecutive_no_tool_count = 0

        total_tool_calls += len(tool_calls_in_msg)
        if verbose:
            print(f"🔧 Tool calls: {len(tool_calls_in_msg)}")
        
        # Process each tool call
        final_answer_result = None
        for i, tool_call in enumerate(tool_calls_in_msg, 1):
            tool_name = tool_call["function"]["name"]
            # Sanitize: strip leaked channel tokens (e.g. "search_web<|channel|>commentary")
            if "<|" in tool_name:
                tool_name = tool_name.split("<|")[0]
            raw_args = tool_call["function"].get("arguments", "")
            _args_were_malformed = False
            try:
                tool_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                _args_were_malformed = True
                # For final_answer / refine_draft, try regex extraction before giving up
                if tool_name in ("final_answer", "refine_draft") and raw_args:
                    _recovered = _recover_final_answer_from_raw(raw_args)
                    if _recovered:
                        _param = "answer" if tool_name == "final_answer" else "content"
                        tool_args = {_param: _recovered}
                        if verbose:
                            print(f"   🔧 Recovered {tool_name} from malformed JSON ({len(_recovered)} chars)")
                    else:
                        tool_args = {}
                        if verbose:
                            print(f"   ⚠️  Malformed tool arguments for {tool_name}, could not recover")
                else:
                    tool_args = {}
                    if verbose:
                        print(f"   ⚠️  Malformed tool arguments for {tool_name}, using empty args")
            
            if verbose:
                print(f"   [{i}] {tool_name}")
            
            # ══════════════════════════════════════════════════════════
            # ══════════════════════════════════════════════════════════
            # VIRTUAL TOOLS (all depths)
            # ══════════════════════════════════════════════════════════
            
            # ── think (all depths) ────────────────────────────────────
            if tool_name == "think":
                thought = tool_args.get("thought", "").strip()
                # Reject empty or shallow thinks — must be a real reflection
                _MIN_THINK_CHARS = 20
                if len(thought) < _MIN_THINK_CHARS:
                    think_output = (
                        "[Think rejected — too brief] Your think() call must be a "
                        "substantive reflection. Include:\n"
                        "  1. What data you have gathered so far\n"
                        "  2. What is still missing for the task\n"
                        "  3. Your plan for the next 1-2 tool calls\n"
                        "Call think() again with at least a few sentences of reflection."
                    )
                    if verbose:
                        print(f"       💭 [rejected — {len(thought)} chars < {_MIN_THINK_CHARS}]")
                else:
                    think_output = f"[Thought recorded]\n{thought}"
                    # Only substantive thinks break the consecutive search chain
                    _consecutive_search_count = 0
                    if verbose:
                        _preview = thought[:120] + ("…" if len(thought) > 120 else "")
                        print(f"       💭 {_preview}")
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=think_output,
                    duration_s=0, child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": think_output,
                })
                continue
            
            # ── search_available_tools (all depths) ───────────────────
            # Intercept here so we list only the tools this agent actually
            # has (available_tools), not the global TOOLS list.
            if tool_name == "search_available_tools":
                _sat_tool_name = tool_args.get("tool_name")
                if _sat_tool_name:
                    # Return full schema for one specific tool
                    _sat_match = next(
                        (t for t in available_tools if t["function"]["name"] == _sat_tool_name),
                        None,
                    )
                    if _sat_match:
                        _sat_output = json.dumps(_sat_match, indent=2)
                    else:
                        _sat_names = [t["function"]["name"] for t in available_tools]
                        _sat_output = (
                            f"No tool named '{_sat_tool_name}'. "
                            f"Available tools: {', '.join(_sat_names)}"
                        )
                else:
                    # Compact summary of all available tools
                    _sat_lines = []
                    for _sat_t in available_tools:
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
                    _sat_output = f"Available tools ({len(available_tools)}):\n" + "\n".join(_sat_lines)
                
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=_sat_output,
                    duration_s=0, child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _sat_output,
                })
                if verbose:
                    print(f"       📋 Listed {len(available_tools)} available tools")
                continue
            
            # ══════════════════════════════════════════════════════════
            # ROOT-ONLY VIRTUAL TOOLS (depth == 0)
            # ══════════════════════════════════════════════════════════
            
            # ── refine_draft (root only) ──────────────────────────────
            if tool_name == "refine_draft" and _depth == 0:
                draft_content = tool_args.get("content", "")
                if len(draft_content.strip()) < 50:
                    draft_output = (
                        "Draft too short — write a complete, self-contained answer. "
                        "Include findings, citations, and structure. "
                        "Each call replaces the entire draft."
                    )
                else:
                    draft_versions.append((turn, draft_content))
                    ver = len(draft_versions)
                    
                    # Write to draft file
                    try:
                        with open(_draft_path, "w", encoding="utf-8") as f:
                            f.write(f"<!-- Draft v{ver} | turn {turn} -->\n")
                            f.write(draft_content)
                    except Exception as e:
                        if verbose:
                            print(f"       ⚠️  Draft file write failed: {e}")
                    
                    # MemoryStore: upsert so sub-agents can access via memory_keys
                    memory.upsert(
                        key="draft_latest",
                        tool_name="refine_draft",
                        turn=turn,
                        content=draft_content,
                        description="latest draft answer",
                    )
                    
                    draft_output = (
                        f"✅ Draft v{ver} saved ({len(draft_content):,} chars).\n\n"
                        f"Next steps:\n"
                        f"  - To research gaps: conduct_research(task='...')\n"
                        f"  - To update draft: refine_draft(content='...')\n"
                        f"  - To publish: research_complete()"
                    )
                    if verbose:
                        print(f"       📝 Draft v{ver} saved ({len(draft_content):,} chars) → {_draft_path}")
                
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=draft_output,
                    duration_s=0, child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": draft_output,
                })
                continue
            
            # ── research_complete (root only) ─────────────────────────
            if tool_name == "research_complete" and _depth == 0:
                # Read draft from file
                _draft_content = ""
                if _draft_path and os.path.exists(_draft_path):
                    try:
                        with open(_draft_path, "r", encoding="utf-8") as f:
                            _raw = f.read()
                        # Strip the HTML comment header
                        _draft_content = re.sub(r'^<!--.*?-->\n?', '', _raw, count=1).strip()
                    except Exception as e:
                        if verbose:
                            print(f"       ⚠️  Failed to read draft file: {e}")
                
                if len(_draft_content.strip()) < 50:
                    # Draft is empty/too short — reject and ask to draft first
                    _reject_msg = (
                        "Cannot publish — your draft is empty or too short. "
                        "Call refine_draft(content='...') first with a complete answer, "
                        "then call research_complete() to publish it."
                    )
                    tc_record = ToolCallRecord(
                        tool_name=tool_name, tool_args=tool_args,
                        tool_call_id=tool_call["id"], output=_reject_msg,
                        duration_s=0, child_trace=None,
                    )
                    turn_record.tool_calls.append(tc_record)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": _reject_msg,
                    })
                    if verbose:
                        print(f"       ⚠️  research_complete rejected — draft is empty")
                    continue

                # ── Format compliance check (deterministic, before LLM verifier) ──
                _format_issues = []
                if "**Final Answer:**" not in _draft_content and "**Final Answer**" not in _draft_content:
                    _format_issues.append(
                        "Missing '**Final Answer:**' section. Your draft MUST start with "
                        "'**Final Answer:**' followed by the direct answer."
                    )
                if "**Sources:**" not in _draft_content and "**Sources**" not in _draft_content:
                    _format_issues.append(
                        "Missing '**Sources:**' section. Include at least one source URL."
                    )
                if _format_issues:
                    _format_msg = (
                        "Cannot publish — your draft does not follow the required format.\n\n"
                        "ISSUES:\n- " + "\n- ".join(_format_issues) + "\n\n"
                        "Required structure:\n"
                        "  **Final Answer:**\n  [direct answer]\n\n"
                        "  **Sources:**\n  - [Source](URL)\n\n"
                        "  **Details:** (optional)\n  [supporting context]\n\n"
                        "Fix with refine_draft(), then call research_complete() again."
                    )
                    tc_record = ToolCallRecord(
                        tool_name=tool_name, tool_args=tool_args,
                        tool_call_id=tool_call["id"], output=_format_msg,
                        duration_s=0, child_trace=None,
                        metadata={"format_check": "failed", "issues": _format_issues},
                    )
                    turn_record.tool_calls.append(tc_record)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": _format_msg,
                    })
                    if verbose:
                        print(f"       ⚠️  Format check failed: {_format_issues}")
                    continue

                # Draft is good — run gatekeeper verification (if enabled)
                _should_publish = True
                _verify_feedback = ""
                _verify_text_raw = ""   # full verifier response for trace
                
                if (VERIFY_BEFORE_PUBLISH
                    and VERIFIER_PROMPT
                    and _verification_rejections < _MAX_VERIFICATION_REJECTIONS):
                    try:
                        # Build findings summary for cross-checking
                        _findings_section = ""
                        if findings:
                            _findings_lines = []
                            _findings_total = 0
                            for i, f in enumerate(findings, 1):
                                _entry = f"{i}. {f}"
                                if _findings_total + len(_entry) > 15000:
                                    _findings_lines.append(f"... ({len(findings) - i + 1} more entries truncated)")
                                    break
                                _findings_lines.append(_entry)
                                _findings_total += len(_entry)
                            _findings_section = (
                                "\n\nRESEARCH FINDINGS (raw worker outputs for cross-checking):\n"
                                + "\n".join(_findings_lines)
                            )

                        _verify_messages = [
                            {
                                "role": "system",
                                "content": VERIFIER_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"QUESTION:\n{user_input}\n\n"
                                    f"DRAFT ANSWER:\n{_draft_content}"
                                    f"{_findings_section}"
                                ),
                            },
                        ]
                        _verify_payload = {
                            "model": model,
                            "messages": _verify_messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                        if verbose:
                            print(f"       🔍 Verifier: auditing draft (attempt {_verification_rejections + 1})...")
                        _verify_resp = requests.post(
                            f"{VLLM_API_URL}/chat/completions",
                            json=_verify_payload,
                        )
                        if _verify_resp.status_code == 200:
                            _verify_result = _verify_resp.json()
                            _verify_text = (
                                _verify_result.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            ).strip()
                            _verify_text_raw = _verify_text  # preserve for trace
                            # Strip markdown bold/italic (** or *) before matching
                            # — models often wrap "Verdict" in bold despite the prompt
                            _verify_clean = _verify_text.replace("*", "").upper()
                            if "VERDICT: REVISION_NEEDED" in _verify_clean or "VERDICT:REVISION_NEEDED" in _verify_clean:
                                _should_publish = False
                                _verification_rejections += 1
                                # Extract the feedback (everything after VERDICT line)
                                # Tolerate markdown: **Verdict:** or *Verdict:*
                                _verify_feedback = re.sub(
                                    r'^.*?\*{0,2}Verdict\*{0,2}:\*{0,2}\s*REVISION_NEEDED\s*',
                                    '', _verify_text, count=1, flags=re.IGNORECASE | re.DOTALL
                                ).strip()
                                if not _verify_feedback:
                                    _verify_feedback = _verify_text
                                if verbose:
                                    print(f"       ⚠️  Verifier: REVISION_NEEDED (rejection {_verification_rejections}/{_MAX_VERIFICATION_REJECTIONS})")
                            else:
                                if verbose:
                                    print(f"       ✅ Verifier: APPROVED")
                        else:
                            if verbose:
                                print(f"       ⚠️  Verifier API error ({_verify_resp.status_code}), skipping verification")
                    except Exception as e:
                        if verbose:
                            print(f"       ⚠️  Verifier error: {e}, skipping verification")
                elif _verification_rejections >= _MAX_VERIFICATION_REJECTIONS:
                    if verbose:
                        print(f"       ⏩ Verifier: max rejections reached, force-publishing")
                
                # ── STAGE 2: Spot-check (search-verify key claims) ────
                # Only runs if Stage 1 approved AND spot-check is enabled
                # AND we haven't exhausted spot-check rejections.
                _spot_check_meta = {}  # trace metadata for spot-check
                if (
                    _should_publish
                    and SPOT_CHECK_ENABLED
                    and SPOTCHECK_EXTRACT_PROMPT
                    and SPOTCHECK_COMPARE_PROMPT
                    and _spot_check_rejections < _MAX_SPOT_CHECK_REJECTIONS
                ):
                    try:
                        if verbose:
                            print(f"       🔎 Spot-check: extracting checkable claims...")

                        # Step 1: Extract claims from draft
                        _sc_extract_msgs = [
                            {"role": "system", "content": SPOTCHECK_EXTRACT_PROMPT},
                            {"role": "user", "content": (
                                f"QUESTION:\n{user_input}\n\n"
                                f"DRAFT ANSWER:\n{_draft_content}"
                            )},
                        ]
                        _sc_extract_resp = requests.post(
                            f"{VLLM_API_URL}/chat/completions",
                            json={
                                "model": model,
                                "messages": _sc_extract_msgs,
                                "temperature": 0.3,
                                "max_tokens": 1024,
                            },
                        )
                        _sc_claims = []
                        if _sc_extract_resp.status_code == 200:
                            _sc_extract_text = (
                                _sc_extract_resp.json().get("choices", [{}])[0]
                                .get("message", {}).get("content", "")
                            ).strip()
                            _spot_check_meta["extract_response"] = _sc_extract_text
                            # Parse JSON — handle markdown fences
                            _sc_json_text = re.sub(r'^```(?:json)?\s*', '', _sc_extract_text, flags=re.MULTILINE)
                            _sc_json_text = re.sub(r'```\s*$', '', _sc_json_text, flags=re.MULTILINE).strip()
                            try:
                                _sc_parsed = json.loads(_sc_json_text)
                                _sc_claims = _sc_parsed.get("claims", [])
                            except json.JSONDecodeError:
                                if verbose:
                                    print(f"       ⚠️  Spot-check: failed to parse claims JSON, skipping")
                        else:
                            if verbose:
                                print(f"       ⚠️  Spot-check: extract API error ({_sc_extract_resp.status_code}), skipping")

                        if _sc_claims:
                            if verbose:
                                print(f"       🔎 Spot-check: verifying {len(_sc_claims)} claim(s)...")

                            # Step 2: Search for each claim + fetch & summarize top result (parallelized)
                            def _verify_one_claim(claim_obj):
                                """Search + fetch + summarize for a single claim. Thread-safe."""
                                _q = claim_obj.get("search_query", "")
                                _c = claim_obj.get("claim", "")
                                if not _q:
                                    return None  # skip claims without a query
                                try:
                                    _sr, _ = dispatch_tool_call(
                                        "search_web",
                                        {"q": _q, "num_results": 5},
                                        _depth=0,
                                    )
                                    _ps = ""
                                    _um = re.search(r'URL:\s*(https?://\S+)', _sr)
                                    if _um:
                                        _top_url = _um.group(1)
                                        try:
                                            _ps = _focused_page_summary(
                                                url=_top_url,
                                                focus=f"Verify this claim: {_c}",
                                                model=model,
                                                vllm_url=VLLM_API_URL,
                                                max_summary_tokens=1024,
                                            )
                                            if _ps.startswith(("Failed to fetch", "Page fetched but", "Summarization error")):
                                                _ps = ""
                                            elif verbose:
                                                print(f"       📄 Spot-check: summarized {_top_url[:60]} ({len(_ps)} chars)")
                                        except Exception:
                                            pass
                                    return {
                                        "claim": _c,
                                        "search_query": _q,
                                        "search_results": _sr[:3000],
                                        "page_content": _ps,
                                    }
                                except Exception as e:
                                    if verbose:
                                        print(f"       ⚠️  Spot-check search error: {e}")
                                    return {
                                        "claim": _c,
                                        "search_query": _q,
                                        "search_results": f"(search failed: {e})",
                                        "page_content": "",
                                    }

                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            _sc_max_workers = min(len(_sc_claims), 4)
                            with ThreadPoolExecutor(max_workers=_sc_max_workers) as _sc_pool:
                                _sc_futures = {_sc_pool.submit(_verify_one_claim, c): c for c in _sc_claims}
                                _sc_evidence = []
                                for fut in as_completed(_sc_futures):
                                    result = fut.result()
                                    if result is not None:
                                        _sc_evidence.append(result)

                            _spot_check_meta["claims_checked"] = len(_sc_evidence)

                            # Short-circuit: if >50% of claims have degraded evidence,
                            # skip comparison and auto-PASS (no positive evidence of error).
                            if _sc_evidence:
                                _sc_degraded = sum(
                                    1 for ev in _sc_evidence
                                    if ev["search_results"].startswith("(search failed")
                                    or (not ev["page_content"] and len(ev["search_results"]) < 200)
                                )
                                if _sc_degraded > len(_sc_evidence) / 2:
                                    _spot_check_meta["spot_check_verdict"] = "PASSED"
                                    _spot_check_meta["spot_check_skipped_reason"] = "insufficient_evidence"
                                    _spot_check_meta["degraded_claims"] = _sc_degraded
                                    if verbose:
                                        print(f"       ⏩ Spot-check: {_sc_degraded}/{len(_sc_evidence)} claims have degraded evidence, auto-PASS")
                                else:
                                    pass  # fall through to Step 3

                            if _sc_evidence and "spot_check_skipped_reason" not in _spot_check_meta:
                                # Step 3: Compare claims against search evidence + page content
                                _sc_evidence_parts = []
                                for i, ev in enumerate(_sc_evidence, 1):
                                    _part = (
                                        f"CLAIM {i}: {ev['claim']}\n"
                                        f"SEARCH QUERY: {ev['search_query']}\n"
                                        f"SEARCH RESULTS:\n{ev['search_results']}"
                                    )
                                    if ev.get("page_content"):
                                        _part += (
                                            f"\n\nSUMMARY OF TOP RESULT:\n"
                                            f"{ev['page_content']}"
                                        )
                                    _sc_evidence_parts.append(_part)
                                _sc_evidence_text = "\n\n".join(_sc_evidence_parts)
                                _sc_compare_msgs = [
                                    {"role": "system", "content": SPOTCHECK_COMPARE_PROMPT},
                                    {"role": "user", "content": (
                                        f"QUESTION:\n{user_input}\n\n"
                                        f"DRAFT ANSWER (excerpt of checked claims):\n\n"
                                        f"{_sc_evidence_text}"
                                    )},
                                ]
                                _sc_compare_resp = requests.post(
                                    f"{VLLM_API_URL}/chat/completions",
                                    json={
                                        "model": model,
                                        "messages": _sc_compare_msgs,
                                        "temperature": 0.3,
                                        "max_tokens": 2048,
                                    },
                                )
                                if _sc_compare_resp.status_code == 200:
                                    _sc_compare_text = (
                                        _sc_compare_resp.json().get("choices", [{}])[0]
                                        .get("message", {}).get("content", "")
                                    ).strip()
                                    _spot_check_meta["compare_response"] = _sc_compare_text

                                    # Strip markdown bold/italic before matching
                                    _sc_clean = _sc_compare_text.replace("*", "").upper()
                                    if "VERDICT: SPOT_CHECK_FAILED" in _sc_clean or "VERDICT:SPOT_CHECK_FAILED" in _sc_clean:
                                        _should_publish = False
                                        _spot_check_rejections += 1
                                        _spot_check_meta["spot_check_verdict"] = "FAILED"
                                        # Extract feedback after the VERDICT line
                                        _verify_feedback = re.sub(
                                            r'^.*?\*{0,2}Verdict\*{0,2}:\*{0,2}\s*SPOT_CHECK_FAILED\s*',
                                            '', _sc_compare_text, count=1,
                                            flags=re.IGNORECASE | re.DOTALL,
                                        ).strip()
                                        if not _verify_feedback:
                                            _verify_feedback = _sc_compare_text
                                        _verify_text_raw = f"[SPOT-CHECK] {_sc_compare_text}"
                                        if verbose:
                                            print(f"       ❌ Spot-check: FAILED — factual issues found (rejection {_spot_check_rejections}/{_MAX_SPOT_CHECK_REJECTIONS})")
                                    else:
                                        _spot_check_meta["spot_check_verdict"] = "PASSED"
                                        if verbose:
                                            print(f"       ✅ Spot-check: PASSED")
                                else:
                                    if verbose:
                                        print(f"       ⚠️  Spot-check: compare API error ({_sc_compare_resp.status_code}), skipping")
                        else:
                            # ── Refusal Challenge ─────────────────────────────
                            # No checkable claims — might be a refusal.
                            # If the draft says "couldn't find / unavailable / etc."
                            # we do an independent search to see if it's honest.
                            _refusal_patterns = [
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
                            _draft_lower = _draft_content.lower()
                            _is_refusal = any(re.search(p, _draft_lower) for p in _refusal_patterns)

                            if _is_refusal and SPOTCHECK_REFUSAL_PROMPT:
                                if verbose:
                                    print(f"       🔎 Spot-check: draft looks like a refusal — launching refusal challenge...")

                                # Step A: Extract a search query from the question itself
                                # (simple approach: just use the original question as query)
                                _rc_query = user_input[:200]  # keep reasonable length

                                try:
                                    # Step B: Independent search
                                    _rc_search_result, _ = dispatch_tool_call(
                                        "search_web",
                                        {"q": _rc_query, "num_results": 5},
                                        _depth=0,
                                    )

                                    # Step C: Fetch & summarize the top result
                                    _rc_page_summary = ""
                                    _rc_url_match = re.search(r'URL:\s*(https?://\S+)', _rc_search_result)
                                    if _rc_url_match:
                                        _rc_top_url = _rc_url_match.group(1)
                                        try:
                                            _rc_page_summary = _focused_page_summary(
                                                url=_rc_top_url,
                                                focus=f"Find information relevant to: {user_input[:300]}",
                                                model=model,
                                                vllm_url=VLLM_API_URL,
                                                max_summary_tokens=1024,
                                            )
                                            if _rc_page_summary.startswith(("Failed to fetch", "Page fetched but", "Summarization error")):
                                                _rc_page_summary = ""
                                        except Exception:
                                            pass

                                    # Step D: LLM adjudicates
                                    _rc_evidence_text = (
                                        f"SEARCH RESULTS:\n{_rc_search_result[:4000]}"
                                    )
                                    if _rc_page_summary:
                                        _rc_evidence_text += (
                                            f"\n\nSUMMARY OF TOP RESULT:\n{_rc_page_summary}"
                                        )

                                    _rc_msgs = [
                                        {"role": "system", "content": SPOTCHECK_REFUSAL_PROMPT},
                                        {"role": "user", "content": (
                                            f"QUESTION:\n{user_input}\n\n"
                                            f"DRAFT ANSWER (REFUSAL):\n{_draft_content[:2000]}\n\n"
                                            f"INDEPENDENT EVIDENCE:\n{_rc_evidence_text}"
                                        )},
                                    ]
                                    _rc_resp = requests.post(
                                        f"{VLLM_API_URL}/chat/completions",
                                        json={
                                            "model": model,
                                            "messages": _rc_msgs,
                                            "temperature": 0.3,
                                            "max_tokens": 1024,
                                        },
                                    )
                                    if _rc_resp.status_code == 200:
                                        _rc_text = (
                                            _rc_resp.json().get("choices", [{}])[0]
                                            .get("message", {}).get("content", "")
                                        ).strip()
                                        _spot_check_meta["refusal_challenge_response"] = _rc_text
                                        _rc_clean = _rc_text.replace("*", "").upper()

                                        if "VERDICT: REFUSAL_CHALLENGED" in _rc_clean or "VERDICT:REFUSAL_CHALLENGED" in _rc_clean:
                                            _should_publish = False
                                            _spot_check_rejections += 1
                                            # Extract feedback after the VERDICT line
                                            _verify_feedback = re.sub(
                                                r'^.*?\*{0,2}Verdict\*{0,2}:\*{0,2}\s*REFUSAL_CHALLENGED\s*',
                                                '', _rc_text, count=1,
                                                flags=re.IGNORECASE | re.DOTALL,
                                            ).strip()
                                            if not _verify_feedback:
                                                _verify_feedback = _rc_text
                                            _verify_text_raw = f"[SPOT-CHECK] Refusal challenged: {_rc_text}"
                                            _spot_check_meta["spot_check_verdict"] = "REFUSAL_CHALLENGED"
                                            if verbose:
                                                print(f"       ❌ Refusal challenge: evidence found — refusal was premature (rejection {_spot_check_rejections}/{_MAX_SPOT_CHECK_REJECTIONS})")
                                        else:
                                            _spot_check_meta["spot_check_verdict"] = "REFUSAL_JUSTIFIED"
                                            if verbose:
                                                print(f"       ✅ Refusal challenge: refusal appears justified")
                                    else:
                                        if verbose:
                                            print(f"       ⚠️  Refusal challenge: API error ({_rc_resp.status_code}), skipping")

                                except Exception as e:
                                    if verbose:
                                        print(f"       ⚠️  Refusal challenge error: {e}, skipping")
                            elif verbose:
                                print(f"       ℹ️  Spot-check: no checkable claims extracted, skipping")

                    except Exception as e:
                        if verbose:
                            print(f"       ⚠️  Spot-check error: {e}, skipping")
                elif _spot_check_rejections >= _MAX_SPOT_CHECK_REJECTIONS:
                    if verbose:
                        print(f"       ⏩ Spot-check: max rejections reached, skipping")

                if _should_publish:
                    # Publish the draft
                    _verify_meta = {}
                    if _verify_text_raw:
                        _verify_meta["verification_verdict"] = "APPROVED"
                        _verify_meta["verification_response"] = _verify_text_raw
                    _verify_meta["verification_attempt"] = _verification_rejections + 1
                    # Include spot-check metadata if it ran
                    if _spot_check_meta:
                        _verify_meta["spot_check"] = _spot_check_meta
                    tc_record = ToolCallRecord(
                        tool_name=tool_name, tool_args=tool_args,
                        tool_call_id=tool_call["id"], output=_draft_content,
                        duration_s=0, child_trace=None,
                        metadata=_verify_meta if _verify_meta else None,
                    )
                    turn_record.tool_calls.append(tc_record)
                    final_answer_result = _draft_content
                    if verbose:
                        print(f"   ✅ research_complete — publishing draft ({len(_draft_content):,} chars)")
                else:
                    # Return verifier/spot-check feedback to root
                    _is_spot_check_reject = _verify_text_raw.startswith("[SPOT-CHECK]")
                    if _is_spot_check_reject:
                        _reject_msg = (
                            f"Spot-check found factual issues in your draft that should be corrected:\n\n"
                            f"{_verify_feedback}\n\n"
                            f"Please correct these specific claims with refine_draft(), then call "
                            f"research_complete() again. "
                            f"(Spot-check rejection {_spot_check_rejections}/{_MAX_SPOT_CHECK_REJECTIONS} — "
                            f"{'next attempt will force-publish' if _spot_check_rejections >= _MAX_SPOT_CHECK_REJECTIONS else f'{_MAX_SPOT_CHECK_REJECTIONS - _spot_check_rejections} correction(s) remaining'})"
                        )
                    else:
                        _reject_msg = (
                            f"Draft review found issues that should be addressed before publishing:\n\n"
                            f"{_verify_feedback}\n\n"
                            f"Please fix these issues with refine_draft(), then call research_complete() again. "
                            f"(Attempt {_verification_rejections}/{_MAX_VERIFICATION_REJECTIONS} — "
                            f"{'next attempt will force-publish' if _verification_rejections >= _MAX_VERIFICATION_REJECTIONS else f'{_MAX_VERIFICATION_REJECTIONS - _verification_rejections} revision(s) remaining'})"
                        )
                    _reject_meta = {
                        "verification_verdict": "SPOT_CHECK_FAILED" if _is_spot_check_reject else "REVISION_NEEDED",
                        "verification_response": _verify_text_raw or _verify_feedback,
                        "verification_attempt": _verification_rejections,
                        "spot_check_attempt": _spot_check_rejections,
                    }
                    if _spot_check_meta:
                        _reject_meta["spot_check"] = _spot_check_meta
                    tc_record = ToolCallRecord(
                        tool_name=tool_name, tool_args=tool_args,
                        tool_call_id=tool_call["id"], output=_reject_msg,
                        duration_s=0, child_trace=None,
                        metadata=_reject_meta,
                    )
                    turn_record.tool_calls.append(tc_record)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": _reject_msg,
                    })
                continue
            
            # ── conduct_research (root only) ──────────────────────────
            if tool_name == "conduct_research" and _depth == 0:
                # Dispatch as a sub-agent (same as old spawn_agent)
                tc_start = time.time()
                child_trace = None
                try:
                    # Remap to spawn_agent_wrapper parameters
                    output, child_trace = dispatch_tool_call(
                        "spawn_agent", tool_args,
                        _depth=_depth, model=model,
                        reasoning_effort=reasoning_effort,
                        _sandbox_files=_sandbox_files,
                        _memory_store=memory,
                    )
                    if verbose and len(output) < 200:
                        print(f"       → {output}")
                    elif verbose:
                        print(f"       → {output[:200]}...")
                except Exception as e:
                    output = f"ERROR: {str(e)}"
                    if verbose:
                        print(f"       → ❌ {output}")
                tc_duration = round(time.time() - tc_start, 3)
                
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=output,
                    duration_s=tc_duration, child_trace=child_trace,
                )
                turn_record.tool_calls.append(tc_record)
                
                # Track in findings and memory
                if not output.startswith("ERROR:"):
                    findings.append(f"[conduct_research] {output[:1500].strip()}")
                    desc = str(tool_args.get("task", ""))[:60]
                    mem_key = memory.add(
                        tool_name="conduct_research",
                        turn=turn,
                        content=output,
                        description=desc,
                    )
                    if plan is not None:
                        plan.record_tool_call(
                            tool_name="conduct_research",
                            tool_args=tool_args,
                            output=output,
                            memory_key=mem_key,
                            is_error=False,
                        )
                else:
                    mem_key = None
                    if plan is not None:
                        plan.record_tool_call(
                            tool_name="conduct_research",
                            tool_args=tool_args,
                            output=output,
                            memory_key=None,
                            is_error=True,
                        )
                
                # Build the message for context
                msg_output = re.sub(
                    r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
                    r"\1[file content saved to trace]\2",
                    output, flags=re.DOTALL
                )
                if (
                    SYMBOLIC_REFERENCES
                    and mem_key is not None
                    and len(msg_output) > SYMBOLIC_THRESHOLD
                ):
                    msg_output = make_symbolic(
                        tool_name="conduct_research",
                        tool_args=tool_args,
                        output=msg_output,
                        memory_key=mem_key,
                    )
                    if verbose:
                        print(f"       📎 Symbolic ref: {len(output):,} → {len(msg_output):,} chars [{mem_key}]")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(msg_output),
                })
                continue
            
            # ── summarize_webpage (root only) ─────────────────────────
            # Lightweight: fetch URL directly + single LLM call to summarize.
            # No sub-agent overhead — much faster than conduct_research for
            # targeted page reads where the root already knows the URL.
            if tool_name == "summarize_webpage" and _depth == 0:
                tc_start = time.time()
                _sw_url = tool_args.get("url", "")
                _sw_focus = tool_args.get("focus", "")
                
                if not _sw_url:
                    _sw_output = "ERROR: url is required."
                else:
                    try:
                        _sw_output = _focused_page_summary(
                            url=_sw_url,
                            focus=_sw_focus,
                            model=model,
                            vllm_url=VLLM_API_URL,
                            max_summary_tokens=min(4096, max_tokens or 4096),
                        )
                        # Add conduct_research hint on fetch failures
                        if _sw_output.startswith("Failed to fetch") or _sw_output.startswith("Page fetched but"):
                            _sw_output += (
                                " Try conduct_research instead — a sub-agent can "
                                "retry with different strategies."
                            )
                    except Exception as e:
                        _sw_output = f"ERROR: {str(e)}"
                
                tc_duration = round(time.time() - tc_start, 3)
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=_sw_output,
                    duration_s=tc_duration, child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                
                # Track in findings and memory
                if not _sw_output.startswith("ERROR:"):
                    findings.append(f"[summarize_webpage] {_sw_output[:1500].strip()}")
                    _sw_desc = f"summary of {_sw_url[:50]}"
                    if _sw_focus:
                        _sw_desc += f" (focus: {_sw_focus[:30]})"
                    mem_key = memory.add(
                        tool_name="summarize_webpage",
                        turn=turn,
                        content=_sw_output,
                        description=_sw_desc,
                    )
                    if plan is not None:
                        plan.record_tool_call(
                            tool_name="summarize_webpage",
                            tool_args=tool_args,
                            output=_sw_output,
                            memory_key=mem_key,
                            is_error=False,
                        )
                else:
                    mem_key = None
                    if plan is not None:
                        plan.record_tool_call(
                            tool_name="summarize_webpage",
                            tool_args=tool_args,
                            output=_sw_output,
                            memory_key=None,
                            is_error=True,
                        )
                
                # Build message — apply symbolic compression if large
                msg_output = _sw_output
                if (
                    SYMBOLIC_REFERENCES
                    and mem_key is not None
                    and len(msg_output) > SYMBOLIC_THRESHOLD
                ):
                    msg_output = make_symbolic(
                        tool_name="summarize_webpage",
                        tool_args=tool_args,
                        output=msg_output,
                        memory_key=mem_key,
                    )
                    if verbose:
                        print(f"       📎 Symbolic ref: {len(_sw_output):,} → {len(msg_output):,} chars [{mem_key}]")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(msg_output),
                })
                if verbose:
                    _preview = _sw_output[:200].replace('\n', ' ')
                    print(f"       🌐 Summarized {_sw_url[:60]} ({tc_duration:.1f}s): {_preview}...")
                continue
            
            # ── compress_findings (root only) ─────────────────────────
            # Reads all MemoryStore entries, sends them to a single LLM
            # call for structured compression. Returns organized summary
            # without polluting root context with raw data.
            if tool_name == "compress_findings" and _depth == 0:
                tc_start = time.time()
                _cf_focus = tool_args.get("focus", "")
                
                # Gather memory entries
                _cf_entries = memory.entries
                if not _cf_entries:
                    _cf_output = (
                        "No research findings to compress yet. "
                        "Use conduct_research first to gather data, then call "
                        "compress_findings to organize it."
                    )
                else:
                    try:
                        # Build the data block from memory entries
                        _cf_data_parts = []
                        _cf_total_chars = 0
                        for entry in _cf_entries:
                            _entry_content = entry.content
                            # Cap individual entries to prevent single huge entry from dominating
                            if len(_entry_content) > 5000:
                                _entry_content = _entry_content[:5000] + "\n... [truncated]"
                            _cf_data_parts.append(
                                f"=== [{entry.key}] "
                                f"(source: {entry.source_tool}, "
                                f"turn {entry.turn}) ===\n"
                                f"{_entry_content}"
                            )
                            _cf_total_chars += len(_entry_content)
                        
                        _cf_data_block = "\n\n".join(_cf_data_parts)
                        
                        # Cap total to avoid context overflow on the compression call
                        if len(_cf_data_block) > 60000:
                            _cf_data_block = _cf_data_block[:60000] + "\n\n... [remaining entries truncated]"
                        
                        _focus_instruction = (
                            f"\n\nPRIORITY FOCUS: {_cf_focus}"
                            if _cf_focus else ""
                        )
                        
                        _compress_messages = [
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
                                    "  - Be concise but complete — compress prose, keep facts\n"
                                    "  - If a finding is well-supported by multiple sources, note that"
                                    f"{_focus_instruction}"
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Compress and organize the following {len(_cf_entries)} research "
                                    f"entries ({_cf_total_chars:,} total characters):\n\n"
                                    f"{_cf_data_block}"
                                ),
                            }
                        ]
                        
                        _cf_payload = {
                            "model": model,
                            "messages": _compress_messages,
                            "temperature": 0.3,
                            "max_tokens": min(8192, max_tokens),
                        }
                        _cf_resp = requests.post(
                            f"{VLLM_API_URL}/chat/completions",
                            json=_cf_payload,
                        )
                        if _cf_resp.status_code == 200:
                            _cf_result = _cf_resp.json()
                            _cf_choices = _cf_result.get("choices", [])
                            if _cf_choices:
                                _cf_output = _cf_choices[0].get("message", {}).get("content", "")
                                if not _cf_output or not _cf_output.strip():
                                    _cf_output = (
                                        f"LLM returned empty compression. "
                                        f"You have {len(_cf_entries)} entries in memory. "
                                        f"Try refine_draft directly using the findings summaries."
                                    )
                            else:
                                _cf_output = "LLM returned no choices for compression."
                        else:
                            _error_detail = ""
                            try:
                                _error_detail = _cf_resp.json().get("error", {}).get("message", "")[:200]
                            except Exception:
                                pass
                            _cf_output = (
                                f"Compression LLM call failed (HTTP {_cf_resp.status_code}). "
                                f"{_error_detail}"
                            )
                    except Exception as e:
                        _cf_output = f"ERROR: {str(e)}"
                
                tc_duration = round(time.time() - tc_start, 3)
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=_cf_output,
                    duration_s=tc_duration, child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                
                # Store the compressed findings in memory too
                if not _cf_output.startswith("ERROR:") and not _cf_output.startswith("No research"):
                    _cf_desc = f"compressed findings ({len(_cf_entries)} entries)"
                    if _cf_focus:
                        _cf_desc += f" [focus: {_cf_focus[:30]}]"
                    memory.upsert(
                        key="compressed_findings",
                        tool_name="compress_findings",
                        turn=turn,
                        content=_cf_output,
                        description=_cf_desc,
                    )
                    findings.append(f"[compress_findings] {_cf_output[:1500].strip()}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _cf_output,
                })
                if verbose:
                    print(f"       📦 Compressed {len(_cf_entries)} findings ({tc_duration:.1f}s, {len(_cf_output):,} chars output)")
                continue
            
            # ══════════════════════════════════════════════════════════
            # SUB-AGENT / SHARED TOOLS (all depths)
            # ══════════════════════════════════════════════════════════
            
            # ── Check for final_answer (sub-agents and synthesizers) ──
            if tool_name == "final_answer":
                # Root should use research_complete, not final_answer.
                # If the model hallucinated final_answer at depth 0,
                # redirect it: save content as a draft and nudge toward
                # the proper flow (verifier + spot-check).
                if _depth == 0:
                    _fa_content = tool_args.get("answer", "").strip()
                    if _fa_content and len(_fa_content) >= 20:
                        # Treat as refine_draft — save to draft file
                        draft_versions.append((turn, _fa_content))
                        ver = len(draft_versions)
                        try:
                            with open(_draft_path, "w", encoding="utf-8") as f:
                                f.write(f"<!-- Draft v{ver} | turn {turn} -->\n")
                                f.write(_fa_content)
                        except Exception:
                            pass
                        memory.upsert(
                            key="draft_latest",
                            tool_name="refine_draft",
                            turn=turn,
                            content=_fa_content,
                            description="latest draft answer",
                        )
                        _redirect_msg = (
                            f"✅ Draft v{ver} saved ({len(_fa_content):,} chars).\n\n"
                            "NOTE: You called final_answer, but you should use "
                            "research_complete() to publish. This runs a quality "
                            "review before publishing. Call research_complete() now."
                        )
                    else:
                        _redirect_msg = (
                            "ERROR: final_answer is not available at root level. "
                            "Use refine_draft(content='...') to write your answer, "
                            "then research_complete() to publish it."
                        )
                    tc_record = ToolCallRecord(
                        tool_name=tool_name, tool_args=tool_args,
                        tool_call_id=tool_call["id"], output=_redirect_msg,
                        duration_s=0, child_trace=None,
                    )
                    turn_record.tool_calls.append(tc_record)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": _redirect_msg,
                    })
                    if verbose:
                        print(f"   ↩️  Root called final_answer — redirected to draft v{len(draft_versions)}")
                    continue

                final_content = tool_args.get("answer", "")
                
                # If answer is empty/trivial AND args were malformed AND we have
                # turns left, reject and ask the model to retry instead of
                # silently accepting an empty answer.
                if len(final_content.strip()) < 20 and _args_were_malformed:
                    _has_turns_left = turn_length is None or turn < turn_length
                    if _has_turns_left:
                        _reject_msg = (
                            "ERROR: Your final_answer call had malformed JSON arguments and "
                            "the answer could not be recovered. Please call final_answer again "
                            "with your complete answer as a simple string. Make sure to properly "
                            "escape any quotes or special characters in the answer text."
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": _reject_msg,
                        })
                        tc_record = ToolCallRecord(
                            tool_name=tool_name, tool_args=tool_args,
                            tool_call_id=tool_call["id"], output=_reject_msg,
                            duration_s=0, child_trace=None,
                        )
                        turn_record.tool_calls.append(tc_record)
                        if verbose:
                            print(f"   🔄 Rejected malformed final_answer — asking model to retry")
                        continue
                
                # Backfill: if the model called final_answer with an empty/trivial
                # answer, scan backwards for substantive assistant content
                if len(final_content.strip()) < 20:
                    _NON_ANSWER_PREFIXES = (
                        "i should", "i'll", "i need", "i will", "let me",
                        "okay", "ok,", "ok ", "sure", "now i",
                        "let's", "alright",
                    )
                    _STRUCTURE_SIGNALS = ("|", "#", "- ", "1.", "1)", "{", "**")
                    
                    for msg in reversed(messages):
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
                            if verbose:
                                print(f"   ↩️  Backfilled empty final_answer from prior response ({len(candidate)} chars)")
                            break
                    
                    # If still empty and we have research data, break to synthesis
                    if len(final_content.strip()) < 20 and (findings or memory):
                        if verbose:
                            print(f"   🔄 Empty final_answer with {len(findings)} findings — breaking to synthesis pipeline")
                        tc_record = ToolCallRecord(
                            tool_name=tool_name, tool_args=tool_args,
                            tool_call_id=tool_call["id"],
                            output="[answer lost — routing to synthesis]",
                            duration_s=0, child_trace=None,
                        )
                        turn_record.tool_calls.append(tc_record)
                        _degenerated = True
                        break
                
                tc_record = ToolCallRecord(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_call_id=tool_call["id"],
                    output=final_content,
                    duration_s=0,
                    child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                final_answer_result = final_content
                if verbose:
                    print(f"   ✅ final_answer received")
                continue
            
            # Execute tool (pass _depth so spawn_agent knows its recursion level)
            tc_start = time.time()
            child_trace = None
            try:
                output, child_trace = dispatch_tool_call(tool_name, tool_args, _depth=_depth, model=model, reasoning_effort=reasoning_effort, _sandbox_files=_sandbox_files, _memory_store=memory)
                if verbose and len(output) < 200:
                    print(f"       → {output}")
                elif verbose:
                    print(f"       → {output[:200]}...")
            except Exception as e:
                output = f"ERROR: {str(e)}"
                if verbose:
                    print(f"       → ❌ {output}")
            tc_duration = round(time.time() - tc_start, 3)
            
            # Track consecutive identical errors to break degenerate retry loops
            if output.startswith("ERROR:"):
                error_sig = f"{tool_name}:{output[:200]}"
                if error_sig == last_error_signature:
                    consecutive_error_count += 1
                else:
                    consecutive_error_count = 1
                    last_error_signature = error_sig
                
                if consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
                    output += (
                        f"\n\nFATAL: This same error has occurred {consecutive_error_count} times in a row. "
                        "STOP retrying. Either try a completely different approach, simplify your code, "
                        "or call final_answer with what you have so far."
                    )
                    if verbose:
                        print(f"       ⚠️  Degenerate retry loop detected ({consecutive_error_count}x)")
            else:
                consecutive_error_count = 0
                last_error_signature = None
            
            # ── Consecutive search enforcement (sub-agents only) ──────
            # Root can't search directly (uses conduct_research instead).
            # For sub-agents, after N consecutive search/fetch calls,
            # force a think() call to break the loop.
            if _depth > 0 and tool_name in ("search_web", "fetch_url", "read_pdf", "extract_tables", "fetch_cached", "wikipedia_lookup"):
                _consecutive_search_count += 1
                if _consecutive_search_count > _MAX_CONSECUTIVE_SEARCHES:
                    output = (
                        f"⛔ SEARCH LIMIT: {_consecutive_search_count} consecutive search/fetch calls. "
                        "Output suppressed. You MUST call think() now to reflect on:\n"
                        "  1. What data you have gathered so far\n"
                        "  2. What is still missing\n"
                        "  3. Your plan for the next steps\n"
                        "After think(), you can resume searching."
                    )
                    if verbose:
                        print(f"       ⛔  Consecutive search #{_consecutive_search_count} — blocked, must think()")
                elif _consecutive_search_count == _CONSECUTIVE_SEARCH_WARNING:
                    output += (
                        f"\n\n⚠️ WARNING: {_consecutive_search_count} consecutive search/fetch calls. "
                        "Call think() to reflect on what you have before your next "
                        "search — the next call will be blocked until you do."
                    )
            elif tool_name != "think":
                _consecutive_search_count = 0

            # ── Empty/low-quality result detection (sub-agents) ───────
            # When search returns no results or fetch returns very little,
            # nudge the agent to adapt: refine query, try different source.
            if _depth > 0 and not output.startswith(("ERROR:", "\u26d4")):
                _empty_nudge = ""
                if tool_name == "search_web":
                    # Detect empty search results (no URL lines = no results)
                    _has_urls = bool(re.search(r'URL:\s*https?://', output))
                    if not _has_urls or len(output.strip()) < 100:
                        _empty_nudge = (
                            "\n\n\u26a0\ufe0f EMPTY/POOR RESULTS: This search returned little or no useful data. "
                            "Before retrying, try:\n"
                            "  1. SIMPLIFY your query \u2014 remove quotes, dates, or specific terms\n"
                            "  2. Try SYNONYMS or alternate phrasing\n"
                            "  3. Try a DIFFERENT source type (Wikipedia, government site, etc.)\n"
                            "  4. If the data may not exist publicly, note it as a gap\n"
                            "Do NOT repeat the same query. Call think() to plan a different approach."
                        )
                elif tool_name in ("fetch_url", "extract_tables", "wikipedia_lookup"):
                    # Detect very short / empty page content
                    _content_len = len(output.strip())
                    _is_blocked = any(p in output.lower() for p in [
                        "blocked", "access denied", "403", "captcha",
                        "cloudflare", "do not retry this url",
                    ])
                    if _is_blocked or _content_len < 150:
                        # Tailor advice based on URL type to avoid circular suggestions
                        _tool_url = tool_args.get("url", "")
                        _is_archive = "web.archive.org" in _tool_url
                        _is_jina = "r.jina.ai" in _tool_url or "s.jina.ai" in _tool_url

                        if _is_archive or _is_jina:
                            _empty_nudge = (
                                "\n\n\u26a0\ufe0f BLOCKED/EMPTY: This is already an archived/rendered URL and it failed. "
                                "Do NOT retry it. Instead:\n"
                                "  1. Search for the same information from a COMPLETELY DIFFERENT source\n"
                                "  2. Try wikipedia_lookup() if the topic has a Wikipedia article\n"
                                "  3. Accept the gap and note it in your final_answer"
                            )
                        else:
                            _empty_nudge = (
                                "\n\n\u26a0\ufe0f LOW-QUALITY/BLOCKED CONTENT: This page returned very little usable data. "
                                "Do NOT retry the same URL. Instead:\n"
                                "  1. Search for the same information from a DIFFERENT source\n"
                                "  2. Try fetch_cached() for a Wayback Machine copy\n"
                                "  3. Try wikipedia_lookup() if the topic has a Wikipedia article\n"
                                "  4. Accept the gap and note it in your final_answer"
                            )
                if _empty_nudge:
                    output += _empty_nudge
                    if verbose:
                        print(f"       \u26a0\ufe0f  Empty/low-quality result detected \u2014 nudge injected")
            
            # Record tool call in trace
            tc_record = ToolCallRecord(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call["id"],
                output=output,
                duration_s=tc_duration,
                child_trace=child_trace,
            )
            turn_record.tool_calls.append(tc_record)
            
            # ── Track findings for synthesis fallback ─────────────────
            # Keep a compact log of non-error tool outputs so we can
            # inject them into the synthesis prompt if the model runs out.
            _is_error = output.startswith("ERROR:")
            if not _is_error and tool_name not in ("search_available_tools",):
                snippet = output[:1500].strip()
                if snippet:
                    findings.append(f"[{tool_name}] {snippet}")
                
                # ── Memory store: full-fidelity capture ───────────────
                # For fetch_url: store the FULL parsed page (from cache)
                # rather than the truncated tool output. This enables
                # sub-agents and synthesis to query the complete content
                # programmatically via spawn_agent(memory_keys=[...]).
                desc = ""
                memory_content = output  # default: store tool output as-is
                if tool_name == "search_web":
                    desc = str(tool_args.get("q", ""))[:60]
                elif tool_name == "fetch_url":
                    desc = str(tool_args.get("url", ""))[:60]
                    # Try to get full parsed text from page cache
                    try:
                        from .tool_store import _page_text_cache, _page_text_cache_lock, _URL_CACHE_TTL
                        _fetch_url = str(tool_args.get("url", ""))
                        with _page_text_cache_lock:
                            _ptc = _page_text_cache.get(_fetch_url)
                        if _ptc and (time.time() - _ptc[1] < _URL_CACHE_TTL):
                            memory_content = _ptc[0]  # full parsed text
                    except Exception:
                        pass  # fall back to truncated output
                elif tool_name == "read_pdf":
                    desc = str(tool_args.get("url", ""))[:60]
                elif tool_name == "spawn_agent":
                    desc = str(tool_args.get("task", ""))[:60]
                mem_key = memory.add(
                    tool_name=tool_name,
                    turn=turn,
                    content=memory_content,
                    description=desc,
                )
            else:
                mem_key = None
            
            # ── Research plan tracking (root only) ────────────────────
            if plan is not None and tool_name not in ("search_available_tools",):
                plan.record_tool_call(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output=output,
                    memory_key=mem_key,
                    is_error=_is_error,
                )
            
            # Add tool result to messages (strip base64 blobs to avoid
            # blowing up the context window — the trace keeps the full output)
            msg_output = re.sub(
                r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
                r"\1[file content saved to trace]\2",
                output, flags=re.DOTALL
            )
            
            # ── Symbolic references (root orchestrator only) ──────────
            # Replace large outputs with compact summary + memory key.
            if (
                SYMBOLIC_REFERENCES
                and _depth == 0
                and mem_key is not None
                and len(msg_output) > SYMBOLIC_THRESHOLD
            ):
                msg_output = make_symbolic(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output=msg_output,
                    memory_key=mem_key,
                )
                if verbose:
                    print(f"       📎 Symbolic ref: {len(output):,} → {len(msg_output):,} chars [{mem_key}]")
            
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(msg_output)
            }
            messages.append(tool_message)
        
        turn_record.duration_s = round(time.time() - turn_start, 3)
        episode.turns.append(turn_record)
        
        # ── Auto-inject reflection prompt after search/fetch (sub-agents) ──
        # If the worker just did search/fetch calls without a think, nudge it
        # to pause and assess before the next action. This prevents blind
        # search loops and forces structured reasoning.
        _SEARCH_TOOLS = {"search_web", "fetch_url", "read_pdf", "extract_tables", "fetch_cached", "wikipedia_lookup"}
        if _depth > 0 and not _degenerated and final_answer_result is None:
            _turn_tool_names = [tc.tool_name for tc in turn_record.tool_calls]
            _did_search = any(t in _SEARCH_TOOLS for t in _turn_tool_names)
            _did_think = "think" in _turn_tool_names
            if _did_search and not _did_think:
                # Build a concise summary of what tools returned this turn
                _search_summaries = []
                for tc in turn_record.tool_calls:
                    if tc.tool_name in _SEARCH_TOOLS:
                        _out_preview = (tc.output or "")[:200].strip()
                        if _out_preview:
                            _search_summaries.append(f"  [{tc.tool_name}]: {_out_preview}")
                _summary_block = "\n".join(_search_summaries[:3])  # max 3
                messages.append({
                    "role": "system",
                    "content": (
                        "🔍 REFLECT before your next action. Call think(thought='...') to assess:\n"
                        "  - What key information did you just find?\n"
                        "  - What specific data is still missing?\n"
                        "  - Do you have enough to answer, or should you search more?\n"
                        "  - If searching more: what DIFFERENT query or source should you try?\n"
                        + (f"\nThis turn's results:\n{_summary_block}" if _summary_block else "")
                    ),
                })
                if verbose:
                    print(f"       🔍 Auto-injected reflection prompt")
        
        # If empty final_answer triggered synthesis break, propagate to while loop
        if _degenerated:
            break
        
        # If final_answer was called in this batch, finalize now
        if final_answer_result is not None:
            if verbose:
                print(f"\n📝 Final Response:\n{final_answer_result}")
            return _finalize(final_answer_result)
    
    # ── Post-loop synthesis ────────────────────────────────────────────
    # Two paths lead here:
    #   1. Normal: turn limit reached (context is long but not corrupted)
    #   2. Degeneration: model failed to produce tool calls 3x in a row
    # Root has no turn limit, so only degeneration or context exhaustion
    # brings root here. In either case, the draft IS the answer.
    
    # ── Draft → final answer (root architecture) ─────────────────────
    # Root iteratively refined its draft via refine_draft. If a draft
    # exists, it is the best answer we have — no verification needed
    # because the root already had full context while writing it.
    if draft_versions:
        _draft_turn, latest_draft = draft_versions[-1]
        if len(latest_draft.strip()) >= 50:
            if verbose:
                print(f"\n📝 Using draft v{len(draft_versions)} (turn {_draft_turn}, "
                      f"{len(latest_draft):,} chars) as final answer")
            return _finalize(latest_draft)
    
    if _degenerated:
        if verbose:
            print(f"\n⚠️  Model degenerated — skipping Stage 1, going straight to fresh synthesis sub-agent")
    else:
        if verbose:
            print(f"\n⚠️  Turn limit reached — forcing final_answer synthesis")
    
    # Build a findings summary to inject into the synthesis prompt
    findings_block = ""
    if findings:
        # Take last ~15 findings, cap total to ~3000 chars
        recent = findings[-15:]
        findings_text = "\n".join(f"{i+1}. {f}" for i, f in enumerate(recent))
        if len(findings_text) > 3000:
            findings_text = findings_text[:3000] + "\n... (truncated)"
        findings_block = (
            "\n\nHere are your key research findings so far:\n"
            f"{findings_text}\n"
        )
    
    # Log memory store stats
    if memory and verbose:
        stats = memory.compression_stats()
        print(f"📦 Memory store: {stats['entries']} entries, "
              f"{stats['raw_chars']:,} raw chars → "
              f"{stats['base64_chars']:,} compressed chars "
              f"({stats['savings_pct']}% savings)")
    
    q_echo_synth = user_input[:500] + ("…" if len(user_input) > 500 else "")
    
    # ── Gap check before synthesis (plan_state only) ───────────────────
    # Inject a gap assessment so the model knows what data is strong/weak
    # and can address gaps explicitly in the synthesized answer.
    gap_note = ""
    if plan is not None:
        gap_msg = plan.render_gap_check()
        if gap_msg:
            gap_note = f"\n\n{gap_msg}\n"
            if verbose:
                print(f"📊  Gap check injected before synthesis")

    # ── Stage 1: Forced final_answer in current context ────────────────
    # Skip this entirely if the model degenerated — the corrupted context
    # is what caused the failure, so retrying in it is pointless.
    # NOTE: Always force with final_answer schema here regardless of depth.
    # Root's research_complete (the normal terminal tool) reads from the
    # draft file, but we only reach here when there IS no usable draft,
    # so we need the model to produce an answer directly.
    _FINAL_ANSWER_SCHEMA = next(
        (t for t in TOOLS if t["function"]["name"] == "final_answer"),
        TERMINAL_TOOL,
    )
    if not _degenerated:
        messages.append({
            "role": "user",
            "content": (
                "You have run out of turns. Call final_answer NOW with your best "
                "response based on everything gathered so far.\n\n"
                f"ORIGINAL QUESTION:\n{q_echo_synth}\n"
                f"{findings_block}\n"
                f"{gap_note}"
                "Synthesize these findings into a clear, well-cited answer to the "
                "question above. Address any noted gaps honestly — state what is "
                "well-supported and what remains uncertain."
            ),
        })
        
        # Use proper max_tokens estimation (not the raw context_window)
        approx_input_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        synth_max_tokens = max(context_window - approx_input_tokens - TOKEN_SAFETY_MARGIN, 256)
        synth_max_tokens = min(synth_max_tokens, max_tokens)
    
    # Try synthesis up to 2 times (skipped when degenerated)
    for synth_attempt in range(0 if _degenerated else 2):
        try:
            synth_response = _call_api(synth_max_tokens, tools_override=[_FINAL_ANSWER_SCHEMA])
            if synth_response.status_code == 200:
                synth_result = synth_response.json()
                synth_choices = synth_result.get("choices")
                if synth_choices and isinstance(synth_choices, list) and len(synth_choices) > 0:
                    synth_msg = synth_choices[0].get("message", {})
                    messages.append(synth_msg)
                    # Extract from final_answer tool call
                    final_content = _extract_final_answer(synth_msg)
                    if final_content.strip():
                        # Record the synthesis turn in the trace
                        synth_usage = synth_result.get("usage", {})
                        synth_record = TurnRecord(
                            turn_number=turn + 1 + synth_attempt,
                            assistant_content=final_content,
                            raw_assistant_message=synth_msg,
                            prompt_tokens=synth_usage.get("prompt_tokens", 0),
                            completion_tokens=synth_usage.get("completion_tokens", 0),
                            total_tokens=synth_usage.get("total_tokens", 0),
                        )
                        synth_record.duration_s = 0
                        episode.turns.append(synth_record)
                        if verbose:
                            print(f"✅ Synthesis turn produced response (attempt {synth_attempt + 1})")
                            print(f"\n📝 Final Response:\n{final_content}")
                        return _finalize(final_content)
                    else:
                        if verbose:
                            print(f"⚠️  Synthesis attempt {synth_attempt + 1} returned empty — retrying")
                else:
                    if verbose:
                        print(f"❌ Synthesis attempt {synth_attempt + 1} returned no choices: {str(synth_result)[:200]}")
            else:
                # Handle context overflow on synthesis
                error_msg = str(synth_response.json().get("error", {}).get("message", ""))
                if "max_tokens" in error_msg or "input tokens" in error_msg:
                    match = re.search(r'has (\d+) input tokens', error_msg)
                    if match:
                        input_tokens = int(match.group(1))
                        synth_max_tokens = max(context_window - input_tokens - TOKEN_SAFETY_MARGIN, 256)
                        if verbose:
                            print(f"⚠️  Synthesis context overflow, retrying with {synth_max_tokens} max_tokens")
                        continue
                if verbose:
                    print(f"❌ Synthesis turn API error: {synth_response.status_code}")
        except Exception as e:
            if verbose:
                print(f"❌ Synthesis attempt {synth_attempt + 1} failed: {e}")
    
    # ── Synthesis sub-agent via sandbox file ────────────────────────
    # Upload the full MemoryStore as a JSON file to the sandbox,
    # then spawn a fresh sub-agent whose job is to query the data
    # PROGRAMMATICALLY (via execute_code) and synthesize an answer.
    #
    # Key insight: the research data never enters the LLM context.
    # It lives on the sandbox filesystem. The sub-agent does programmatic
    # search/filter, so it can handle arbitrarily large datasets without
    # context window corruption.
    
    if memory and _sandbox_files is None:
        # Serialize memory → JSON → base64 for sandbox upload
        memory_json = memory.to_json()
        memory_b64 = base64.b64encode(memory_json.encode("utf-8")).decode("ascii")
        
        if verbose:
            stats = memory.compression_stats()
            print(f"📦 Uploading {stats['entries']} memory entries to sandbox "
                  f"({len(memory_json):,} bytes as research_data.json)")
        
        synthesis_task = (
            f"Answer this question using the research data in research_data.json:\n\n"
            f"{user_input}"
        )
        
        sandbox_files = {"research_data.json": memory_b64}
        
        try:
            from .config import SUB_AGENT_TURN_BUDGET
            synth_turn_budget = min(SUB_AGENT_TURN_BUDGET, 10)  # synthesis needs fewer turns
            
            if verbose:
                print(f"🔄 Spawning synthesis sub-agent ({synth_turn_budget} turn budget, "
                      f"research_data.json pre-loaded in sandbox)")
            
            synth_result = dispatch(
                user_input=synthesis_task,
                turn_length=synth_turn_budget,
                verbose=verbose,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                reasoning_effort=reasoning_effort,
                example_id=example_id,
                _depth=_depth + 1,
                _sandbox_files=sandbox_files,
                _is_synthesizer=True,
            )
            
            final_content = synth_result.get("final_response", "")
            
            # Record the synthesis sub-agent as a turn in the parent trace
            if synth_result.get("trace"):
                synth_child_trace = synth_result["trace"]
                synth_record = TurnRecord(
                    turn_number=turn + 3,
                    assistant_content=final_content,
                    raw_assistant_message={"synthesis_sub_agent": True},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )
                # Add a synthetic tool call record for the sub-agent
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
                episode.turns.append(synth_record)
            
            if final_content and final_content.strip() and not final_content.startswith("[Auto-extracted"):
                if verbose:
                    print(f"✅ Synthesis sub-agent succeeded")
                    print(f"\n📝 Final Response:\n{final_content}")
                return _finalize(final_content)
            else:
                if verbose:
                    print(f"⚠️  Synthesis sub-agent returned empty/fallback response — "
                          f"falling through to absolute fallback")
        except Exception as e:
            if verbose:
                print(f"❌ Synthesis sub-agent failed: {e}")
    
    # Absolute fallback — extract useful content from tool results
    final_content = _build_fallback_response(messages)
    
    if verbose:
        if final_content:
            print(f"\n📝 Response (raw fallback from tool results):\n{final_content[:500]}")
        else:
            print(f"\n❌ No content could be extracted from conversation history")
    
    return _finalize(final_content)
