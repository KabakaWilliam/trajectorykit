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

from .config import MODEL_NAME, VLLM_API_URL, SYSTEM_PROMPT, WORKER_PROMPT, SYNTHESIZER_PROMPT, TOKEN_SAFETY_MARGIN, get_model_profile
from .memory import MemoryStore
from .tool_store import TOOLS, dispatch_tool_call
from .tracing import EpisodeTrace, TurnRecord, ToolCallRecord


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
    # Root agent (depth 0): orchestrator prompt, full tool list
    # Sub-agents (depth >= 1): worker prompt, spawn_agent hidden
    # Synthesis sub-agent (_sandbox_files set): synthesizer prompt, code + final_answer only
    if _sandbox_files:
        system_prompt = SYNTHESIZER_PROMPT
        available_tools = [t for t in TOOLS if t["function"]["name"] in ("execute_code", "final_answer", "search_available_tools")]
    elif _depth == 0:
        system_prompt = SYSTEM_PROMPT
        available_tools = TOOLS
    else:
        system_prompt = WORKER_PROMPT
        available_tools = [t for t in TOOLS if t["function"]["name"] != "spawn_agent"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # Schema for the final_answer tool only — used when forcing termination
    FINAL_ANSWER_TOOL = next(t for t in available_tools if t["function"]["name"] == "final_answer")

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
    
    # ── Memory store ──────────────────────────────────────────────────
    # Full-fidelity storage of tool outputs. Compressed at synthesis time
    # and passed to the synthesis sub-agent for programmatic querying.
    memory = MemoryStore()
    
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

        # ── Periodic question reminder (every 8 turns) ────────────────────
        # During long horizons the model can lose sight of the original question.
        _REMINDER_INTERVAL = 8
        if turn > 1 and turn % _REMINDER_INTERVAL == 0:
            q_snippet = user_input[:300] + ("…" if len(user_input) > 300 else "")
            messages.append({"role": "system", "content": (
                f"🔎 REMINDER — You are answering this question:\n"
                f"{q_snippet}\n\n"
                "Stay focused on gathering information relevant to this question. "
                "If you already have enough, call final_answer."
            )})
            if verbose:
                print(f"🔎  Injected periodic question reminder (turn {turn})")

        # ── Budget-aware tool restriction ─────────────────────────────────
        # Progressive warnings at multiple checkpoints, then force final_answer.
        q_echo = user_input[:500] + ("…" if len(user_input) > 500 else "")
        tools_for_turn = None  # None = use full TOOLS list
        if turn_length is not None:
            remaining = turn_length - turn
            if remaining == 4:
                # Early warning
                messages.append({"role": "system", "content": (
                    "📋 BUDGET CHECK: You have 5 turns remaining (including this one). "
                    "Start consolidating your findings toward answering this question:\n"
                    f"{q_echo}\n\n"
                    "If you have enough information, consider calling final_answer soon."
                )})
                if verbose:
                    print(f"📋  Injected budget check (5 turns left)")
            elif remaining == 2:
                # Urgent warning
                messages.append({"role": "system", "content": (
                    "⚠️ BUDGET WARNING: You have 3 turns remaining (including this one). "
                    "Synthesize what you have to answer this question:\n"
                    f"{q_echo}\n\n"
                    "Call final_answer NOW. Do not start new research."
                )})
                if verbose:
                    print(f"⚠️  Injected budget warning (3 turns left)")
            elif remaining == 1:
                # Last chance
                messages.append({"role": "system", "content": (
                    "🚨 LAST CHANCE: You have 2 turns remaining (including this one). "
                    "On your next turn you will be FORCED to call final_answer. "
                    "Finish any last tool call NOW, then call final_answer for:\n"
                    f"{q_echo}"
                )})
                if verbose:
                    print(f"🚨  Injected last chance warning (2 turns left)")
            elif remaining == 0:
                # Last turn: only final_answer is available
                tools_for_turn = [FINAL_ANSWER_TOOL]
                if verbose:
                    print(f"🔒 Restricting tools to final_answer only (last turn)")

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
                        "Use the exact tool names: search_web, fetch_url, execute_code, "
                        "spawn_agent, read_pdf, or final_answer."
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
        # Process them, and if any call is final_answer → terminate.
        tool_calls_in_msg = assistant_message.get("tool_calls") or []
        if not tool_calls_in_msg:
            # Shouldn't happen with tool_choice="required", but vLLM's tool-call
            # parser sometimes fails to extract calls from the model output,
            # returning empty tool_calls with content text instead.
            # Nudge the model to retry with a proper tool call.
            # Only give up and finalize after repeated failures.
            consecutive_no_tool_count += 1
            
            final_content = assistant_message.get("content", "") or ""
            
            if consecutive_no_tool_count >= 3:
                # Model keeps refusing to use tools — accept whatever it gave us
                if verbose:
                    print(f"⚠️  Model produced text without tool calls {consecutive_no_tool_count}x in a row — accepting as final answer")
                    if final_content.strip():
                        print(f"\n📝 Final Response:\n{final_content[:300]}")
                turn_record.duration_s = round(time.time() - turn_start, 3)
                episode.turns.append(turn_record)
                return _finalize(final_content)
            else:
                # Nudge model to call a tool — include its own content so it can
                # wrap it in final_answer without losing it.
                if verbose:
                    print(f"⚠️  No tool calls returned (attempt {consecutive_no_tool_count}/3) — nudging model to use tools")
                # Truncate very long content to avoid blowing up context
                content_echo = final_content[:3000] if final_content.strip() else ""
                if content_echo:
                    nudge = (
                        "Your previous response contained your answer as plain text but "
                        "without a tool call. You MUST call a tool.\n\n"
                        "If the response below IS your final answer, call `final_answer` now "
                        "and COPY YOUR COMPLETE ANSWER into the `answer` parameter — do NOT "
                        "leave it empty.\n\n"
                        "If you still need to research, call a research tool instead.\n\n"
                        f"YOUR PREVIOUS RESPONSE (for reference):\n{content_echo}"
                    )
                else:
                    nudge = (
                        "Your response was empty with no tool calls. "
                        "Either call a research tool (search_web, execute_code, fetch_url) "
                        "to continue working, or call `final_answer` with your complete answer."
                        "If you're unsure what tools are available or how to use them, "
                        "call `search_available_tools` to see all options and their parameters."
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
            try:
                tool_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                tool_args = {}
                if verbose:
                    print(f"   ⚠️  Malformed tool arguments for {tool_name}, using empty args")
            
            if verbose:
                print(f"   [{i}] {tool_name}")
            
            # ── Check for final_answer ────────────────────────────────
            if tool_name == "final_answer":
                final_content = tool_args.get("answer", "")
                
                # Backfill: if the model called final_answer with an empty/trivial
                # answer, it likely already wrote its real answer as plain text in
                # a previous turn. Scan backwards for the most recent substantive
                # assistant content.
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
                        # Skip meta-commentary
                        lower = candidate[:40].lower()
                        if any(lower.startswith(p) for p in _NON_ANSWER_PREFIXES):
                            continue
                        # Prefer structured content (tables, headers, lists, JSON)
                        has_structure = any(s in candidate[:500] for s in _STRUCTURE_SIGNALS)
                        if has_structure or len(candidate) > 200:
                            final_content = candidate
                            if verbose:
                                print(f"   ↩️  Backfilled empty final_answer from prior response ({len(candidate)} chars)")
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
                # Don't break — record remaining tool calls in this batch
                # but we'll finalize after the loop
                continue
            
            # Execute tool (pass _depth so spawn_agent knows its recursion level)
            tc_start = time.time()
            child_trace = None
            try:
                output, child_trace = dispatch_tool_call(tool_name, tool_args, _depth=_depth, model=model, reasoning_effort=reasoning_effort, _sandbox_files=_sandbox_files)
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
            if not output.startswith("ERROR:") and tool_name not in ("search_available_tools", "get_current_time", "add_numbers"):
                snippet = output[:1500].strip()
                if snippet:
                    findings.append(f"[{tool_name}] {snippet}")
                
                # ── Memory store: full-fidelity capture ───────────────
                # Store the FULL output (not truncated) for zero-loss synthesis.
                # Description is auto-extracted from the tool args when available.
                desc = ""
                if tool_name == "search_web":
                    desc = str(tool_args.get("q", ""))[:60]
                elif tool_name == "fetch_url":
                    desc = str(tool_args.get("url", ""))[:60]
                elif tool_name == "read_pdf":
                    desc = str(tool_args.get("url", ""))[:60]
                elif tool_name == "spawn_agent":
                    desc = str(tool_args.get("task", ""))[:60]
                memory.add(
                    tool_name=tool_name,
                    turn=turn,
                    content=output,
                    description=desc,
                )
            
            # Add tool result to messages (strip base64 blobs to avoid
            # blowing up the context window — the trace keeps the full output)
            msg_output = re.sub(
                r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
                r"\1[file content saved to trace]\2",
                output, flags=re.DOTALL
            )
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(msg_output)
            }
            messages.append(tool_message)
        
        turn_record.duration_s = round(time.time() - turn_start, 3)
        episode.turns.append(turn_record)
        
        # If final_answer was called in this batch, finalize now
        if final_answer_result is not None:
            if verbose:
                print(f"\n📝 Final Response:\n{final_answer_result}")
            return _finalize(final_answer_result)
    
    # ── Turn limit reached — force one last final_answer call ─────────
    # We should rarely get here because the last turn already forces
    # final_answer via tool_choice. But as a safety net:
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
    messages.append({
        "role": "user",
        "content": (
            "You have run out of turns. Call final_answer NOW with your best "
            "response based on everything gathered so far.\n\n"
            f"ORIGINAL QUESTION:\n{q_echo_synth}\n"
            f"{findings_block}\n"
            "Synthesize these findings into a clear, well-cited answer to the "
            "question above."
        ),
    })
    
    # Use proper max_tokens estimation (not the raw context_window)
    approx_input_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
    synth_max_tokens = max(context_window - approx_input_tokens - TOKEN_SAFETY_MARGIN, 256)
    synth_max_tokens = min(synth_max_tokens, max_tokens)
    
    # Try synthesis up to 2 times
    for synth_attempt in range(2):
        try:
            synth_response = _call_api(synth_max_tokens, tools_override=[FINAL_ANSWER_TOOL])
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
