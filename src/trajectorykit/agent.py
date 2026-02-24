"""
Main agent loop for the trajectorykit system.
"""

import requests
import json
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any

from .config import MODEL_NAME, VLLM_API_URL, SYSTEM_PROMPT, WORKER_PROMPT, TOKEN_SAFETY_MARGIN, get_model_profile
from .tool_store import TOOLS, dispatch_tool_call
from .tracing import EpisodeTrace, TurnRecord, ToolCallRecord


def dispatch(
    user_input: str,
    turn_length: Optional[int] = 5,
    verbose: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    _depth: int = 0,
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
    if _depth == 0:
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
        return requests.post(f"{VLLM_API_URL}/chat/completions", json=payload)
    
    turn = 0
    total_tool_calls = 0
    consecutive_error_count = 0
    last_error_signature = None
    MAX_CONSECUTIVE_ERRORS = 3  # Break retry loops after 3 identical failures
    
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

        # ── Budget-aware tool restriction ─────────────────────────────────
        # On the last turn, restrict tools to ONLY final_answer so the model
        # must produce a final response. On second-to-last, inject a warning.
        tools_for_turn = None  # None = use full TOOLS list
        if turn_length is not None:
            remaining = turn_length - turn
            if remaining == 1:
                # Second-to-last turn: warn the model
                messages.append({"role": "system", "content": (
                    "⚠️ BUDGET WARNING: You have 2 turns remaining (including this one). "
                    "Wrap up your work. On your next turn you will be forced to call "
                    "final_answer, so finish any last tool calls NOW."
                )})
                if verbose:
                    print(f"⚠️  Injected budget warning (2 turns left)")
            elif remaining == 0:
                # Last turn: only final_answer is available
                tools_for_turn = [FINAL_ANSWER_TOOL]
                if verbose:
                    print(f"🔒 Restricting tools to final_answer only (last turn)")

        # Call vLLM with tools
        response = _call_api(effective_max_tokens, tools_override=tools_for_turn)
        
        result = response.json()
        
        if response.status_code != 200:
            # If context overflow, try again with dynamically reduced max_tokens
            error_msg = str(result.get("error", {}).get("message", ""))
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
            else:
                if verbose:
                    print(f"❌ API Error: {result}")
                return _finalize(f"Error: {result}")
        
        # Guard against malformed API responses (200 but missing expected fields)
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
        messages.append(assistant_message)
        
        # Start building turn record
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
            # Shouldn't happen with tool_choice="required", but handle gracefully.
            # Treat any content as a final answer.
            final_content = assistant_message.get("content", "") or ""
            if final_content.strip():
                if verbose:
                    print(f"✅ Model produced text without tool calls (unexpected with required)")
                    print(f"\n📝 Final Response:\n{final_content}")
                turn_record.duration_s = round(time.time() - turn_start, 3)
                episode.turns.append(turn_record)
                return _finalize(final_content)
            else:
                # Empty response with no tool calls — nudge and continue
                if verbose:
                    print(f"⚠️  Empty response with no tool calls — nudging model")
                messages.append({
                    "role": "user",
                    "content": "Your response was empty. Call the `search_available_tools` tool to get all the tools you need or call `final_answer` tool with your response."
                })
                turn_record.duration_s = round(time.time() - turn_start, 3)
                episode.turns.append(turn_record)
                continue

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
                output, child_trace = dispatch_tool_call(tool_name, tool_args, _depth=_depth, model=model, reasoning_effort=reasoning_effort)
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
    messages.append({
        "role": "user",
        "content": (
            "You have run out of turns. Call final_answer NOW with your best "
            "response based on everything gathered so far."
        ),
    })
    try:
        effective_max_tokens = max_tokens
        synth_response = _call_api(effective_max_tokens, tools_override=[FINAL_ANSWER_TOOL])
        if synth_response.status_code == 200:
            synth_result = synth_response.json()
            synth_choices = synth_result.get("choices")
            if synth_choices and isinstance(synth_choices, list) and len(synth_choices) > 0:
                synth_msg = synth_choices[0].get("message", {})
                messages.append(synth_msg)
                # Extract from final_answer tool call
                synth_tc = (synth_msg.get("tool_calls") or [{}])[0]
                synth_args_raw = synth_tc.get("function", {}).get("arguments", "{}")
                try:
                    synth_args = json.loads(synth_args_raw) if synth_args_raw else {}
                except json.JSONDecodeError:
                    synth_args = {}
                final_content = synth_args.get("answer", synth_msg.get("content", "") or "")
                # Record the synthesis turn in the trace
                synth_usage = synth_result.get("usage", {})
                synth_record = TurnRecord(
                    turn_number=turn + 1,
                    assistant_content=final_content,
                    raw_assistant_message=synth_msg,
                    prompt_tokens=synth_usage.get("prompt_tokens", 0),
                    completion_tokens=synth_usage.get("completion_tokens", 0),
                    total_tokens=synth_usage.get("total_tokens", 0),
                )
                synth_record.duration_s = 0
                episode.turns.append(synth_record)
                if verbose:
                    print(f"✅ Synthesis turn produced response")
                    print(f"\n📝 Final Response:\n{final_content}")
                return _finalize(final_content)
            else:
                if verbose:
                    print(f"❌ Synthesis turn returned no choices: {str(synth_result)[:200]}")
        else:
            if verbose:
                print(f"❌ Synthesis turn API error: {synth_response.status_code}")
    except Exception as e:
        if verbose:
            print(f"❌ Synthesis turn failed: {e}")
    
    # Absolute fallback — grab whatever we can from message history
    final_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            if content.strip():
                final_content = content
                break
    
    if verbose:
        print(f"\n📝 Response (fallback):\n{final_content}")
    
    return _finalize(final_content)
