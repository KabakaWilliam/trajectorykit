"""
Main agent loop for the trajectorykit system.
"""

import requests
import json
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any

from .config import MODEL_NAME, VLLM_API_URL, SYSTEM_PROMPT, SUB_AGENT_SYSTEM_PROMPT, TOKEN_SAFETY_MARGIN, MAX_INLINE_CHARS, get_model_profile
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
    
    # Use a lean worker prompt for sub-agents, full orchestrator prompt for root
    system_prompt = SYSTEM_PROMPT if _depth == 0 else SUB_AGENT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # Scratchpad: auto-stores large tool outputs, retrievable via recall() tool
    scratchpad = {}

    def _call_api(effective_max_tokens):
        """Build payload and call the chat completions API."""
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
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
        
        # Dynamically cap max_tokens to fit within the context window
        effective_max_tokens = max_tokens

        # Call vLLM with tools using generation settings
        response = _call_api(effective_max_tokens)
        
        result = response.json()
        
        if response.status_code != 200:
            # If context overflow, try again with dynamically reduced max_tokens
            error_msg = str(result.get("error", {}).get("message", ""))
            if "max_tokens" in error_msg or "max_completion_tokens" in error_msg:
                # Extract input token count from error or estimate from usage
                import re as _re
                match = _re.search(r'has (\d+) input tokens', error_msg)
                if match:
                    input_tokens = int(match.group(1))
                    effective_max_tokens = context_window - input_tokens - TOKEN_SAFETY_MARGIN
                    if effective_max_tokens > 0:
                        if verbose:
                            print(f"⚠️  max_tokens too large, retrying with {effective_max_tokens}")
                        response = _call_api(effective_max_tokens)
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
        
        assistant_message = result["choices"][0]["message"]
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
        
        # Check if model made tool calls
        tool_calls_made = 0
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            tool_calls_made = len(assistant_message["tool_calls"])
            total_tool_calls += tool_calls_made
            
            if verbose:
                print(f"🔧 Tool calls: {tool_calls_made}")
            
            # Process each tool call
            for i, tool_call in enumerate(assistant_message["tool_calls"], 1):
                tool_name = tool_call["function"]["name"]
                raw_args = tool_call["function"].get("arguments", "")
                try:
                    tool_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    tool_args = {}
                    if verbose:
                        print(f"   ⚠️  Malformed tool arguments for {tool_name}, using empty args")
                
                if verbose:
                    print(f"   [{i}] {tool_name}")
                
                # Execute tool (pass _depth so spawn_agent knows its recursion level)
                tc_start = time.time()
                child_trace = None
                try:
                    output, child_trace = dispatch_tool_call(tool_name, tool_args, _depth=_depth, model=model, reasoning_effort=reasoning_effort, scratchpad=scratchpad)
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
                            "or respond with what you have so far."
                        )
                        if verbose:
                            print(f"       ⚠️  Degenerate retry loop detected ({consecutive_error_count}x)")
                else:
                    consecutive_error_count = 0
                    last_error_signature = None
                
                # Record tool call in trace (always stores full output)
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
                
                # Auto-store large outputs to scratchpad to keep context lean
                if len(msg_output) > MAX_INLINE_CHARS:
                    store_key = f"{tool_name}_t{turn}_{i}"
                    scratchpad[store_key] = msg_output
                    preview = msg_output[:200].replace('\n', ' ')
                    msg_output = f"[Stored as '{store_key}'] ({len(scratchpad[store_key])} chars) Preview: {preview}..."
                    if verbose:
                        print(f"       📦 Auto-stored to scratchpad as '{store_key}'")
                
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(msg_output)
                }
                messages.append(tool_message)
        else:
            # Model didn't call tools via structured format.
            # Check if it emitted tool calls as raw text (Qwen sometimes
            # falls back to <tool_call> tags instead of structured calls).
            # Also check the reasoning field — Qwen3 sometimes puts tool calls there.
            raw_content = assistant_message.get("content", "") or ""
            reasoning_content = assistant_message.get("reasoning", "") or assistant_message.get("reasoning_content", "") or ""
            text_with_tool_calls = raw_content if "<tool_call>" in raw_content else (reasoning_content if "<tool_call>" in reasoning_content else "")
            
            if text_with_tool_calls:
                import re as _re
                source = "reasoning" if text_with_tool_calls == reasoning_content else "content"
                # Extract all <tool_call>...</tool_call> blocks
                tc_pattern = _re.compile(
                    r'<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)',
                    _re.DOTALL
                )
                matches = tc_pattern.findall(text_with_tool_calls)
                if matches:
                    if verbose:
                        print(f"⚠️  Detected {len(matches)} text-format tool call(s) in {source} — re-dispatching")
                    for idx, match_str in enumerate(matches, 1):
                        try:
                            tc_obj = json.loads(match_str)
                            tool_name = tc_obj.get("name", "")
                            tool_args = tc_obj.get("arguments", {})
                            if isinstance(tool_args, str):
                                tool_args = json.loads(tool_args)

                            fake_id = f"text_tc_{turn}_{idx}"

                            if verbose:
                                print(f"   [{idx}] {tool_name} (text-format)")

                            tc_start = time.time()
                            child_trace = None
                            try:
                                output, child_trace = dispatch_tool_call(tool_name, tool_args, _depth=_depth, model=model, reasoning_effort=reasoning_effort, scratchpad=scratchpad)
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
                                tool_name=tool_name,
                                tool_args=tool_args,
                                tool_call_id=fake_id,
                                output=output,
                                duration_s=tc_duration,
                                child_trace=child_trace,
                            )
                            turn_record.tool_calls.append(tc_record)
                            total_tool_calls += 1

                            msg_output = re.sub(
                                r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
                                r"\1[file content saved to trace]\2",
                                output, flags=re.DOTALL
                            )
                            
                            # Auto-store large outputs to scratchpad
                            if len(msg_output) > MAX_INLINE_CHARS:
                                store_key = f"{tool_name}_t{turn}_{idx}"
                                scratchpad[store_key] = msg_output
                                preview = msg_output[:200].replace('\n', ' ')
                                msg_output = f"[Stored as '{store_key}'] ({len(scratchpad[store_key])} chars) Preview: {preview}..."
                                if verbose:
                                    print(f"       📦 Auto-stored to scratchpad as '{store_key}'")
                            
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": fake_id,
                                "content": str(msg_output)
                            }
                            messages.append(tool_message)
                        except (json.JSONDecodeError, KeyError) as e:
                            if verbose:
                                print(f"   ⚠️  Failed to parse text tool call: {e}")
                            continue

                    # Continue the loop — don't finalize yet
                    turn_record.duration_s = round(time.time() - turn_start, 3)
                    episode.turns.append(turn_record)
                    continue

            # Model is truly done
            if verbose:
                print(f"✅ Task completed (no more tool calls needed)")
            
            final_content = assistant_message.get("content", "")
            if verbose and final_content:
                print(f"\n📝 Final Response:")
                print(f"{final_content}")
            
            turn_record.duration_s = round(time.time() - turn_start, 3)
            episode.turns.append(turn_record)
            return _finalize(final_content)
        
        turn_record.duration_s = round(time.time() - turn_start, 3)
        episode.turns.append(turn_record)
    
    # Return last assistant message if we hit turn limit
    final_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "content" in msg:
            final_content = msg["content"]
            break
    
    if verbose:
        print(f"\n📝 Response at turn limit:")
        print(f"{final_content}")
    
    return _finalize(final_content)
