"""
Main agent loop for the trajectorykit system.
"""

import requests
import json
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any

from .config import MODEL_NAME, VLLM_API_URL, SYSTEM_PROMPT
from .tool_store import TOOLS, dispatch_tool_call
from .tracing import EpisodeTrace, TurnRecord, ToolCallRecord


def dispatch(
    user_input: str,
    turn_length: Optional[int] = 5,
    verbose: bool = True,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    _depth: int = 0,
) -> Dict[str, Any]:
    """
    Run an agentic loop with iterative tool calling and response refinement.
    
    Args:
        user_input: Initial user message describing the task
        turn_length: Maximum number of turns. None for unlimited (runs until completion)
        verbose: Print detailed turn-by-turn output
        max_tokens: Maximum tokens per generation (default: 2000)
        temperature: Sampling temperature for model (default: 0.7)
        _depth: Current recursion depth (internal — used by spawn_agent)
    
    Returns:
        Dictionary with:
            - 'final_response': The model's final answer
            - 'turns': Number of turns completed
            - 'tool_calls': Total number of tool calls made
            - 'messages': Full conversation history
            - 'trace': EpisodeTrace capturing the full execution tree
    """

    # Initialize trace
    episode = EpisodeTrace(
        depth=_depth,
        user_input=user_input,
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        turn_length=turn_length,
        started_at=datetime.now().isoformat(),
    )
    episode_start = time.time()
    
    messages = [
        {"role": "SYSTEM", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    turn = 0
    total_tool_calls = 0
    
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
        
        # Call vLLM with tools using generation settings
        response = requests.post(
            f"{VLLM_API_URL}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        result = response.json()
        
        if response.status_code != 200:
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
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                if verbose:
                    print(f"   [{i}] {tool_name}")
                
                # Execute tool (pass _depth so spawn_agent knows its recursion level)
                tc_start = time.time()
                child_trace = None
                try:
                    output, child_trace = dispatch_tool_call(tool_name, tool_args, _depth=_depth)
                    if verbose and len(output) < 200:
                        print(f"       → {output}")
                    elif verbose:
                        print(f"       → {output[:200]}...")
                except Exception as e:
                    output = f"ERROR: {str(e)}"
                    if verbose:
                        print(f"       → ❌ {output}")
                tc_duration = round(time.time() - tc_start, 3)
                
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
        else:
            # Model didn't call tools - it's done
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
