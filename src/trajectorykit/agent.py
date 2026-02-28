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
from .memory import MemoryStore, extract_summary, build_memory_index
from .tool_store import TOOLS, EXECUTE_CODE_TOOL, dispatch_tool_call
from .tracing import EpisodeTrace, TurnRecord, ToolCallRecord
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
logger = logging.getLogger(__name__)


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
    _shell=None,
    _progress_fn=None,
    _wall_clock_limit: Optional[float] = None,
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
        _progress_fn: Optional callback(msg: str) for reporting progress to parent.
                      Used by sub-agents so the orchestrator can print live updates.
        _wall_clock_limit: Optional wall-clock timeout in seconds. When exceeded,
                           the agent loop exits gracefully. Used for sub-agents (600s).
    
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
    
    # ── Persistent sandbox shell ──────────────────────────────────────────
    # Root dispatch creates a SandboxShell; sub-agents inherit it via _shell.
    # Synthesis sub-agents (with _sandbox_files) use sandbox_shell (or execute_code fallback).
    owns_shell = False          # True only if *this* dispatch created the shell
    shell = _shell              # May be inherited from parent
    if shell is None and _depth == 0 and not _sandbox_files:
        try:
            from .sandbox_shell import SandboxShell
            shell = SandboxShell(timeout=30, network=True)
            owns_shell = True
            if verbose:
                print(f"\U0001f4e6 Sandbox shell session started: {shell.session_id}")
        except Exception as e:
            logger.warning(f"Could not start sandbox shell: {e}")
            # Agent will still work — sandbox_shell calls will return an error

    # ── Depth-aware prompt & tool selection ──────────────────────────────
    # Root agent (depth 0): orchestrator prompt, full tool list (sandbox_shell)
    # Sub-agents (depth >= 1): worker prompt, spawn_agent hidden
    # Synthesis sub-agent (_sandbox_files set): synthesizer prompt, sandbox_shell + final_answer
    #   Falls back to legacy execute_code if no shell is available.
    if _sandbox_files:
        system_prompt = SYNTHESIZER_PROMPT
        if shell is not None:
            available_tools = [t for t in TOOLS if t["function"]["name"] in ("sandbox_shell", "final_answer", "search_available_tools")]
        else:
            available_tools = [EXECUTE_CODE_TOOL] + [t for t in TOOLS if t["function"]["name"] in ("final_answer", "search_available_tools")]
    elif _depth == 0:
        system_prompt = SYSTEM_PROMPT
        available_tools = TOOLS
    else:
        # Sub-agents: no spawn_agent, no planning tools, no store_memory
        # They keep recall (read-only) so they can retrieve truncated outputs
        _ORCHESTRATOR_ONLY = {"spawn_agent", "create_plan", "update_plan", "store_memory"}
        system_prompt = WORKER_PROMPT
        available_tools = [t for t in TOOLS if t["function"]["name"] not in _ORCHESTRATOR_ONLY]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # Schema for the final_answer tool only — used when forcing termination
    FINAL_ANSWER_TOOL = next(t for t in available_tools if t["function"]["name"] == "final_answer")

    # Schema for create_plan only — used during free planning turn
    CREATE_PLAN_TOOL = next((t for t in available_tools if t["function"]["name"] == "create_plan"), None)

    # ── Plan state ────────────────────────────────────────────────────
    # Structured plan object with goal + subtasks (each with status + result).
    # Set by the free planning turn (create_plan) and updated via update_plan
    # or auto-marked by spawn_agent when subtask_id is provided.
    current_plan: Optional[dict] = None  # {"goal": str, "subtasks": [{"id": int, "task": str, "status": str, "result": str|None}]}
    plan_history: list[dict] = []  # [{"turn": int, "action": str, "detail": str}]

    _STATUS_ICONS = {
        "not_started": "⬜", "in_progress": "🔄",
        "done": "✅", "failed": "❌", "skipped": "⏭️",
    }

    def _render_plan() -> str:
        """Render the structured plan as a readable checklist string."""
        if current_plan is None:
            return ""
        lines = [f"Goal: {current_plan['goal']}"]
        counts = {"done": 0, "in_progress": 0, "not_started": 0, "failed": 0, "skipped": 0}
        for st in current_plan["subtasks"]:
            icon = _STATUS_ICONS.get(st["status"], "⬜")
            counts[st["status"]] = counts.get(st["status"], 0) + 1
            line = f"  {st['id']}. {icon} {st['task']}"
            if st.get("result"):
                line += f"\n     → {st['result']}"
            lines.append(line)
        total = len(current_plan["subtasks"])
        done = counts["done"]
        prog = counts["in_progress"]
        fail = counts["failed"]
        lines.append(f"\nProgress: {done}/{total} done" + (f", {prog} in progress" if prog else "") + (f", {fail} failed" if fail else ""))
        return "\n".join(lines)

    def _call_api(effective_max_tokens, tools_override=None):
        """Build payload and call the chat completions API.
        
        Args:
            effective_max_tokens: Max tokens for this call.
            tools_override: If provided, use this list of tool schemas instead
                            of the full TOOLS list. Used to restrict the model
                            to only final_answer on the last turn.
        """
        # Filter out create_plan from regular turns (only available during planning turn)
        effective_tools = tools_override if tools_override is not None else available_tools
        if tools_override is None and current_plan is not None:
            # After plan is created, remove create_plan from available tools
            effective_tools = [t for t in available_tools if t["function"]["name"] != "create_plan"]
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": effective_tools,
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
        return requests.post(f"{VLLM_API_URL}/chat/completions", json=payload, timeout=300)
    
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
        # Close the persistent sandbox shell only if this dispatch created it
        if shell is not None and owns_shell:
            try:
                shell.close()
                if verbose:
                    print(f"\U0001f4e6 Sandbox shell session closed: {shell.session_id}")
            except Exception:
                pass
        episode.final_response = final_content
        episode.total_turns = turn
        episode.total_tool_calls = total_tool_calls
        episode.ended_at = datetime.now().isoformat()
        episode.duration_s = round(time.time() - episode_start, 3)
        # Store plan history in the trace for observability
        if plan_history:
            episode.plan_history = plan_history
        episode.compute_recursive_stats()
        return {
            'final_response': final_content,
            'turns': turn,
            'tool_calls': total_tool_calls,
            'messages': messages,
            'trace': episode,
        }
    
    # ── Free planning turn ────────────────────────────────────────────
    # Before the main loop, give the model a free turn to create a plan.
    # This doesn't count against the turn budget.
    # Only for the root orchestrator (depth 0) — sub-agents skip planning.
    if CREATE_PLAN_TOOL is not None and _depth == 0:
        if verbose:
            print(f"\n{'─'*70}")
            print(f"PLANNING TURN (free — does not count against budget)")
            print(f"{'─'*70}")
        
        # Inject a planning instruction
        messages.append({"role": "system", "content": (
            "Before you begin research, create a plan. Call create_plan with:\n"
            "- goal: A clear statement of what you're answering\n"
            "- subtasks: A list of specific, actionable subtasks\n\n"
            "Example: create_plan(goal='Find population growth rate of Lagos 2010-2023', "
            "subtasks=['Find 2010 census data', 'Find 2023 estimate', 'Calculate CAGR', 'Cross-verify with UN data'])\n\n"
            "The plan is shown as a checklist on every turn. Subtasks are auto-marked "
            "when you use spawn_agent(subtask_id=N)."
        )})
        
        # Call API with only create_plan + final_answer (for trivial questions)
        planning_tools = [CREATE_PLAN_TOOL, FINAL_ANSWER_TOOL]
        approx_input_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        planning_max_tokens = max(context_window - approx_input_tokens - TOKEN_SAFETY_MARGIN, 256)
        planning_max_tokens = min(planning_max_tokens, max_tokens)
        max_completion = profile.get("max_completion_tokens")
        if max_completion:
            planning_max_tokens = min(planning_max_tokens, max_completion)
        
        planning_payload = {
            "model": model,
            "messages": messages,
            "tools": planning_tools,
            "tool_choice": "required",
            "temperature": temperature,
            "max_tokens": planning_max_tokens,
        }
        if reasoning_effort and profile.get("supports_reasoning_effort"):
            planning_payload["reasoning_effort"] = reasoning_effort
        chat_template_kwargs = profile.get("chat_template_kwargs")
        if chat_template_kwargs:
            planning_payload["chat_template_kwargs"] = chat_template_kwargs
        
        try:
            if verbose:
                print(f"⏳ Calling vLLM (planning)...")
            planning_response = requests.post(f"{VLLM_API_URL}/chat/completions", json=planning_payload, timeout=300)
            if planning_response.status_code == 200:
                planning_result = planning_response.json()
                planning_choices = planning_result.get("choices", [])
                if planning_choices:
                    planning_msg = planning_choices[0].get("message", {})
                    
                    # Clean and append assistant message
                    clean_planning_msg = {
                        "role": planning_msg["role"],
                        "content": planning_msg.get("content") or "",
                    }
                    if planning_msg.get("tool_calls"):
                        clean_planning_msg["tool_calls"] = planning_msg["tool_calls"]
                    messages.append(clean_planning_msg)
                    
                    # Record planning turn in trace
                    planning_usage = planning_result.get("usage", {})
                    planning_record = TurnRecord(
                        turn_number=0,  # planning turn is turn 0
                        assistant_content=planning_msg.get("content"),
                        raw_assistant_message=planning_msg,
                        prompt_tokens=planning_usage.get("prompt_tokens", 0),
                        completion_tokens=planning_usage.get("completion_tokens", 0),
                        total_tokens=planning_usage.get("total_tokens", 0),
                    )
                    planning_start = time.time()
                    
                    # Process tool calls from planning turn
                    planning_tool_calls = planning_msg.get("tool_calls") or []
                    for ptc in planning_tool_calls:
                        ptc_name = ptc["function"]["name"]
                        if "<|" in ptc_name:
                            ptc_name = ptc_name.split("<|")[0]
                        raw_args = ptc["function"].get("arguments", "")
                        try:
                            ptc_args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            ptc_args = {}
                        
                        if ptc_name == "create_plan":
                            # Parse structured plan: goal + subtasks list
                            goal = ptc_args.get("goal", "")
                            subtasks_raw = ptc_args.get("subtasks", [])
                            # Fallback: if model used old format (single "plan" string), parse it
                            if not goal and not subtasks_raw and ptc_args.get("plan"):
                                plan_text = ptc_args["plan"]
                                lines = [l.strip() for l in plan_text.split("\n") if l.strip()]
                                goal = lines[0] if lines else "Research"
                                subtasks_raw = lines[1:] if len(lines) > 1 else [goal]
                                # Strip leading numbering like "1. " or "- "
                                import re as _re
                                subtasks_raw = [_re.sub(r"^[\d]+[\.\)]\s*", "", s).strip() for s in subtasks_raw]
                                subtasks_raw = [_re.sub(r"^[-*]\s*", "", s).strip() for s in subtasks_raw]
                            if isinstance(subtasks_raw, str):
                                subtasks_raw = [s.strip() for s in subtasks_raw.split("\n") if s.strip()]
                            
                            current_plan = {
                                "goal": goal or "Research",
                                "subtasks": [
                                    {"id": i + 1, "task": t, "status": "not_started", "result": None}
                                    for i, t in enumerate(subtasks_raw) if t
                                ]
                            }
                            plan_history.append({"turn": 0, "action": "create", "detail": goal})
                            
                            rendered = _render_plan()
                            confirmation = (
                                f"Plan created and stored. It will be shown as a checklist on every turn.\n\n"
                                f"{rendered}\n\n"
                                "Tip: Use spawn_agent(task='...', subtask_id=N) to auto-mark subtasks. "
                                "Use update_plan for strategic changes (failures, pivots, new subtasks)."
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": ptc["id"],
                                "content": confirmation
                            })
                            
                            tc_record = ToolCallRecord(
                                tool_name="create_plan",
                                tool_args=ptc_args,
                                tool_call_id=ptc["id"],
                                output=confirmation,
                                duration_s=0,
                                child_trace=None,
                            )
                            planning_record.tool_calls.append(tc_record)
                            
                            if verbose:
                                print(f"📋 Plan created:")
                                for line in rendered.split("\n"):
                                    print(f"   {line}")
                        
                        elif ptc_name == "final_answer":
                            # Trivial question — model chose to answer directly
                            final_content = ptc_args.get("answer", "")
                            tc_record = ToolCallRecord(
                                tool_name="final_answer",
                                tool_args=ptc_args,
                                tool_call_id=ptc["id"],
                                output=final_content,
                                duration_s=0,
                                child_trace=None,
                            )
                            planning_record.tool_calls.append(tc_record)
                            planning_record.duration_s = round(time.time() - planning_start, 3)
                            episode.turns.append(planning_record)
                            if verbose:
                                print(f"   ✅ Trivial question — answering directly")
                                print(f"\n📝 Final Response:\n{final_content}")
                            return _finalize(final_content)
                    
                    planning_record.duration_s = round(time.time() - planning_start, 3)
                    episode.turns.append(planning_record)
                    total_tool_calls += len(planning_tool_calls)
            else:
                if verbose:
                    print(f"⚠️  Planning turn failed (HTTP {planning_response.status_code}) — proceeding without plan")
        except Exception as e:
            if verbose:
                print(f"⚠️  Planning turn failed ({e}) — proceeding without plan")
    
    while True:
        # Check turn limit
        if turn_length is not None and turn >= turn_length:
            if verbose:
                print(f"\n⏹️  Reached maximum turns ({turn_length})")
            break
        
        # Check wall-clock timeout (for sub-agents)
        if _wall_clock_limit and (time.time() - episode_start) > _wall_clock_limit:
            if _progress_fn:
                _progress_fn(f"wall-clock timeout ({_wall_clock_limit:.0f}s)")
            if verbose:
                print(f"\n⏱️  Wall-clock timeout ({_wall_clock_limit:.0f}s) — forcing completion")
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

        # ── Plan injection (every turn) ──────────────────────────────────
        # If there's an active plan, inject it as a system message so the
        # model always has its goal and progress visible.
        if current_plan and turn > 1:
            _PLAN_INJECT_INTERVAL = 3  # inject plan every N turns to save tokens
            if turn % _PLAN_INJECT_INTERVAL == 1 or turn <= 3:
                rendered = _render_plan()
                q_snippet = user_input[:300] + ("…" if len(user_input) > 300 else "")
                messages.append({"role": "system", "content": (
                    f"📋 CURRENT PLAN (turn {turn}"
                    + (f"/{turn_length}" if turn_length else "")
                    + f"):\n{rendered}\n\n"
                    f"🔎 QUESTION: {q_snippet}\n\n"
                    "Use spawn_agent(task='...', subtask_id=N) to work on ⬜ subtasks. "
                    "If you have enough info, call final_answer."
                )})
                if verbose:
                    print(f"📋  Injected plan reminder (turn {turn})")
        elif turn > 1 and turn % 8 == 0:
            # Fallback: if no plan was created, use the old question-only reminder
            q_snippet = user_input[:300] + ("…" if len(user_input) > 300 else "")
            messages.append({"role": "system", "content": (
                f"🔎 REMINDER — You are answering this question:\n"
                f"{q_snippet}\n\n"
                "Stay focused on gathering information relevant to this question. "
                "If you already have enough, call final_answer."
            )})
            if verbose:
                print(f"🔎  Injected periodic question reminder (turn {turn})")

        # ── Memory index injection (sub-agents) ──────────────────────────
        # Every 4 turns, remind the agent what's stored in memory so it
        # knows it can use recall() to retrieve full content.
        _MEMORY_INJECT_INTERVAL = 4
        if (
            _depth > 0
            and turn > 1
            and turn % _MEMORY_INJECT_INTERVAL == 0
            and memory.entries
        ):
            mem_index = build_memory_index(memory.entries)
            messages.append({"role": "system", "content": (
                f"{mem_index}\n\n"
                "⚠️ The summaries above are TRUNCATED. To get the full content, "
                "you MUST call the recall tool: recall(key=\"<key>\").\n"
                "recall is a real tool in your toolbox — issue it as a tool call, "
                "do NOT just describe or discuss it in your response.\n"
                "For large pages stored on disk, prefer sandbox_shell with grep/head "
                "to extract only what you need (see the 📦 hints after each fetch)."
            )})
            if verbose:
                print(f"📦  Injected memory index ({len(memory.entries)} entries, turn {turn})")

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
        if verbose:
            print(f"⏳ Calling vLLM...")
        if _progress_fn:
            _progress_fn(f"turn {turn}: thinking...")
        try:
            response = _call_api(effective_max_tokens, tools_override=tools_for_turn)
        except requests.exceptions.Timeout:
            if verbose:
                print(f"⏱️  vLLM request timed out (>300s) — retrying once")
            try:
                response = _call_api(effective_max_tokens, tools_override=tools_for_turn)
            except requests.exceptions.Timeout:
                if verbose:
                    print(f"⏱️  vLLM request timed out again — skipping turn")
                messages.append({"role": "user", "content": (
                    "Your previous request timed out. Try a simpler approach, "
                    "or call final_answer with what you have so far."
                )})
                turn_record = TurnRecord(
                    turn_number=turn,
                    assistant_content=None,
                    raw_assistant_message={"error": "vLLM timeout after 2 attempts (>300s each)"},
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                )
                turn_record.duration_s = round(time.time() - turn_start, 3)
                episode.turns.append(turn_record)
                continue
        
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
        
        # ── Helper: post-dispatch processing (shared by sync + parallel paths) ──
        _FAILURE_PATTERNS = (
            "exit code: 1", "exit code: 2", "error:", "traceback",
            "exception", "failed", "could not", "unable to",
            "no such file", "command not found", "permission denied",
        )
        def _handle_tool_output(tool_name, tool_args, tool_call, output, child_trace, tc_duration, _spawn_subtask_id):
            """Process a completed tool dispatch: auto-mark done/failed, error-track, trace, memory, message."""
            nonlocal consecutive_error_count, last_error_signature
            
            # Detect metadata dict returned by smart_fetch_wrapper (not a real child trace)
            _tool_meta = None
            if isinstance(child_trace, dict):
                _tool_meta = child_trace
                child_trace = None
            
            # Auto-mark plan subtask on spawn_agent return
            if _spawn_subtask_id is not None and current_plan is not None and not output.startswith("ERROR:"):
                matched = [s for s in current_plan["subtasks"] if s["id"] == _spawn_subtask_id]
                if matched:
                    # Parse spawn_agent JSON to detect success vs failure
                    _mark_status = "done"
                    try:
                        parsed_output = json.loads(output)
                        result_snippet = str(parsed_output.get("response", ""))[:150]
                        turns_used = parsed_output.get("turns_used", 0)
                        # Detect failure: budget exhausted + error-looking response
                        from .config import SUB_AGENT_TURN_BUDGET
                        budget = tool_args.get("turn_length", SUB_AGENT_TURN_BUDGET)
                        response_lower = result_snippet.lower()
                        if turns_used >= budget and any(p in response_lower for p in _FAILURE_PATTERNS):
                            _mark_status = "failed"
                        elif any(p in response_lower for p in ("exit code: 1", "exit code: 2")) and "success" not in response_lower:
                            _mark_status = "failed"
                    except (json.JSONDecodeError, AttributeError):
                        result_snippet = output[:150]
                    
                    matched[0]["status"] = _mark_status
                    matched[0]["result"] = result_snippet
                    action_tag = "auto_done" if _mark_status == "done" else "auto_failed"
                    icon = "✅" if _mark_status == "done" else "❌"
                    plan_history.append({"turn": turn, "action": action_tag, "detail": f"Subtask {_spawn_subtask_id}: {result_snippet[:80]}"})
                    if verbose:
                        print(f"   {icon} Auto-marked subtask {_spawn_subtask_id} as {_mark_status}")
            
            if verbose and len(output) < 200:
                print(f"       → {output}")
            elif verbose:
                print(f"       → {output[:200]}...")
            
            # Progress callback for parent visibility
            if _progress_fn:
                _progress_fn(f"turn {turn}: {tool_name} ({tc_duration:.1f}s)")
            
            # Track consecutive identical errors
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
            
            # Symbolic memory: store full output, summarize for context
            if not output.startswith("ERROR:") and tool_name not in ("search_available_tools", "get_current_time", "add_numbers"):
                desc = ""
                if tool_name == "search_web":
                    desc = str(tool_args.get("q", ""))[:60]
                elif tool_name in ("fetch_url", "smart_fetch"):
                    desc = str(tool_args.get("url", ""))[:60]
                elif tool_name == "read_pdf":
                    desc = str(tool_args.get("url", ""))[:60]
                elif tool_name == "spawn_agent":
                    desc = str(tool_args.get("task", ""))[:60]
                elif tool_name == "sandbox_shell":
                    desc = str(tool_args.get("command", ""))[:60]
                
                mem_key = memory.add(
                    tool_name=tool_name,
                    turn=turn,
                    content=output,
                    description=desc,
                )
                
                if shell is not None:
                    try:
                        mem_dir = shell.session_workspace / "memory"
                        mem_dir.mkdir(exist_ok=True)
                        mem_file = mem_dir / f"{mem_key}.txt"
                        # If the tool provided full (un-truncated) content, save that to disk
                        # so agents can selectively grep/head through it.
                        disk_content = (_tool_meta or {}).get("_full_content") or output
                        mem_file.write_text(disk_content)
                    except Exception as e:
                        logger.debug(f"Could not write memory file: {e}")
                
                snippet = output[:1500].strip()
                if snippet:
                    findings.append(f"[{tool_name}] {snippet}")
                
                summary = extract_summary(tool_name, output, tool_args)
                
                if len(summary) < len(output):
                    has_full = bool((_tool_meta or {}).get("_full_content"))
                    if has_full:
                        msg_output = (
                            f"{summary}\n\n"
                            f"📦 Full page stored on disk: /workspace/memory/{mem_key}.txt\n"
                            f"   → Use sandbox_shell to search: grep -i 'keyword' /workspace/memory/{mem_key}.txt\n"
                            f"   → Or peek: head -200 /workspace/memory/{mem_key}.txt\n"
                            f"   → Or recall(key=\"{mem_key}\") for the truncated version"
                        )
                    else:
                        msg_output = f"{summary}\n\n📦 Full output stored as: {mem_key} (use recall to retrieve)"
                else:
                    msg_output = summary
            else:
                msg_output = output
            
            msg_output = re.sub(
                r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
                r"\1[file content saved to trace]\2",
                msg_output, flags=re.DOTALL
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(msg_output)
            })
        
        # ── Pre-scan for parallel spawn optimization ──────────────────
        _spawn_tc_count = sum(
            1 for tc in tool_calls_in_msg
            if tc["function"]["name"].split("<|")[0] == "spawn_agent"
        )
        _spawn_executor = None
        _deferred_spawns = []
        if _spawn_tc_count >= 2:
            _spawn_executor = ThreadPoolExecutor(max_workers=_spawn_tc_count)
            if verbose:
                print(f"   ⚡ {_spawn_tc_count} spawn_agents detected — will dispatch in parallel")
        
        for i, tool_call in enumerate(tool_calls_in_msg, 1):
            tool_name = tool_call["function"]["name"]
            # Sanitize: strip leaked channel tokens (e.g. "search_web<|channel|>commentary")
            if "<|" in tool_name:
                tool_name = tool_name.split("<|")[0]
            raw_args = tool_call["function"].get("arguments", "")
            try:
                tool_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                # Fallback 1: try Python literal (model sometimes emits single-quoted dicts)
                import ast as _ast
                try:
                    parsed = _ast.literal_eval(raw_args)
                    tool_args = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    tool_args = {}
                
                if not tool_args:
                    # Could not recover — tell the model what went wrong
                    if verbose:
                        print(f"   ⚠️  Malformed tool arguments for {tool_name}: {raw_args[:200]}")
                    error_msg = (
                        f"ERROR: Your arguments for {tool_name} were malformed JSON and could not be parsed.\n"
                        f"Raw arguments received: {raw_args[:300]}\n\n"
                        f"FIX: Call {tool_name} again with valid JSON arguments. "
                        f"Make sure to use double quotes for strings and escape any special characters."
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": error_msg
                    })
                    tc_record = ToolCallRecord(
                        tool_name=tool_name,
                        tool_args={"_raw_malformed": raw_args[:500]},
                        tool_call_id=tool_call["id"],
                        output=error_msg,
                        duration_s=0,
                        child_trace=None,
                    )
                    turn_record.tool_calls.append(tc_record)
                    continue
                elif verbose:
                    print(f"   ⚠️  Recovered tool arguments for {tool_name} via ast.literal_eval")
            
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
            
            # ── Check for create_plan / update_plan ───────────────────
            if tool_name == "create_plan":
                # Parse structured plan: goal + subtasks list
                goal = tool_args.get("goal", "")
                subtasks_raw = tool_args.get("subtasks", [])
                # Fallback: if model used old format (single "plan" string)
                if not goal and not subtasks_raw and tool_args.get("plan"):
                    plan_text = tool_args["plan"]
                    lines = [l.strip() for l in plan_text.split("\n") if l.strip()]
                    goal = lines[0] if lines else "Research"
                    subtasks_raw = lines[1:] if len(lines) > 1 else [goal]
                    import re as _re
                    subtasks_raw = [_re.sub(r"^[\d]+[\.\)]\s*", "", s).strip() for s in subtasks_raw]
                    subtasks_raw = [_re.sub(r"^[-*]\s*", "", s).strip() for s in subtasks_raw]
                if isinstance(subtasks_raw, str):
                    subtasks_raw = [s.strip() for s in subtasks_raw.split("\n") if s.strip()]
                
                current_plan = {
                    "goal": goal or "Research",
                    "subtasks": [
                        {"id": i + 1, "task": t, "status": "not_started", "result": None}
                        for i, t in enumerate(subtasks_raw) if t
                    ]
                }
                plan_history.append({"turn": turn, "action": "create", "detail": goal})
                
                rendered = _render_plan()
                confirmation = (
                    f"Plan created and stored.\n\n{rendered}\n\n"
                    "Tip: Use spawn_agent(task='...', subtask_id=N) to auto-mark subtasks."
                )
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": confirmation
                })
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=confirmation,
                    duration_s=0, child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                
                if verbose:
                    print(f"   📋 Plan created:")
                    for line in rendered.split("\n"):
                        print(f"      {line}")
                continue
            
            if tool_name == "update_plan":
                if current_plan is None:
                    confirmation = "ERROR: No plan exists yet. Call create_plan first."
                else:
                    changes = []
                    # Handle subtask status update
                    st_id = tool_args.get("subtask_id")
                    if st_id is not None:
                        matched = [s for s in current_plan["subtasks"] if s["id"] == st_id]
                        if matched:
                            st = matched[0]
                            new_status = tool_args.get("status", st["status"])
                            result_note = tool_args.get("result")
                            old_status = st["status"]
                            st["status"] = new_status
                            if result_note:
                                st["result"] = result_note
                            changes.append(f"Subtask {st_id}: {old_status} → {new_status}" + (f" ({result_note})" if result_note else ""))
                        else:
                            changes.append(f"Warning: subtask_id {st_id} not found in plan")
                    
                    # Handle adding a new subtask
                    new_subtask = tool_args.get("add_subtask")
                    if new_subtask:
                        next_id = max(s["id"] for s in current_plan["subtasks"]) + 1 if current_plan["subtasks"] else 1
                        current_plan["subtasks"].append({
                            "id": next_id, "task": new_subtask,
                            "status": "not_started", "result": None
                        })
                        changes.append(f"Added subtask {next_id}: {new_subtask}")
                    
                    # Handle goal revision
                    new_goal = tool_args.get("new_goal")
                    if new_goal:
                        current_plan["goal"] = new_goal
                        changes.append(f"Goal revised: {new_goal}")
                    
                    # Fallback: old-style full plan replacement
                    if not changes and tool_args.get("plan"):
                        plan_text = tool_args["plan"]
                        current_plan["goal"] = plan_text
                        changes.append("Plan replaced (free-text)")
                    
                    plan_history.append({"turn": turn, "action": "update", "detail": "; ".join(changes)})
                    rendered = _render_plan()
                    confirmation = f"Plan updated: {'; '.join(changes)}\n\n{rendered}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": confirmation
                })
                tc_record = ToolCallRecord(
                    tool_name=tool_name, tool_args=tool_args,
                    tool_call_id=tool_call["id"], output=confirmation,
                    duration_s=0, child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                
                if verbose:
                    print(f"   📝 Plan updated: {'; '.join(changes) if current_plan else 'no plan'}")
                    if current_plan:
                        for line in _render_plan().split("\n"):
                            print(f"      {line}")
                continue
            
            # ── Check for recall ──────────────────────────────────────
            if tool_name == "recall":
                recall_key = tool_args.get("key", "")
                
                if recall_key == "index":
                    # Return the memory index
                    recall_output = build_memory_index(memory.entries)
                else:
                    # Look up the key in memory store
                    recall_content = memory.get(recall_key)
                    if recall_content is not None:
                        recall_output = recall_content
                    else:
                        # Try reading from disk
                        recall_output = None
                        if shell is not None:
                            try:
                                mem_file = shell.session_workspace / "memory" / f"{recall_key}.txt"
                                if mem_file.exists():
                                    recall_output = mem_file.read_text()
                            except Exception:
                                pass
                        if recall_output is None:
                            available_keys = memory.keys()
                            recall_output = (
                                f"ERROR: Key '{recall_key}' not found in memory.\n"
                                f"Available keys: {', '.join(available_keys) if available_keys else '(none)'}\n"
                                "Use recall(key='index') to see all stored entries."
                            )
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(recall_output)
                })
                
                tc_record = ToolCallRecord(
                    tool_name="recall",
                    tool_args=tool_args,
                    tool_call_id=tool_call["id"],
                    output=str(recall_output)[:500],
                    duration_s=0,
                    child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                
                if verbose:
                    preview = str(recall_output)[:150]
                    print(f"   🔍 Recalled {recall_key}: {preview}...")
                continue
            
            # ── Check for store_memory ────────────────────────────────
            if tool_name == "store_memory":
                store_content = tool_args.get("content", "")
                store_desc = tool_args.get("description", "user_note")
                
                mem_key = memory.add(
                    tool_name="store_memory",
                    turn=turn,
                    content=store_content,
                    description=store_desc,
                )
                
                # Write to disk
                if shell is not None:
                    try:
                        mem_dir = shell.session_workspace / "memory"
                        mem_dir.mkdir(exist_ok=True)
                        mem_file = mem_dir / f"{mem_key}.txt"
                        mem_file.write_text(store_content)
                    except Exception as e:
                        logger.debug(f"Could not write memory file: {e}")
                
                confirmation = f"Stored as: {mem_key} ({len(store_content):,} chars). Use recall(key='{mem_key}') to retrieve."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": confirmation
                })
                
                tc_record = ToolCallRecord(
                    tool_name="store_memory",
                    tool_args=tool_args,
                    tool_call_id=tool_call["id"],
                    output=confirmation,
                    duration_s=0,
                    child_trace=None,
                )
                turn_record.tool_calls.append(tc_record)
                
                if verbose:
                    print(f"   💾 Stored: {mem_key} ({len(store_content):,} chars)")
                continue
            
            # Execute tool
            tc_start = time.time()
            child_trace = None
            
            # ── Auto-mark plan subtask on spawn_agent dispatch ────────
            _spawn_subtask_id = None
            if tool_name == "spawn_agent" and current_plan is not None:
                _spawn_subtask_id = tool_args.get("subtask_id")
                if _spawn_subtask_id is not None:
                    matched = [s for s in current_plan["subtasks"] if s["id"] == _spawn_subtask_id]
                    if matched:
                        matched[0]["status"] = "in_progress"
                        plan_history.append({"turn": turn, "action": "auto_in_progress", "detail": f"Subtask {_spawn_subtask_id}"})
                        if verbose:
                            print(f"   🔄 Auto-marked subtask {_spawn_subtask_id} as in_progress")
            
            # Build progress callback for sub-agent visibility (Fix A)
            _sub_progress = None
            if verbose and tool_name == "spawn_agent":
                _stid = _spawn_subtask_id  # capture for closure
                _sub_progress = lambda msg, _id=_stid: print(
                    f"   ↳ [sub{'' if _id is None else f' #{_id}'}] {msg}")
            
            # ── Parallel dispatch for concurrent spawn_agents (Fix B) ──
            if _spawn_executor is not None and tool_name == "spawn_agent":
                future = _spawn_executor.submit(
                    dispatch_tool_call, tool_name, tool_args,
                    _depth=_depth, model=model, reasoning_effort=reasoning_effort,
                    _sandbox_files=_sandbox_files, _shell=shell,
                    _progress_fn=_sub_progress
                )
                _deferred_spawns.append({
                    'future': future, 'tool_name': tool_name,
                    'tool_args': tool_args, 'tool_call': tool_call,
                    'subtask_id': _spawn_subtask_id, 'tc_start': tc_start,
                })
                if verbose:
                    print(f"   ⏳ Queued for parallel dispatch ({len(_deferred_spawns)}/{_spawn_tc_count})")
                continue
            
            # ── Synchronous dispatch ──────────────────────────────────
            try:
                output, child_trace = dispatch_tool_call(
                    tool_name, tool_args, _depth=_depth, model=model,
                    reasoning_effort=reasoning_effort, _sandbox_files=_sandbox_files,
                    _shell=shell, _progress_fn=_sub_progress
                )
            except Exception as e:
                output = f"ERROR: {str(e)}"
                child_trace = None
                if verbose:
                    print(f"       → ❌ {output}")
            tc_duration = round(time.time() - tc_start, 3)
            _handle_tool_output(tool_name, tool_args, tool_call, output, child_trace, tc_duration, _spawn_subtask_id)
        
        # ── Process parallel spawn results ────────────────────────────
        if _deferred_spawns:
            if verbose:
                print(f"   ⌛ Waiting for {len(_deferred_spawns)} parallel sub-agents...")
            for ds in _deferred_spawns:
                try:
                    output, child_trace = ds['future'].result(timeout=660)
                except Exception as e:
                    output, child_trace = f"ERROR: {str(e)}", None
                    if verbose:
                        print(f"       → ❌ {output}")
                tc_duration = round(time.time() - ds['tc_start'], 3)
                _handle_tool_output(
                    ds['tool_name'], ds['tool_args'], ds['tool_call'],
                    output, child_trace, tc_duration, ds['subtask_id']
                )
            _spawn_executor.shutdown(wait=False)
            _deferred_spawns.clear()
        
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
    # Upload the full MemoryStore as a JSON file to the shell workspace,
    # then spawn a fresh sub-agent whose job is to query the data
    # PROGRAMMATICALLY (via sandbox_shell) and synthesize an answer.
    #
    # Key insight: the research data never enters the LLM context.
    # It lives on the shell filesystem. The sub-agent does programmatic
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
        
        # Upload the file directly to the shell's workspace (if available).
        # The synthesizer will read it with sandbox_shell(command="python3 ...").
        # sandbox_files is still passed as a flag (+ fallback for execute_code
        # in case shell is unavailable).
        if shell is not None:
            research_path = shell.session_workspace / "research_data.json"
            research_path.write_text(memory_json)
        
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
                _shell=shell,
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
