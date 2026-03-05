"""
Typed state for the trajectorykit agent loop.

Replaces the ~30 scattered local variables in the old dispatch() monolith
with a single, typed, introspectable object.  All node functions receive
and mutate this state.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    MODEL_NAME,
    SYSTEM_PROMPT,
    WORKER_PROMPT,
    SYNTHESIZER_PROMPT,
    TOKEN_SAFETY_MARGIN,
    TRACES_DIR,
    get_model_profile,
    SYMBOLIC_REFERENCES,
    SYMBOLIC_THRESHOLD,
    PLAN_STATE,
    PLAN_INJECT_INTERVAL,
    MAX_VERIFICATION_REJECTIONS,
    MAX_SPOT_CHECK_REJECTIONS,
)
from .memory import MemoryStore
from .plan import ResearchPlan
from .tool_store import TOOLS, ROOT_TOOLS
from .tracing import EpisodeTrace, TurnRecord, ToolCallRecord


@dataclass
class AgentState:
    """Complete typed state for a single dispatch() execution.

    Every local variable that was scattered through the old dispatch()
    monolith is now a named field here.  Node functions receive this
    object and mutate it in place — no globals, no closures.
    """

    # ── Core parameters ───────────────────────────────────────────────
    user_input: str
    turn_length: Optional[int]
    verbose: bool
    max_tokens: int
    temperature: float
    model: str
    reasoning_effort: Optional[str]
    example_id: str
    depth: int
    sandbox_files: Optional[Dict[str, str]]
    is_synthesizer: bool

    # ── Model profile ─────────────────────────────────────────────────
    profile: dict
    context_window: int

    # ── Messages ──────────────────────────────────────────────────────
    messages: List[dict]

    # ── Tool configuration ────────────────────────────────────────────
    system_prompt: str
    available_tools: List[dict]
    terminal_tool_name: str
    terminal_tool: dict

    # ── Trace ─────────────────────────────────────────────────────────
    episode: EpisodeTrace
    episode_start: float

    # ── Turn tracking ─────────────────────────────────────────────────
    turn: int = 0
    total_tool_calls: int = 0

    # ── Error tracking ────────────────────────────────────────────────
    consecutive_error_count: int = 0
    consecutive_no_tool_count: int = 0
    last_error_signature: Optional[str] = None

    # ── Research tracking ─────────────────────────────────────────────
    findings: List[str] = field(default_factory=list)
    consecutive_search_count: int = 0

    # ── Memory & Plan ─────────────────────────────────────────────────
    memory: MemoryStore = field(default_factory=MemoryStore)
    plan: Optional[ResearchPlan] = None

    # ── Draft management (root only) ──────────────────────────────────
    draft_path: Optional[str] = None
    draft_versions: List[Tuple[int, str]] = field(default_factory=list)

    # ── Verification (root only) ──────────────────────────────────────
    verification_rejections: int = 0
    spot_check_rejections: int = 0

    # ── Cycle-enforcement tracking (root only) ────────────────────────
    conduct_research_count: int = 0
    draft_revised_since_rejection: bool = True  # starts True (no rejection yet)

    # ── Degeneration flag ─────────────────────────────────────────────
    degenerated: bool = False

    # ── Constants (configurable per run via config.yaml) ──────────────
    MAX_CONSECUTIVE_ERRORS: int = 3
    MAX_CONSECUTIVE_SEARCHES: int = 6
    CONSECUTIVE_SEARCH_WARNING: int = 5
    MAX_VERIFICATION_REJECTIONS: int = MAX_VERIFICATION_REJECTIONS
    MAX_SPOT_CHECK_REJECTIONS: int = MAX_SPOT_CHECK_REJECTIONS


def create_state(
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
) -> AgentState:
    """Initialize an AgentState from dispatch() parameters.

    Mirrors the top ~200 lines of the old dispatch() function:
    parameter resolution, trace creation, prompt/tool selection,
    state variable initialization.
    """

    # ── Resolve model and its profile ─────────────────────────────────
    model = model or MODEL_NAME
    profile = get_model_profile(model)
    context_window = profile["context_window"]

    if temperature is None:
        temperature = profile.get("default_temperature", 0.7)
    if max_tokens is None:
        max_tokens = context_window
    if reasoning_effort is None and profile.get("supports_reasoning_effort"):
        reasoning_effort = profile.get("default_reasoning_effort")

    # After resolution, these are guaranteed non-None
    assert temperature is not None
    assert max_tokens is not None

    # ── Initialize trace ──────────────────────────────────────────────
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

    # ── Depth-aware prompt & tool selection ────────────────────────────
    if _is_synthesizer:
        system_prompt = SYNTHESIZER_PROMPT
        available_tools = [
            t for t in TOOLS
            if t["function"]["name"] in ("execute_code", "final_answer", "search_available_tools")
        ]
    elif _depth == 0:
        system_prompt = SYSTEM_PROMPT
        available_tools = ROOT_TOOLS
    else:
        system_prompt = WORKER_PROMPT
        available_tools = [
            t for t in TOOLS
            if t["function"]["name"] not in ("spawn_agent", "recall_memory", "update_draft")
        ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    # ── Terminal tool schema ──────────────────────────────────────────
    terminal_tool_name = "research_complete" if _depth == 0 else "final_answer"
    terminal_tool = next(
        (t for t in available_tools if t["function"]["name"] == terminal_tool_name),
        next((t for t in TOOLS if t["function"]["name"] == "final_answer"), available_tools[0]),
    )

    # ── Draft file path (root only) ──────────────────────────────────
    draft_path: Optional[str] = None
    if _depth == 0:
        os.makedirs(TRACES_DIR, exist_ok=True)
        draft_path = os.path.join(TRACES_DIR, f"draft_{episode.trace_id}.md")
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write("")

    # ── Research plan (root only) ─────────────────────────────────────
    plan: Optional[ResearchPlan] = None
    if PLAN_STATE and _depth == 0:
        q_plan = user_input[:300] + ("\u2026" if len(user_input) > 300 else "")
        plan = ResearchPlan(question=q_plan)
        plan.INJECT_EVERY_N_TURNS = PLAN_INJECT_INTERVAL

    return AgentState(
        user_input=user_input,
        turn_length=turn_length,
        verbose=verbose,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        reasoning_effort=reasoning_effort,
        example_id=example_id or "",
        depth=_depth,
        sandbox_files=_sandbox_files,
        is_synthesizer=_is_synthesizer,
        profile=profile,
        context_window=context_window,
        messages=messages,
        system_prompt=system_prompt,
        available_tools=available_tools,
        terminal_tool_name=terminal_tool_name,
        terminal_tool=terminal_tool,
        episode=episode,
        episode_start=time.time(),
        draft_path=draft_path,
        plan=plan,
    )
