"""
Main agent entry point for the trajectorykit system.

This module is now a thin wrapper. The real implementation lives in:
  - agent_state.py  — AgentState dataclass and create_state() factory
  - nodes.py        — individual tool handler functions + TOOL_HANDLERS dispatch table
  - runner.py       — turn loop (run_agent_loop) and post-loop synthesis

dispatch() remains here as the public API so that all existing imports
(``from .agent import dispatch``, ``from trajectorykit.agent import dispatch``)
continue to work without changes.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from .agent_state import create_state
from .chain import analyze_chain
from . import config as _cfg
from .runner import run_agent_loop

import logging
logger = logging.getLogger(__name__)


def dispatch(
    user_input: str,
    turn_length: Optional[int] = 5,
    verbose: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    example_id: Optional[str] = None,
    config_path: Optional[str] = None,
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
        config_path: Path to a YAML config file. If None, uses the default config
                     (configs/default.yaml). Use this to switch between modes, e.g.
                     "configs/experiments/gpt_oss_deep_research_bench.yaml" for
                     report/bench mode.
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
    # Reload config if a custom path was provided (root dispatch only)
    if config_path is not None and _depth == 0:
        _cfg.load_config(config_path)
        if verbose:
            print(f"⚙️  Config loaded: {config_path} (draft_format={_cfg.DRAFT_FORMAT})")

    state = create_state(
        user_input=user_input,
        turn_length=turn_length,
        verbose=verbose,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        reasoning_effort=reasoning_effort,
        example_id=example_id,
        _depth=_depth,
        _sandbox_files=_sandbox_files,
        _is_synthesizer=_is_synthesizer,
    )

    # ── Pre-dispatch chain analysis (root only) ──────────────────────
    if _depth == 0 and _cfg.CHAIN_ANALYSIS_ENABLED and _cfg.CHAIN_ANALYSIS_PROMPT:
        chain_plan = analyze_chain(
            question=user_input,
            prompt_template=_cfg.CHAIN_ANALYSIS_PROMPT,
            model=state.model,
            vllm_url=_cfg.VLLM_API_URL,
            temperature=state.temperature,
            max_tokens=state.max_tokens,
        )
        state.chain_plan = chain_plan
        # Seed the research plan with chain steps
        if chain_plan.has_chain and state.plan is not None:
            state.plan.seed_from_chain(chain_plan)
        if verbose:
            if chain_plan.has_chain:
                print(f"⛓  Chain detected: {len(chain_plan.chain_steps)} steps, "
                      f"{len(chain_plan.parallel_tasks)} parallel tasks")
            else:
                print(f"⚡  No causal chain detected "
                      f"({len(chain_plan.parallel_tasks)} parallel tasks)")

    return run_agent_loop(state)

