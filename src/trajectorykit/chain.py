"""
Pre-dispatch causal chain analysis.

Before the orchestrator loop starts, we ask an LLM to decompose the
question and detect information-dependency chains.  The resulting
ChainPlan is attached to AgentState and used downstream for:

  - Seeding the ResearchPlan with ordered subtasks
  - Injecting chain context into the orchestrator's first turns
  - Enforcing step ordering in handle_conduct_research (hard-block
    on out-of-order dispatch)

Public API:
    analyze_chain(question, model, vllm_url, temperature) -> ChainPlan
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ChainStep:
    """A single lookup step in a causal chain."""

    step: int                          # 1-based step number
    lookup: str                        # research task (may contain placeholders)
    depends_on: Optional[int] = None   # step number this depends on, or None
    placeholder: str = ""              # e.g. "[step_1_result]"
    resolved_value: Optional[str] = None  # filled once the step completes

    @property
    def is_resolved(self) -> bool:
        return self.resolved_value is not None

    @property
    def is_unlocked(self) -> bool:
        """True if this step can be dispatched (dependency resolved or none)."""
        return self.depends_on is None or self.resolved_value is not None

    def lookup_resolved(self, plan: "ChainPlan") -> str:
        """Return the lookup text with all resolved placeholders filled in."""
        text = self.lookup
        for s in plan.chain_steps:
            if s.resolved_value is not None:
                text = text.replace(s.placeholder, s.resolved_value)
        return text


@dataclass
class ChainPlan:
    """Result of pre-dispatch chain analysis."""

    has_chain: bool = False
    chain_steps: List[ChainStep] = field(default_factory=list)
    parallel_tasks: List[str] = field(default_factory=list)

    # ── Query helpers ─────────────────────────────────────────────────

    def next_unlocked_step(self) -> Optional[ChainStep]:
        """Return the next chain step that is unlocked but unresolved."""
        for s in self.chain_steps:
            if not s.is_resolved and s.is_unlocked:
                return s
            # Also check: dependency is resolved
            if not s.is_resolved and s.depends_on is not None:
                dep = self.get_step(s.depends_on)
                if dep is not None and dep.is_resolved:
                    return s
        return None

    def get_step(self, step_num: int) -> Optional[ChainStep]:
        """Get a chain step by its step number."""
        for s in self.chain_steps:
            if s.step == step_num:
                return s
        return None

    def all_resolved(self) -> bool:
        """True when every chain step has a resolved value."""
        return all(s.is_resolved for s in self.chain_steps)

    def resolve_step(self, step_num: int, value: str) -> None:
        """Mark a chain step as resolved with the given value."""
        step = self.get_step(step_num)
        if step is not None:
            step.resolved_value = value
            logger.info("Chain step %d resolved → %s", step_num, value[:100])

    def contest_step(self, step_num: int) -> List[int]:
        """Clear a chain step's resolved value and cascade to dependents.

        Returns the list of step numbers that were cleared.
        """
        to_clear: set[int] = {step_num}
        # Cascade: any step depending on a cleared step must also clear
        changed = True
        while changed:
            changed = False
            for s in self.chain_steps:
                if s.step not in to_clear and s.depends_on in to_clear:
                    to_clear.add(s.step)
                    changed = True
        cleared = []
        for s in self.chain_steps:
            if s.step in to_clear and s.is_resolved:
                logger.info(
                    "Chain step %d contested (was: %s)",
                    s.step, s.resolved_value[:80] if s.resolved_value else "?",
                )
                s.resolved_value = None
                cleared.append(s.step)
        return sorted(cleared)

    def unresolved_dependencies_for(self, task_text: str) -> List[ChainStep]:
        """Return unresolved chain steps whose placeholders appear in task_text."""
        missing = []
        for s in self.chain_steps:
            if not s.is_resolved and s.placeholder and s.placeholder in task_text:
                missing.append(s)
        return missing

    # ── Rendering ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for trace storage."""
        return {
            "has_chain": self.has_chain,
            "chain_steps": [
                {
                    "step": s.step,
                    "lookup": s.lookup,
                    "depends_on": s.depends_on,
                    "placeholder": s.placeholder,
                    "resolved_value": s.resolved_value,
                }
                for s in self.chain_steps
            ],
            "parallel_tasks": self.parallel_tasks,
        }

    def render(self) -> str:
        """Render the chain plan as a context block for the orchestrator."""
        if not self.has_chain and not self.parallel_tasks:
            return ""

        lines = ["═══ CHAIN ANALYSIS ═══"]

        if self.has_chain and self.chain_steps:
            lines.append("")
            lines.append("⛓ CAUSAL CHAIN (must resolve in order):")
            for s in self.chain_steps:
                if s.is_resolved:
                    icon = "✅"
                    detail = f" → {s.resolved_value}"
                elif s.depends_on is not None:
                    dep = self.get_step(s.depends_on)
                    if dep is not None and dep.is_resolved:
                        icon = "🔓"  # unlocked — ready to dispatch
                        detail = f"  [READY — resolve next]"
                    else:
                        icon = "🔒"  # locked — waiting on dependency
                        detail = f"  [waiting on step {s.depends_on}]"
                else:
                    icon = "🔓"  # no dependency — always unlocked
                    detail = ""
                lookup_display = s.lookup_resolved(self) if s.depends_on else s.lookup
                lines.append(f"  {icon} Step {s.step}: {lookup_display}{detail}")

        if self.parallel_tasks:
            lines.append("")
            lines.append("⚡ PARALLEL TASKS (independent — can dispatch anytime):")
            for t in self.parallel_tasks:
                lines.append(f"  • {t}")

        if self.has_chain:
            lines.append("")
            lines.append(
                "IMPORTANT: Resolve chain steps IN ORDER. Do NOT skip ahead — "
                "later steps depend on earlier results."
            )

        lines.append("═══════════════════════")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# CHAIN ANALYSIS LLM CALL
# ═══════════════════════════════════════════════════════════════════════

def _extract_xml_tag(text: str, tag: str) -> str:
    """Extract content between <tag> and </tag>. Falls back to full text."""
    match = re.search(
        rf'<{tag}>\s*(.*?)\s*</{tag}>',
        text, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    # Fallback: opening tag but no closing (truncated)
    match_open = re.search(rf'<{tag}>\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if match_open:
        return match_open.group(1).strip()
    return text.strip()


def _parse_chain_response(raw: str) -> ChainPlan:
    """Parse the LLM's <analysis> JSON into a ChainPlan."""
    json_text = _extract_xml_tag(raw, "analysis")

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        # Try to salvage — find anything that looks like JSON
        m = re.search(r'\{.*\}', json_text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.warning("Chain analysis: failed to parse JSON, treating as no-chain")
                return ChainPlan(has_chain=False)
        else:
            logger.warning("Chain analysis: no JSON found, treating as no-chain")
            return ChainPlan(has_chain=False)

    has_chain = data.get("has_chain", False)

    chain_steps = []
    for raw_step in data.get("chain_steps", []):
        chain_steps.append(ChainStep(
            step=raw_step.get("step", len(chain_steps) + 1),
            lookup=raw_step.get("lookup", ""),
            depends_on=raw_step.get("depends_on"),
            placeholder=raw_step.get("placeholder", f"[step_{len(chain_steps)+1}_result]"),
        ))

    parallel_tasks = data.get("parallel_tasks", [])

    return ChainPlan(
        has_chain=has_chain,
        chain_steps=chain_steps,
        parallel_tasks=parallel_tasks,
    )


def analyze_chain(
    question: str,
    prompt_template: str,
    model: str,
    vllm_url: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> ChainPlan:
    """Run the pre-dispatch chain analysis LLM call.

    Parameters
    ----------
    question : str
        The user's research question.
    prompt_template : str
        The loaded chain_analysis.txt prompt (already has {current_date} resolved).
    model : str
        Model name for the API call.
    vllm_url : str
        Base URL for the VLLM API.
    temperature : float
        Sampling temperature (lower = more deterministic for analysis).
    max_tokens : int
        Max tokens for the response.

    Returns
    -------
    ChainPlan
        The parsed chain plan. On any failure, returns a no-chain plan
        so the pipeline gracefully degrades.
    """
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Analyze this question for causal chains:\n\n{question}"},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            f"{vllm_url}/chat/completions",
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if content:
                    plan = _parse_chain_response(content)
                    if plan.has_chain:
                        logger.info(
                            "Chain detected: %d steps, %d parallel tasks",
                            len(plan.chain_steps), len(plan.parallel_tasks),
                        )
                    else:
                        logger.info(
                            "No chain detected. %d parallel tasks.",
                            len(plan.parallel_tasks),
                        )
                    return plan
        logger.warning("Chain analysis API call failed (HTTP %s), defaulting to no-chain", resp.status_code)
    except Exception as e:
        logger.warning("Chain analysis failed: %s — defaulting to no-chain", e)

    return ChainPlan(has_chain=False)
