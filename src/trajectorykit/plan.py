"""
ResearchPlan: Structured progress tracker for the root orchestrator.

Mechanically tracks subtasks derived from tool calls (Option A — no model
cooperation needed).  Rendered as a compact system message and injected
into the conversation every N turns, replacing the blunt question-reminder.

The plan gives the orchestrator a "map" of:
  - What subtasks have been started / completed / failed
  - Key findings so far
  - Which memory keys hold the full data
  - How much budget remains
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .chain import ChainPlan


@dataclass
class PlanTask:
    """A single tracked subtask."""

    description: str
    status: str = "pending"          # pending | in_progress | done | failed
    tool_used: Optional[str] = None
    result_summary: Optional[str] = None   # ≤200 chars
    memory_key: Optional[str] = None
    data_quality: str = "none"       # none | weak | strong
    chain_step: Optional[int] = None  # linked chain step number, or None

    # ── Status icons ──────────────────────────────────────────────────
    _ICONS = {
        "pending":     "⬜",
        "in_progress": "🔄",
        "done":        "✅",
        "failed":      "❌",
    }

    _QUALITY_ICONS = {
        "none":   "⚫",
        "weak":   "🟡",
        "strong": "🟢",
    }

    def render(self, index: int) -> str:
        icon = self._ICONS.get(self.status, "?")
        q_icon = self._QUALITY_ICONS.get(self.data_quality, "")
        chain_tag = f" [chain step {self.chain_step}]" if self.chain_step is not None else ""
        line = f"  {icon}{q_icon} [{index}] {self.description}{chain_tag}"
        if self.result_summary:
            line += f" → {self.result_summary}"
        if self.memory_key:
            line += f"  [{self.memory_key}]"
        return line


@dataclass
class ResearchPlan:
    """Root-level research plan tracker.

    Updated mechanically after each tool call — no model cooperation needed.
    """

    question: str                            # original question (truncated)
    subtasks: list[PlanTask] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    current_phase: str = "research"          # research | synthesis

    # ── Injection interval ────────────────────────────────────────────
    INJECT_EVERY_N_TURNS: int = 3

    # ── Public API ────────────────────────────────────────────────────

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        output: str,
        memory_key: Optional[str] = None,
        is_error: bool = False,
    ) -> None:
        """Record a tool call, creating or updating a subtask entry."""
        desc = self._describe(tool_name, tool_args)
        summary = self._extract_summary(output, is_error)
        quality = self._assess_quality(output, is_error, tool_name)

        # Check if there's an existing in_progress task that matches
        existing = self._find_matching(tool_name, desc)
        if existing is not None:
            task = self.subtasks[existing]
            task.status = "failed" if is_error else "done"
            task.result_summary = summary
            task.memory_key = memory_key
            task.data_quality = quality
            return

        # Add new task
        status = "failed" if is_error else "done"
        self.subtasks.append(PlanTask(
            description=desc,
            status=status,
            tool_used=tool_name,
            result_summary=summary,
            memory_key=memory_key,
            data_quality=quality,
        ))

        # Extract key facts from non-error spawn_agent results
        # (they return concise answers that are often the core findings)
        if not is_error and tool_name == "spawn_agent" and len(output) > 30:
            fact = output[:150].strip()
            # Clean JSON wrapper if present
            if fact.startswith("{"):
                import json as _json
                try:
                    parsed = _json.loads(output)
                    resp = parsed.get("response", "")
                    if resp:
                        fact = resp[:150].strip()
                except (ValueError, KeyError):
                    pass
            if fact and fact not in self.key_facts:
                self.key_facts.append(fact)
                # Cap key facts list
                if len(self.key_facts) > 8:
                    self.key_facts = self.key_facts[-8:]

    def should_inject(self, turn: int) -> bool:
        """Return True if the plan should be injected this turn."""
        if turn < 2:
            return False  # wait for at least one tool call
        return turn % self.INJECT_EVERY_N_TURNS == 0

    def render(self, turn: int, total_turns: Optional[int] = None) -> str:
        """Render the plan as a system message string."""
        budget = f"turn {turn}"
        if total_turns:
            budget += f"/{total_turns}"

        lines = [f"📋 RESEARCH PLAN ({budget}):"]
        lines.append(f"Q: {self.question}")
        lines.append("")

        if self.subtasks:
            lines.append("Tasks:")
            for i, task in enumerate(self.subtasks, 1):
                lines.append(task.render(i))
        else:
            lines.append("Tasks: (none recorded yet)")

        if self.key_facts:
            lines.append("")
            lines.append("Key facts:")
            for fact in self.key_facts:
                lines.append(f"  • {fact}")

        # Concise status summary
        done = sum(1 for t in self.subtasks if t.status == "done")
        failed = sum(1 for t in self.subtasks if t.status == "failed")
        pending = sum(1 for t in self.subtasks if t.status == "pending")
        lines.append("")
        lines.append(
            f"Progress: {done} done, {failed} failed, {pending} pending. "
            "If you have enough information, call final_answer."
        )

        # Append gap assessment when there are completed tasks
        if done >= 2:
            gap_lines = self._gap_assessment_lines()
            if gap_lines:
                lines.append("")
                lines.extend(gap_lines)

        return "\n".join(lines)

    def render_gap_check(self) -> Optional[str]:
        """Render a standalone gap analysis for synthesis-transition decisions.

        Called by the agent before triggering synthesis. Returns None if there
        are not enough tasks to assess, otherwise a compact system message.
        """
        done = sum(1 for t in self.subtasks if t.status == "done")
        if done < 1:
            return None

        lines = ["📊 INFORMATION GAP CHECK — review before final_answer:"]
        lines.append(f"Q: {self.question}")
        lines.append("")
        lines.extend(self._gap_assessment_lines())
        return "\n".join(lines)

    def seed_from_chain(self, chain_plan: "ChainPlan") -> None:
        """Pre-populate subtasks from the chain analysis results.

        Chain steps become pending tasks in order, followed by parallel tasks.
        This gives the orchestrator a pre-built research plan to follow.
        """
        for step in chain_plan.chain_steps:
            self.subtasks.append(PlanTask(
                description=step.lookup,
                status="pending",
                chain_step=step.step,
            ))
        for task_desc in chain_plan.parallel_tasks:
            self.subtasks.append(PlanTask(
                description=task_desc,
                status="pending",
            ))

    def find_chain_task(self, chain_step_num: int) -> Optional[PlanTask]:
        """Find the PlanTask linked to a specific chain step number."""
        for t in self.subtasks:
            if t.chain_step == chain_step_num:
                return t
        return None

    def _gap_assessment_lines(self) -> list[str]:
        """Shared gap analysis logic used by render() and render_gap_check()."""
        lines: list[str] = []

        strong = [t for t in self.subtasks if t.data_quality == "strong" and t.status == "done"]
        weak = [t for t in self.subtasks if t.data_quality == "weak" and t.status == "done"]
        failed = [t for t in self.subtasks if t.status == "failed"]
        no_data = [t for t in self.subtasks if t.data_quality == "none" and t.status == "done"]

        lines.append("Data quality:")
        lines.append(f"  🟢 Strong: {len(strong)}  🟡 Weak: {len(weak)}  "
                      f"❌ Failed: {len(failed)}  ⚫ No data: {len(no_data)}")

        # Identify specific gaps
        gaps = []
        for t in failed:
            gaps.append(f"  - FAILED: {t.description}")
        for t in weak:
            gaps.append(f"  - WEAK: {t.description} → {t.result_summary or '(no detail)'}")
        for t in no_data:
            gaps.append(f"  - NO DATA: {t.description}")

        if gaps:
            lines.append("Gaps to consider filling:")
            lines.extend(gaps)
            lines.append("→ Consider spawning targeted sub-agents for these "
                         "gaps before calling final_answer.")
        else:
            lines.append("Coverage looks solid — proceed to final_answer "
                         "when ready.")

        return lines

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _assess_quality(output: str, is_error: bool, tool_name: str) -> str:
        """Heuristic quality assessment of a tool call result.

        Returns 'none', 'weak', or 'strong' based on output signals.
        """
        if is_error:
            return "none"

        text = output.strip()

        # For spawn_agent, unwrap JSON response
        if text.startswith("{") and tool_name == "spawn_agent":
            import json as _json
            try:
                parsed = _json.loads(text)
                text = (parsed.get("response", "") or "").strip()
            except (ValueError, KeyError):
                pass

        length = len(text)

        # Empty or very short → no data
        if length < 50:
            return "none"

        # Heuristic signals for "strong" quality:
        # - Substantial length (multi-paragraph or rich factual content)
        # - Contains numbers/dates (quantitative evidence)
        # - Contains URL-like citations
        has_numbers = False
        has_urls = False
        for tok in text[:2000].split():
            if any(c.isdigit() for c in tok) and len(tok) >= 3:
                has_numbers = True
            if tok.startswith(("http://", "https://", "www.")):
                has_urls = True
            if has_numbers and has_urls:
                break

        # Strong: substantial text + quantitative evidence
        if length >= 300 and has_numbers:
            return "strong"
        # Moderate content with some evidence
        if length >= 150:
            return "weak" if not has_numbers else "strong"
        # Short but factual
        if has_numbers:
            return "weak"

        return "none"

    @staticmethod
    def _describe(tool_name: str, tool_args: dict) -> str:
        """Generate a human-readable subtask description from tool call."""
        if tool_name == "spawn_agent":
            task = tool_args.get("task", "")
            # First sentence or first 80 chars
            first = task.split(".")[0].split("\n")[0][:80]
            return first or "Sub-agent task"
        elif tool_name == "search_web":
            return f'Search: "{tool_args.get("q", "?")}"'
        elif tool_name == "fetch_url":
            url = tool_args.get("url", "?")
            # Shorten URL
            if len(url) > 60:
                url = url[:57] + "..."
            return f"Fetch: {url}"
        elif tool_name == "read_pdf":
            url = tool_args.get("url", "?")
            if len(url) > 60:
                url = url[:57] + "..."
            return f"Read PDF: {url}"
        elif tool_name == "execute_code":
            code = tool_args.get("code", "")
            # First non-empty, non-import line
            for line in code.split("\n"):
                stripped = line.strip()
                if stripped and not stripped.startswith(("import ", "from ", "#", "```")):
                    return f"Code: {stripped[:60]}"
            return "Execute code"
        elif tool_name == "extract_tables":
            url = tool_args.get("url", "?")
            return f"Extract tables: {url[:50]}"
        elif tool_name == "wikipedia_lookup":
            return f'Wikipedia: "{tool_args.get("title", tool_args.get("query", "?"))}"'
        elif tool_name == "fetch_cached":
            return f"Wayback: {tool_args.get('url', '?')[:50]}"
        else:
            return f"{tool_name}"

    @staticmethod
    def _extract_summary(output: str, is_error: bool) -> str:
        """Extract a ≤200 char summary from tool output."""
        if is_error:
            return output[:150].strip()
        text = output.strip()

        # For spawn_agent, extract the response field from JSON
        if text.startswith("{"):
            import json as _json
            try:
                parsed = _json.loads(text)
                resp = parsed.get("response", "")
                if resp:
                    text = resp.strip()
            except (ValueError, KeyError):
                pass

        # For short outputs, use as-is
        if len(text) <= 200:
            return text
        # Take first meaningful line(s) up to 200 chars
        head = text[:200]
        cut = head.rfind(" ")
        if cut > 100:
            head = head[:cut]
        return head + "…"

    def _find_matching(self, tool_name: str, desc: str) -> Optional[int]:
        """Find an existing in_progress subtask that matches this tool call."""
        for i, task in enumerate(self.subtasks):
            if task.status == "in_progress" and task.tool_used == tool_name:
                return i
        return None
