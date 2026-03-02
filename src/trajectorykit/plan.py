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
from typing import Optional


@dataclass
class PlanTask:
    """A single tracked subtask."""

    description: str
    status: str = "pending"          # pending | in_progress | done | failed
    tool_used: Optional[str] = None
    result_summary: Optional[str] = None   # ≤200 chars
    memory_key: Optional[str] = None

    # ── Status icons ──────────────────────────────────────────────────
    _ICONS = {
        "pending":     "⬜",
        "in_progress": "🔄",
        "done":        "✅",
        "failed":      "❌",
    }

    def render(self, index: int) -> str:
        icon = self._ICONS.get(self.status, "?")
        line = f"  {icon} [{index}] {self.description}"
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

        # Check if there's an existing in_progress task that matches
        existing = self._find_matching(tool_name, desc)
        if existing is not None:
            task = self.subtasks[existing]
            task.status = "failed" if is_error else "done"
            task.result_summary = summary
            task.memory_key = memory_key
            return

        # Add new task
        status = "failed" if is_error else "done"
        self.subtasks.append(PlanTask(
            description=desc,
            status=status,
            tool_used=tool_name,
            result_summary=summary,
            memory_key=memory_key,
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

        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────

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
