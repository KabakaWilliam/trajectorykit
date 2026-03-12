"""
Trace: Structured episode recording for the trajectorykit agent system.

Captures the full execution tree including sub-agent traces, tool calls,
timing, and message history. Supports JSON serialization and pretty-printing.
"""

import json
import os
import re
import time
import uuid
import html as html_mod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import TRACES_DIR


@dataclass
class ToolCallRecord:
    """Record of a single tool call within a turn."""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str
    output: str
    duration_s: float = 0.0
    # If this tool call was spawn_agent, the child's full trace is here
    child_trace: Optional["EpisodeTrace"] = None
    # Auxiliary data (verifier results, link checks, etc.) — appears in JSON trace
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TurnRecord:
    """Record of a single turn in the agent loop."""
    turn_number: int
    assistant_content: Optional[str] = None
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    # Raw assistant message from the API (includes tool_calls array etc.)
    raw_assistant_message: Optional[Dict[str, Any]] = None
    duration_s: float = 0.0
    # Token usage from the API response
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Chain analysis snapshot: captures chain step states after this turn
    # Only populated for root (depth==0) turns when a chain plan is active.
    # List of dicts: [{step, placeholder, resolved_value}, ...]
    chain_snapshot: Optional[List[Dict[str, Any]]] = None


@dataclass
class EpisodeTrace:
    """Full trace of a dispatch episode, forming a tree with sub-agent traces."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    example_id: str = ""  # Semantic ID linking trace to a dataset row (e.g. "q070")
    depth: int = 0
    user_input: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    turn_length: Optional[int] = 5

    turns: List[TurnRecord] = field(default_factory=list)
    final_response: str = ""
    total_turns: int = 0
    total_tool_calls: int = 0

    started_at: str = ""
    ended_at: str = ""
    duration_s: float = 0.0

    # Aggregated stats including sub-agents
    total_tool_calls_recursive: int = 0
    total_turns_recursive: int = 0
    total_sub_agents_spawned: int = 0
    # Aggregate token usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    # Chain analysis (pre-dispatch decomposition, root only)
    chain_plan: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full trace tree to a dict (JSON-safe)."""
        def _serialize(obj):
            if isinstance(obj, EpisodeTrace):
                d = {}
                for k, v in obj.__dict__.items():
                    d[k] = _serialize(v)
                return d
            elif isinstance(obj, (TurnRecord, ToolCallRecord)):
                d = {}
                for k, v in obj.__dict__.items():
                    d[k] = _serialize(v)
                return d
            elif isinstance(obj, list):
                return [_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            else:
                return obj
        return _serialize(self)

    def save(self, path: Optional[str] = None) -> str:
        """Save trace to JSON and HTML files inside TRACES_DIR. Returns the JSON file path."""
        if path is None:
            os.makedirs(TRACES_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(TRACES_DIR, f"trace_{timestamp}_{self.trace_id}.json")
        else:
            # Ensure parent directory exists for custom paths too
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        # Also emit an HTML version alongside the JSON
        html_path = path.replace(".json", ".html")
        html_str = render_trace_html(self.to_dict())
        with open(html_path, "w") as f:
            f.write(html_str)
        return html_path

    def compute_recursive_stats(self):
        """Walk the trace tree and compute aggregate stats."""
        self.total_tool_calls_recursive = self.total_tool_calls
        self.total_turns_recursive = self.total_turns
        self.total_sub_agents_spawned = 0
        self.total_prompt_tokens = sum(t.prompt_tokens for t in self.turns)
        self.total_completion_tokens = sum(t.completion_tokens for t in self.turns)
        self.total_tokens = sum(t.total_tokens for t in self.turns)

        for turn in self.turns:
            for tc in turn.tool_calls:
                if tc.child_trace is not None:
                    self.total_sub_agents_spawned += 1
                    tc.child_trace.compute_recursive_stats()
                    self.total_tool_calls_recursive += tc.child_trace.total_tool_calls_recursive
                    self.total_turns_recursive += tc.child_trace.total_turns_recursive
                    self.total_sub_agents_spawned += tc.child_trace.total_sub_agents_spawned
                    self.total_prompt_tokens += tc.child_trace.total_prompt_tokens
                    self.total_completion_tokens += tc.child_trace.total_completion_tokens
                    self.total_tokens += tc.child_trace.total_tokens

    def pretty_print(self, indent: int = 0):
        """Pretty-print the full trace tree."""
        prefix = "  " * indent
        depth_label = f"[depth={self.depth}]" if self.depth > 0 else "[root]"
        agent_label = "🤖 Sub-Agent" if self.depth > 0 else "🏁 Agent"

        print(f"{prefix}{'━' * 60}")
        print(f"{prefix}{agent_label} {depth_label}  trace_id={self.trace_id}")
        print(f"{prefix}  Input: {self.user_input[:100]}{'...' if len(self.user_input) > 100 else ''}")
        print(f"{prefix}  Duration: {self.duration_s:.2f}s | Turns: {self.total_turns} | Tool calls: {self.total_tool_calls}")
        print(f"{prefix}{'━' * 60}")

        for turn in self.turns:
            print(f"{prefix}  ┌─ Turn {turn.turn_number} ({turn.duration_s:.2f}s)")

            if turn.tool_calls:
                for tc in turn.tool_calls:
                    # Truncate args for display
                    args_str = json.dumps(tc.tool_args, default=str)
                    if len(args_str) > 120:
                        args_str = args_str[:120] + "..."
                    print(f"{prefix}  │  🔧 {tc.tool_name}({args_str}) [{tc.duration_s:.2f}s]")

                    # Truncate output for display
                    out_preview = tc.output.replace("\n", "\\n")
                    if len(out_preview) > 150:
                        out_preview = out_preview[:150] + "..."
                    print(f"{prefix}  │     → {out_preview}")

                    # Recurse into child trace
                    if tc.child_trace is not None:
                        tc.child_trace.pretty_print(indent=indent + 3)

            if turn.assistant_content:
                content_preview = turn.assistant_content.replace("\n", "\\n")
                if len(content_preview) > 200:
                    content_preview = content_preview[:200] + "..."
                print(f"{prefix}  │  💬 {content_preview}")

            print(f"{prefix}  └─")

        # Summary
        if self.depth == 0:
            self.compute_recursive_stats()
            root_turns = self.total_turns
            root_calls = self.total_tool_calls
            all_turns = self.total_turns_recursive
            all_calls = self.total_tool_calls_recursive
            sub_turns = all_turns - root_turns
            sub_calls = all_calls - root_calls
            print(f"{prefix}{'═' * 60}")
            print(f"{prefix}📊 Episode Summary:")
            print(f"{prefix}  Total duration:       {self.duration_s:.2f}s")
            print(f"{prefix}  Sub-agents spawned:   {self.total_sub_agents_spawned}")
            print(f"{prefix}")
            print(f"{prefix}  Turns  (root agent):  {root_turns}")
            print(f"{prefix}  Turns  (sub-agents):  {sub_turns}")
            print(f"{prefix}  Turns  (total):       {all_turns}")
            print(f"{prefix}")
            print(f"{prefix}  Tool calls (root):    {root_calls}")
            print(f"{prefix}  Tool calls (sub):     {sub_calls}")
            print(f"{prefix}  Tool calls (total):   {all_calls}")
            print(f"{prefix}")
            print(f"{prefix}  Prompt tokens:        {self.total_prompt_tokens:,}")
            print(f"{prefix}  Completion tokens:    {self.total_completion_tokens:,}")
            print(f"{prefix}  Total tokens:         {self.total_tokens:,}")
            print(f"{prefix}")
            print(f"{prefix}  Final response:       {(self.final_response or '(none)')[:200]}{'...' if self.final_response and len(self.final_response) > 200 else ''}")
            print(f"{prefix}{'═' * 60}")

    def to_html(self) -> str:
        """Generate a self-contained HTML page visualizing this trace."""
        d = self.to_dict()
        return render_trace_html(d)


# ─── HTML Rendering ────────────────────────────────────────────────────


_CSS = """\
:root {
  --white: #ffffff;
  --off: #f7f7f5;
  --border: #e4e4e0;
  --text: #222220;
  --text-mid: #666662;
  --text-light: #999994;
  --accent: #2a5caa;
  --accent-bg: #f0f4fb;
  --success: #2d8a56;
  --warn: #c9860a;
  --error: #c0392b;
  --depth-0: #e4e4e0;
  --depth-1: #c4c8f0;
  --depth-2: #c4e0c8;
  --purple: #7c3aed;
  --serif: "Fraunces", Georgia, serif;
  --sans: "Plus Jakarta Sans", system-ui, sans-serif;
  --mono: "JetBrains Mono", monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--off); color: var(--text); font-family: var(--sans);
  font-size: 15px; line-height: 1.6; font-weight: 300; -webkit-font-smoothing: antialiased;
}

/* ── Layout ── */
.viewer { display: grid; grid-template-columns: 220px 1fr; max-width: 1100px; margin: 0 auto; min-height: 100vh; }

/* ── Sidebar / Timeline ── */
.sidebar {
  border-right: 1px solid var(--border); background: var(--white);
  padding: 24px 0; position: sticky; top: 0; height: 100vh; overflow-y: auto;
}
.sidebar-header { padding: 0 20px 20px; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
.sidebar-title {
  font-family: var(--mono); font-size: 12px; color: var(--text-mid);
  display: flex; align-items: center; gap: 8px;
}
.sidebar-title .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--success); display: inline-block; }
.sidebar-meta { font-family: var(--mono); font-size: 10px; color: var(--text-light); margin-top: 8px; line-height: 1.6; }

.timeline { list-style: none; }
.tl-item {
  position: relative; padding: 8px 20px 8px 36px; cursor: pointer;
  transition: background 0.12s; font-family: var(--mono); font-size: 11px;
  color: var(--text-mid); border-left: 2px solid transparent;
}
.tl-item:hover { background: var(--off); }
.tl-item.active { background: var(--accent-bg); color: var(--accent); border-left-color: var(--accent); }
.tl-item::before {
  content: ""; position: absolute; left: 16px; top: 50%; transform: translateY(-50%);
  width: 6px; height: 6px; border-radius: 50%; border: 1.5px solid var(--border); background: var(--white);
}
.tl-item.active::before { border-color: var(--accent); background: var(--accent); }
.tl-item.sub { padding-left: 50px; font-size: 10px; }
.tl-item.sub::before { left: 30px; width: 5px; height: 5px; }
.tl-item.sub2 { padding-left: 64px; font-size: 10px; }
.tl-item.sub2::before { left: 44px; width: 4px; height: 4px; }
.tl-label {
  display: block; font-size: 9px; color: var(--text-light);
  letter-spacing: 0.1em; text-transform: uppercase; padding: 16px 20px 6px;
}

/* ── Main content ── */
.main { padding: 32px 40px; overflow-y: auto; }

/* ── Prompt banner ── */
.prompt-banner {
  background: var(--white); border: 1px solid var(--border); border-radius: 4px;
  padding: 16px 20px; margin-bottom: 20px;
}
.prompt-label { font-family: var(--mono); font-size: 9px; color: var(--text-light);
  text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.prompt-text { font-size: 14px; color: var(--text); line-height: 1.6; }

/* ── Trace header ── */
.trace-header {
  background: var(--white); border: 1px solid var(--border);
  border-radius: 4px; overflow: hidden; margin-bottom: 28px;
}
.trace-header-bar {
  padding: 14px 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.trace-header-left {
  font-family: var(--mono); font-size: 12px; color: var(--text-mid);
  display: flex; align-items: center; gap: 8px;
}
.trace-header-left .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); }
.trace-header-right { font-family: var(--mono); font-size: 10px; color: var(--text-light); }
.trace-stats { display: grid; grid-template-columns: repeat(5, 1fr); }
.trace-stat {
  padding: 18px 16px; border-right: 1px solid var(--border); text-align: center;
}
.trace-stat:last-child { border-right: none; }
.trace-stat-val {
  font-family: var(--serif); font-size: 26px; font-weight: 300;
  color: var(--text); display: block; line-height: 1; margin-bottom: 6px;
}
.trace-stat-lbl {
  font-family: var(--mono); font-size: 9px; color: var(--text-light);
  text-transform: uppercase; letter-spacing: 0.1em;
}

/* ── Token ratio bar (in stats header) ── */
.token-ratio-bar {
  display: flex; height: 4px; border-radius: 2px; overflow: hidden;
  margin-top: 10px; background: var(--off);
}
.token-ratio-in { background: var(--accent); }
.token-ratio-out { background: var(--success); }
.token-ratio-legend {
  display: flex; justify-content: center; gap: 12px; margin-top: 6px;
  font-family: var(--mono); font-size: 9px; color: var(--text-light);
}
.token-ratio-legend span { display: flex; align-items: center; gap: 4px; }
.token-ratio-legend .swatch {
  display: inline-block; width: 8px; height: 8px; border-radius: 2px;
}

/* ── Turn cards ── */
.turn-card {
  background: var(--white); border: 1px solid var(--border);
  border-radius: 4px; margin-bottom: 16px; overflow: hidden; transition: box-shadow 0.15s;
}
.turn-card:target, .turn-card.highlight { box-shadow: 0 0 0 2px var(--accent); }
.turn-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px; cursor: pointer; user-select: none; transition: background 0.12s;
}
.turn-header:hover { background: var(--off); }
.turn-header-left { display: flex; align-items: center; gap: 12px; }
.turn-badge { font-family: var(--mono); font-size: 10px; padding: 3px 8px; border-radius: 3px; letter-spacing: 0.03em; }
.badge-root { background: #f0f0ee; color: var(--text-mid); }
.badge-sub { background: var(--depth-1); color: #4a4aa0; }
.badge-sub2 { background: var(--depth-2); color: #2d6a3e; }
.badge-tool { background: #fef3cd; color: #856404; }
.badge-reasoning { background: #f3e8ff; color: var(--purple); }
.badge-error { background: #fde8e8; color: var(--error); }
.turn-title { font-family: var(--mono); font-size: 12px; color: var(--text); }
.turn-meta {
  font-family: var(--mono); font-size: 10px; color: var(--text-light);
  display: flex; align-items: center; gap: 12px;
}
.chevron {
  display: inline-block; width: 16px; height: 16px;
  color: var(--text-light); transition: transform 0.2s;
}
.turn-card.open .chevron { transform: rotate(180deg); }

.turn-body { display: none; border-top: 1px solid var(--border); }
.turn-card.open .turn-body { display: block; }

/* ── Sections inside a turn ── */
.turn-section { padding: 16px 20px; border-bottom: 1px solid var(--border); }
.turn-section:last-child { border-bottom: none; }
.turn-section-label {
  font-family: var(--mono); font-size: 9px; color: var(--text-light);
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px;
}

/* ── Reasoning chain ── */
.reasoning {
  font-size: 13px; color: var(--text-mid); line-height: 1.8;
  white-space: pre-wrap; font-family: var(--sans);
}

/* ── Tool call block ── */
.tool-block {
  background: var(--off); border: 1px solid var(--border); border-radius: 3px;
  overflow: hidden; margin-bottom: 8px;
}
.tool-block:last-child { margin-bottom: 0; }
.tool-block-header {
  padding: 8px 14px; font-family: var(--mono); font-size: 11px;
  color: var(--text-mid); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px; cursor: pointer;
}
.tool-block-header:hover { background: #f0f0ee; }
.tool-fn { color: var(--accent); }
.tool-block-body {
  padding: 12px 14px; font-family: var(--mono); font-size: 11px;
  color: var(--text-mid); line-height: 1.6; display: none;
}
.tool-block.open .tool-block-body { display: block; }
.tool-args-pre, .tool-output-pre {
  background: var(--white); padding: 10px; border-radius: 3px;
  font-family: var(--mono); font-size: 11px; overflow-x: auto;
  max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-word;
  border: 1px solid var(--border); margin-top: 6px;
}
.tool-result { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); color: var(--success); }
.tool-error { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); color: var(--error); }

/* ── Verifier metadata block ── */
.verifier-block {
  margin-top: 10px; padding: 10px 14px; border-radius: 3px;
  border: 1px solid var(--border); font-family: var(--mono); font-size: 11px;
}
.verifier-block.approved { background: #f0faf0; border-color: var(--success); }
.verifier-block.rejected { background: #fdf0f0; border-color: var(--error); }
.verifier-badge {
  display: inline-block; padding: 2px 8px; border-radius: 3px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; margin-bottom: 8px;
}
.verifier-badge.approved { background: var(--success); color: white; }
.verifier-badge.rejected { background: var(--error); color: white; }
.verifier-detail {
  margin-top: 6px; padding: 8px; background: var(--white);
  border: 1px solid var(--border); border-radius: 3px;
  max-height: 300px; overflow-y: auto; white-space: pre-wrap;
  word-break: break-word; font-size: 11px; color: var(--text-mid);
}
.verifier-section-label {
  font-size: 9px; color: var(--text-light); text-transform: uppercase;
  letter-spacing: 0.1em; margin-top: 8px; margin-bottom: 4px;
}

/* ── Spot-check metadata block ── */
.spotcheck-block {
  margin-top: 8px; padding: 10px 14px; border-radius: 3px;
  border: 1px solid var(--border); font-family: var(--mono); font-size: 11px;
}
.spotcheck-block.passed { background: #f0faf0; border-color: var(--success); }
.spotcheck-block.failed { background: #fdf0f0; border-color: var(--error); }
.spotcheck-block.challenged { background: #fff8f0; border-color: #e67e22; }
.spotcheck-block.justified { background: #f0f6fa; border-color: #3498db; }
.spotcheck-block.skipped { background: #fffcf0; border-color: #f1c40f; }
.spotcheck-badge {
  display: inline-block; padding: 2px 8px; border-radius: 3px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; margin-bottom: 8px;
}
.spotcheck-badge.passed { background: var(--success); color: white; }
.spotcheck-badge.failed { background: var(--error); color: white; }
.spotcheck-badge.challenged { background: #e67e22; color: white; }
.spotcheck-badge.justified { background: #3498db; color: white; }
.spotcheck-badge.skipped { background: #f1c40f; color: #333; }
.spotcheck-claims {
  margin-top: 6px; padding: 8px; background: var(--white);
  border: 1px solid var(--border); border-radius: 3px; font-size: 11px;
}
.spotcheck-claim-item {
  padding: 4px 0; border-bottom: 1px solid #eee; color: var(--text-mid);
}
.spotcheck-claim-item:last-child { border-bottom: none; }
.spotcheck-toggle {
  cursor: pointer; color: var(--accent); font-size: 10px;
  text-decoration: underline; margin-top: 4px; display: inline-block;
}

/* ── Draft card ── */
.draft-card {
  margin-top: 10px; padding: 14px 16px; border-radius: 4px;
  border: 1px solid #c8d6e5; background: #f8fafe;
  font-family: var(--sans); font-size: 13px; line-height: 1.6;
}
.draft-card.published {
  border-color: var(--success); background: #f0faf0;
}
.draft-card.rejected {
  border-color: var(--error); background: #fdf8f8;
}
.draft-badge {
  display: inline-block; padding: 2px 8px; border-radius: 3px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; margin-bottom: 8px;
  font-family: var(--mono);
}
.draft-badge.version { background: #c8d6e5; color: #2c3e50; }
.draft-badge.published { background: var(--success); color: white; }
.draft-badge.rejected { background: var(--error); color: white; }
.draft-meta {
  font-family: var(--mono); font-size: 10px; color: var(--text-light);
  margin-bottom: 8px;
}
.draft-content {
  padding: 10px 12px; background: var(--white); border: 1px solid var(--border);
  border-radius: 3px; max-height: 400px; overflow-y: auto;
  white-space: pre-wrap; word-break: break-word; font-size: 12px;
  color: var(--text); line-height: 1.65;
}
.draft-toggle {
  cursor: pointer; color: var(--accent); font-size: 10px;
  text-decoration: underline; margin-top: 6px; display: inline-block;
  font-family: var(--mono);
}

/* ── Sub-agent spawn ── */
.spawn-card {
  border-left: 3px solid var(--depth-1); padding: 12px 16px;
  background: #fafaff; border-radius: 0 3px 3px 0; margin-bottom: 8px; cursor: pointer;
}
.spawn-card:last-child { margin-bottom: 0; }
.spawn-card.depth2 { border-left-color: var(--depth-2); background: #f7fbf7; }
.spawn-card:hover { background: #f0f0ff; }
.spawn-label { font-family: var(--mono); font-size: 10px; color: var(--text-light); margin-bottom: 4px; }
.spawn-task { font-size: 13px; color: var(--text); }

/* ── Output / response ── */
.output-block {
  background: var(--off); border: 1px solid var(--border); border-radius: 3px;
  padding: 14px 16px; font-size: 13px; color: var(--text-mid); line-height: 1.7;
  white-space: pre-wrap; word-break: break-word;
}

/* ── Inline images ── */
.tc-img { margin-top: 8px; }
.tc-img img { max-width: 100%; border-radius: 6px; border: 1px solid var(--border); }

/* ── Tokens pill (dual segment) ── */
.tokens-pill {
  display: inline-flex; align-items: center; font-family: var(--mono); font-size: 10px;
  border-radius: 10px; overflow: hidden; border: 1px solid var(--border);
}
.tokens-pill .tok-in {
  padding: 2px 6px 2px 7px; background: var(--accent-bg); color: var(--accent);
}
.tokens-pill .tok-out {
  padding: 2px 7px 2px 6px; background: #edf7f0; color: var(--success);
}
.tokens-pill .tok-sep {
  width: 1px; background: var(--border); align-self: stretch;
}

/* ── Status indicator ── */
.status { display: inline-flex; align-items: center; gap: 5px; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; }
.status-ok .status-dot { background: var(--success); }
.status-warn .status-dot { background: var(--warn); }
.status-err .status-dot { background: var(--error); }

/* ── Final response ── */
.final-card {
  background: var(--white); border: 1px solid var(--border); border-radius: 4px;
  overflow: hidden; margin-top: 24px;
}
.final-header {
  padding: 14px 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono); font-size: 12px; color: var(--success);
}
.final-body { padding: 16px 20px; }

/* ── Rendered markdown in final response ── */
.final-rendered {
  font-size: 14px; line-height: 1.7; color: var(--text);
}
.final-rendered h1 { font-size: 22px; font-weight: 600; margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.final-rendered h2 { font-size: 18px; font-weight: 600; margin: 20px 0 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.final-rendered h3 { font-size: 15px; font-weight: 600; margin: 16px 0 8px; }
.final-rendered p { margin: 8px 0; }
.final-rendered ul, .final-rendered ol { margin: 8px 0; padding-left: 24px; }
.final-rendered li { margin: 4px 0; }
.final-rendered a { color: var(--accent); text-decoration: none; }
.final-rendered a:hover { text-decoration: underline; }
.final-rendered code { background: var(--off); padding: 2px 5px; border-radius: 3px; font-family: var(--mono); font-size: 12px; }
.final-rendered pre { background: var(--off); border: 1px solid var(--border); border-radius: 4px; padding: 12px; overflow-x: auto; }
.final-rendered pre code { background: none; padding: 0; }
.final-rendered blockquote { border-left: 3px solid var(--accent); margin: 8px 0; padding: 4px 16px; color: var(--text-mid); }
.final-rendered table { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 13px; }
.final-rendered th, .final-rendered td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; }
.final-rendered th { background: var(--off); font-weight: 600; }
.final-rendered strong { font-weight: 600; }

/* ── Responsive ── */
@media (max-width: 768px) {
  .viewer { grid-template-columns: 1fr; }
  .sidebar { position: relative; height: auto; border-right: none; border-bottom: 1px solid var(--border); }
  .main { padding: 24px 20px; }
  .trace-stats { grid-template-columns: repeat(3, 1fr); }
}

/* ── Chain analysis panel ── */
.chain-panel {
  background: var(--white); border: 1px solid var(--border);
  border-radius: 4px; margin-bottom: 20px; overflow: hidden;
}
.chain-panel-header {
  padding: 14px 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  cursor: pointer; user-select: none;
}
.chain-panel-header:hover { background: var(--off); }
.chain-panel-title {
  font-family: var(--mono); font-size: 12px; color: var(--text-mid);
  display: flex; align-items: center; gap: 8px;
}
.chain-panel-body { padding: 20px; }
.chain-panel-body.collapsed { display: none; }

/* Chain steps */
.chain-steps { list-style: none; padding: 0; margin: 0; }
.chain-step {
  position: relative; padding: 12px 16px 12px 44px; margin-bottom: 8px;
  border: 1px solid var(--border); border-radius: 4px; background: var(--off);
  transition: all 0.15s;
}
.chain-step:last-child { margin-bottom: 0; }
.chain-step::before {
  content: ""; position: absolute; left: 16px; top: 50%;
  transform: translateY(-50%);
  width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid var(--border); background: var(--white);
}
.chain-step.resolved { border-color: var(--success); background: #f0faf0; }
.chain-step.resolved::before { border-color: var(--success); background: var(--success); }
.chain-step.unlocked { border-color: var(--accent); background: var(--accent-bg); }
.chain-step.unlocked::before { border-color: var(--accent); background: var(--accent-bg); }
.chain-step.locked { opacity: 0.65; }
.chain-step.locked::before { border-color: var(--text-light); background: var(--off); }

.chain-step-num {
  font-family: var(--mono); font-size: 10px; color: var(--text-light);
  letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;
}
.chain-step-lookup { font-size: 13px; color: var(--text); line-height: 1.5; }
.chain-step-dep {
  font-family: var(--mono); font-size: 10px; color: var(--text-light);
  margin-top: 4px;
}
.chain-step-result {
  margin-top: 6px; padding: 6px 10px; border-radius: 3px;
  background: var(--white); border: 1px solid var(--success);
  font-family: var(--mono); font-size: 11px; color: var(--success);
}
/* Connector line between steps */
.chain-step-connector {
  position: absolute; left: 24px; top: -8px;
  width: 2px; height: 8px; background: var(--border);
}
.chain-step:first-child .chain-step-connector { display: none; }

/* Parallel tasks */
.chain-parallel { margin-top: 16px; }
.chain-parallel-title {
  font-family: var(--mono); font-size: 10px; color: var(--text-light);
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;
}
.chain-parallel-item {
  padding: 8px 12px; background: var(--off); border: 1px solid var(--border);
  border-radius: 3px; margin-bottom: 4px; font-size: 12px; color: var(--text-mid);
}
.chain-parallel-item:last-child { margin-bottom: 0; }

/* No-chain badge */
.chain-no-chain {
  font-family: var(--mono); font-size: 11px; color: var(--text-mid);
  padding: 8px 0;
}
.chain-badge {
  display: inline-block; padding: 2px 8px; border-radius: 3px;
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  letter-spacing: 0.05em; text-transform: uppercase;
}
.chain-badge.has-chain { background: #e8f0fe; color: var(--accent); }
.chain-badge.no-chain { background: #f0f0ee; color: var(--text-mid); }

/* ── Inline chain progress tracker (per root turn) ── */
.chain-progress {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  padding: 10px 16px; margin: 0 -16px 8px;
  background: var(--off); border-bottom: 1px solid var(--border);
  border-radius: 0;
}
.chain-progress-label {
  font-family: var(--mono); font-size: 10px; color: var(--text-light);
  letter-spacing: 0.05em; text-transform: uppercase; margin-right: 4px;
}
.chain-pill {
  display: inline-flex; align-items: center; gap: 3px;
  padding: 2px 8px; border-radius: 12px;
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  border: 1px solid var(--border); background: var(--white);
  transition: all 0.3s ease;
}
.chain-pill.locked { color: var(--text-light); opacity: 0.6; }
.chain-pill.unlocked { color: var(--accent); border-color: var(--accent); background: var(--accent-bg); }
.chain-pill.resolved { color: var(--success); border-color: var(--success); background: #f0faf0; }
.chain-pill.just-resolved {
  color: var(--success); border-color: var(--success); background: #d4edda;
  font-weight: 600; box-shadow: 0 0 0 2px rgba(45,138,86,0.2);
  animation: chain-resolve-pulse 0.6s ease-out;
}
@keyframes chain-resolve-pulse {
  0% { transform: scale(1.15); box-shadow: 0 0 0 4px rgba(45,138,86,0.3); }
  100% { transform: scale(1); box-shadow: 0 0 0 2px rgba(45,138,86,0.2); }
}
.chain-pill.contested {
  color: var(--error); border-color: var(--error); background: #fde8e8;
  font-weight: 600; box-shadow: 0 0 0 2px rgba(204,51,51,0.2);
  animation: chain-contest-pulse 0.6s ease-out;
  text-decoration: line-through;
}
@keyframes chain-contest-pulse {
  0% { transform: scale(1.15); box-shadow: 0 0 0 4px rgba(204,51,51,0.3); }
  100% { transform: scale(1); box-shadow: 0 0 0 2px rgba(204,51,51,0.2); }
}

/* ── Collapsible sub-agent groups ── */
.sub-agent-group {
  margin: -4px 0 16px 28px; border-left: 3px solid var(--depth-1);
  border-radius: 0 4px 4px 0;
  position: relative;
}
/* Connecting line from root card down to the sub-agent toggle */
.sub-agent-group::before {
  content: "";
  position: absolute; left: -17px; top: -12px;
  width: 14px; height: 24px;
  border-left: 2px solid var(--depth-1);
  border-bottom: 2px solid var(--depth-1);
  border-radius: 0 0 0 6px;
}
.sub-agent-group-summary {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 16px; cursor: pointer; user-select: none;
  font-family: var(--mono); font-size: 11px; color: var(--text-mid);
  background: var(--white); border: 1px solid var(--border);
  border-left: none; border-radius: 0 4px 4px 0;
  transition: background 0.15s;
}
.sub-agent-group-summary:hover { background: var(--off); }
.sub-agent-group-summary::-webkit-details-marker { display: none; }
.sub-agent-group-summary::marker { display: none; content: ""; }
.sub-agent-group-icon {
  display: inline-block; transition: transform 0.2s ease;
  font-size: 9px; color: var(--text-light);
}
.sub-agent-group[open] .sub-agent-group-icon { transform: rotate(90deg); }
.sub-agent-group-body {
  padding: 8px 0 8px 8px;
  background: linear-gradient(90deg, rgba(196,200,240,0.08) 0%, transparent 50%);
}
.sub-agent-group-body .turn-card {
  margin-bottom: 8px; border-left: 2px solid var(--depth-1);
  font-size: 14px;
}

/* ── Timeline sidebar enhancements ── */
.tl-tool {
  font-size: 9px; color: var(--text-light); display: block;
  margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 140px;
}
.tl-sub-count {
  display: inline-block; padding: 0 4px; border-radius: 8px;
  background: var(--depth-1); color: var(--text-mid);
  font-size: 9px; font-weight: 600; margin-left: 4px;
  vertical-align: middle;
}
.tl-chain-dots {
  display: inline-flex; gap: 2px; margin-left: 4px; vertical-align: middle;
}
.tl-chain-dot { font-size: 8px; line-height: 1; }
.tl-chain-dot.resolved { color: var(--success); }
.tl-chain-dot.pending { color: var(--text-light); }
"""

_JS = """\
function toggle(header) {
  header.closest('.turn-card').classList.toggle('open');
}

function toggleTool(header) {
  header.closest('.tool-block').classList.toggle('open');
}

function toggleChainPanel(header) {
  var body = header.nextElementSibling;
  body.classList.toggle('collapsed');
  var chevron = header.querySelector('.chevron');
  if (chevron) chevron.style.transform = body.classList.contains('collapsed') ? '' : 'rotate(180deg)';
}

function scrollToTurn(id) {
  var el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  el.classList.add('highlight');
  setTimeout(function() { el.classList.remove('highlight'); }, 1200);
}

/* Sidebar click handlers + active tracking */
document.addEventListener('DOMContentLoaded', function() {
  var items = document.querySelectorAll('.tl-item[data-target]');
  items.forEach(function(item) {
    item.addEventListener('click', function() {
      items.forEach(function(i) { i.classList.remove('active'); });
      item.classList.add('active');
      scrollToTurn(item.getAttribute('data-target'));
    });
  });

  /* Keyboard navigation */
  var cards = [];
  var idx = -1;
  document.querySelectorAll('.turn-card').forEach(function(c) { cards.push(c); });
  document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); idx = Math.min(idx + 1, cards.length - 1); cards[idx].scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); idx = Math.max(idx - 1, 0); cards[idx].scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    if ((e.key === 'Enter' || e.key === ' ') && idx >= 0) { e.preventDefault(); var h = cards[idx].querySelector('.turn-header'); if (h) toggle(h); }
  });

  /* Render final response as Markdown */
  var mdSrc = document.getElementById('final-md');
  var mdDst = document.getElementById('final-rendered');
  if (mdSrc && mdDst) {
    var raw = mdSrc.textContent;
    if (typeof marked !== 'undefined') {
      mdDst.innerHTML = marked.parse(raw);
    } else {
      /* marked.js didn't load (offline) — show as preformatted text */
      var pre = document.createElement('pre');
      pre.className = 'output-block';
      pre.textContent = raw;
      mdDst.appendChild(pre);
    }
  }
});
"""


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html_mod.escape(str(text))


def _extract_images_from_output(output: str) -> list:
    """Pull base64 image data from FETCHED FILES sections."""
    imgs = []
    for m in re.finditer(
        r"--- (.+?\.(png|jpg|jpeg|gif|svg|pdf)) \(base64\) ---\n(.+?)(?:\n---|$)",
        output, re.DOTALL
    ):
        imgs.append({"filename": m.group(1), "data": m.group(3).strip()})
    return imgs


def _collect_all_images(trace_dict: dict) -> list:
    """Recursively collect all images from every tool call in a trace."""
    imgs = []
    for turn in trace_dict.get("turns", []):
        for tc in turn.get("tool_calls", []):
            imgs.extend(_extract_images_from_output(tc.get("output", "")))
            if tc.get("child_trace"):
                imgs.extend(_collect_all_images(tc["child_trace"]))
    return imgs


def _flatten_trace(trace_dict: dict, depth: int = 0, counter: list | None = None) -> list:
    """Flatten a nested trace into a sequential list of card descriptors.

    Each card is a dict with keys:
        type     – "turn"
        id       – unique HTML id like "card-1"
        depth    – nesting depth (0 = root)
        turn     – the original turn dict
        turn_num – human-visible turn number within its agent
        agent_label – e.g. "Root", "Sub-Agent #1"
    Sub-agent turns are interleaved right after the parent turn that spawned them.
    """
    if counter is None:
        counter = [0, 0]  # [card_counter, sub_agent_counter]
    cards: list[dict] = []
    turns = trace_dict.get("turns", [])
    if depth == 0:
        agent_label = "Root"
    else:
        counter[1] += 1
        agent_label = f"Sub-Agent #{counter[1]}"

    for turn in turns:
        counter[0] += 1
        card_id = f"card-{counter[0]}"
        cards.append({
            "type": "turn",
            "id": card_id,
            "depth": depth,
            "turn": turn,
            "turn_num": turn.get("turn_number", "?"),
            "agent_label": agent_label,
        })
        # Inline sub-agent turns immediately after the parent turn
        for tc in turn.get("tool_calls", []):
            if tc.get("child_trace"):
                cards.extend(_flatten_trace(tc["child_trace"], depth + 1, counter))

    return cards


def _group_by_root_turn(flat_cards: list) -> list:
    """Group flat cards into root-turn groups.

    Returns a list of groups, where each group is:
        {"root": card_dict, "children": [card_dict, ...]}
    A root card has depth==0; children are all subsequent cards until
    the next root card.  If there are orphan sub-agent cards before any
    root (shouldn't happen), they get a synthetic root.
    """
    groups: list[dict] = []
    current_group: Optional[dict] = None

    for card in flat_cards:
        if card["depth"] == 0:
            if current_group is not None:
                groups.append(current_group)
            current_group = {"root": card, "children": []}
        else:
            if current_group is None:
                # Orphan — shouldn't happen but handle gracefully
                current_group = {"root": card, "children": []}
            else:
                current_group["children"].append(card)

    if current_group is not None:
        groups.append(current_group)

    return groups


def _jaccard_words(a: str, b: str) -> float:
    """Compute Jaccard word-overlap similarity between two strings.

    Normalises by lowercasing and stripping common punctuation so that
    JSON artefacts and parenthesised text don't prevent matches.
    """
    import re as _re
    _punct = _re.compile(r'[^a-z0-9\s]')
    wa = set(_punct.sub('', a.lower()).split())
    wb = set(_punct.sub('', b.lower()).split())
    wa.discard('')
    wb.discard('')
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _containment_ratio(query: str, target: str) -> float:
    """Fraction of *query* words found in *target*.

    Better than Jaccard when query is short and target is long (e.g.
    matching a chain step lookup against a full tool call output).
    """
    import re as _re
    _punct = _re.compile(r'[^a-z0-9\s]')
    wq = set(_punct.sub('', query.lower()).split())
    wt = set(_punct.sub('', target.lower()).split())
    wq -= {'', 'the', 'a', 'an', 'of', 'or', 'and', 'in', 'for', 'to', 'is',
            'eg', 'that', 'this', 'with', 'from', 'by', 'on', 'at', 'as'}
    if not wq:
        return 0.0
    return len(wq & wt) / len(wq)


def _extract_tool_text(tc: dict) -> str:
    """Extract meaningful text from a tool call for chain matching.

    Combines the task text with the output response content, parsing
    JSON outputs to extract the 'response' field when possible.
    """
    task = tc.get("tool_args", {}).get("task", "") if isinstance(tc.get("tool_args"), dict) else ""
    raw_output = tc.get("output", "")

    output_text = raw_output
    if isinstance(raw_output, str) and raw_output.startswith("{"):
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                output_text = parsed.get("response", raw_output)
        except (json.JSONDecodeError, ValueError):
            pass
    elif isinstance(raw_output, dict):
        output_text = raw_output.get("response", str(raw_output))

    return f"{task} {output_text}"


def _synthesize_chain_snapshots(groups: list, chain_plan: dict) -> None:
    """Retroactively inject chain_snapshot into root turns for legacy traces.

    When a trace has chain_plan but was recorded before per-turn snapshot
    capture was added, this function reconstructs approximate chain progress
    by examining tool call tasks/outputs and comparing them against chain step
    lookup text using Jaccard word overlap.

    Mutates group root turn dicts in place.
    """
    chain_steps = chain_plan.get("chain_steps", [])
    if not chain_steps:
        return

    # Build ordered list of step metadata
    steps_meta = []
    for cs in chain_steps:
        steps_meta.append({
            "step": cs.get("step", 0),
            "lookup": cs.get("lookup", ""),
            "depends_on": cs.get("depends_on"),
            "placeholder": cs.get("placeholder", ""),
            "resolved_value": cs.get("resolved_value"),  # final state from chain_plan
        })

    # Track which steps have been "addressed" as we walk through turns.
    # A step is considered addressed when a root turn's conduct_research task
    # or output has significant word overlap with the step's lookup text.
    MATCH_THRESHOLD = 0.50  # containment: ≥50% of lookup words in tool text
    addressed: dict[int, Optional[str]] = {}  # step_num -> summary or None

    for group in groups:
        root_turn = group["root"]["turn"]

        # Check tool calls in this root turn for chain step matches
        for tc in root_turn.get("tool_calls", []):
            tool_name = tc.get("tool_name", "")
            if tool_name != "conduct_research":
                continue
            combined = _extract_tool_text(tc)

            for sm in steps_meta:
                step_num = sm["step"]
                if step_num in addressed:
                    continue
                # Check if dependencies are met
                dep = sm.get("depends_on")
                if dep is not None and dep not in addressed:
                    continue
                similarity = _containment_ratio(sm["lookup"], combined)
                if similarity >= MATCH_THRESHOLD:
                    # Extract brief summary from parsed output
                    raw_out = tc.get("output", "")
                    if isinstance(raw_out, str) and raw_out.startswith("{"):
                        try:
                            summary = json.loads(raw_out).get("response", raw_out)[:120]
                        except (json.JSONDecodeError, ValueError):
                            summary = raw_out[:120]
                    else:
                        summary = str(raw_out)[:120]
                    if len(summary) >= 120:
                        summary += "…"
                    addressed[step_num] = summary or "(addressed)"

        # Build snapshot reflecting current state at this turn
        snapshot = []
        for sm in steps_meta:
            step_num = sm["step"]
            snapshot.append({
                "step": sm["step"],
                "lookup": sm["lookup"],
                "depends_on": sm["depends_on"],
                "placeholder": sm["placeholder"],
                "resolved_value": addressed.get(step_num),
            })

        # Inject into the turn dict so downstream renderers pick it up
        root_turn["chain_snapshot"] = snapshot


_CHEVRON_SVG = (
    '<svg class="chevron" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>'
)


def _depth_badge(depth: int) -> str:
    cls = "badge-root" if depth == 0 else ("badge-sub" if depth == 1 else "badge-sub2")
    label = "root" if depth == 0 else (f"sub" if depth == 1 else f"sub-{depth}")
    return f'<span class="turn-badge {cls}">{label}</span>'


def _render_verification_meta(meta: dict) -> list[str]:
    """Render verifier + spot-check metadata blocks. Returns HTML fragments."""
    parts: list[str] = []
    verdict = meta.get("verification_verdict", "")
    response = meta.get("verification_response", "")
    link_report = meta.get("link_check_report", "")
    attempt = meta.get("verification_attempt", "")

    if verdict == "APPROVED":
        v_cls = "approved"
    elif verdict in ("REVISION_NEEDED", "SPOT_CHECK_FAILED"):
        v_cls = "rejected"
    else:
        v_cls = ""
    badge_label = verdict or "VERIFIED"
    attempt_str = f" (attempt {attempt})" if attempt else ""

    parts.append(f'<div class="verifier-block {v_cls}">')
    parts.append(f'<span class="verifier-badge {v_cls}">\U0001f50d {_esc(badge_label)}{_esc(attempt_str)}</span>')

    if response:
        parts.append(f'<div class="verifier-section-label">Verifier Response</div>')
        parts.append(f'<div class="verifier-detail">{_esc(response)}</div>')

    if link_report:
        parts.append(f'<div class="verifier-section-label">Link Check</div>')
        parts.append(f'<div class="verifier-detail">{_esc(link_report)}</div>')

    parts.append('</div>')  # .verifier-block

    # ── Spot-check metadata ───────────────────────────────────────
    sc = meta.get("spot_check")
    if sc:
        sc_verdict = sc.get("spot_check_verdict", "")
        sc_skipped = sc.get("spot_check_skipped_reason", "")
        sc_claims_n = sc.get("claims_checked", 0)
        sc_extract = sc.get("extract_response", "")
        sc_compare = sc.get("compare_response", "")
        sc_refusal = sc.get("refusal_challenge_response", "")
        sc_degraded = sc.get("degraded_claims", 0)

        if sc_verdict == "PASSED":
            sc_cls = "passed"
        elif sc_verdict in ("FAILED", "SPOT_CHECK_FAILED"):
            sc_cls = "failed"
        elif sc_verdict == "REFUSAL_CHALLENGED":
            sc_cls = "challenged"
        elif sc_verdict == "REFUSAL_JUSTIFIED":
            sc_cls = "justified"
        elif sc_skipped:
            sc_cls = "skipped"
        else:
            sc_cls = ""

        sc_label = sc_verdict or (f"SKIPPED ({sc_skipped})" if sc_skipped else "RAN")
        parts.append(f'<div class="spotcheck-block {sc_cls}">')
        parts.append(f'<span class="spotcheck-badge {sc_cls}">\U0001f9ea Spot-Check: {_esc(sc_label)}</span>')

        if sc_claims_n:
            parts.append(f'<div class="verifier-section-label">Claims Checked: {sc_claims_n}</div>')
        if sc_degraded:
            parts.append(f'<div class="verifier-section-label">Degraded Evidence: {sc_degraded}/{sc_claims_n}</div>')

        if sc_extract:
            _sc_ext_id = f"sc-extract-{id(sc)}"
            parts.append(f'<div class="verifier-section-label">Claim Extraction</div>')
            parts.append(f'<span class="spotcheck-toggle" onclick="var e=document.getElementById(\'{_sc_ext_id}\');e.style.display=e.style.display===\'none\'?\'block\':\'none\'">show/hide</span>')
            parts.append(f'<div id="{_sc_ext_id}" class="verifier-detail" style="display:none">{_esc(sc_extract)}</div>')

        if sc_compare:
            _sc_cmp_id = f"sc-compare-{id(sc)}"
            parts.append(f'<div class="verifier-section-label">Claim Comparison</div>')
            parts.append(f'<span class="spotcheck-toggle" onclick="var e=document.getElementById(\'{_sc_cmp_id}\');e.style.display=e.style.display===\'none\'?\'block\':\'none\'">show/hide</span>')
            parts.append(f'<div id="{_sc_cmp_id}" class="verifier-detail" style="display:none">{_esc(sc_compare)}</div>')

        if sc_refusal:
            _sc_ref_id = f"sc-refusal-{id(sc)}"
            parts.append(f'<div class="verifier-section-label">Refusal Challenge</div>')
            parts.append(f'<span class="spotcheck-toggle" onclick="var e=document.getElementById(\'{_sc_ref_id}\');e.style.display=e.style.display===\'none\'?\'block\':\'none\'">show/hide</span>')
            parts.append(f'<div id="{_sc_ref_id}" class="verifier-detail" style="display:none">{_esc(sc_refusal)}</div>')

        # Chain contestation display
        sc_contested = sc.get("chain_contested", [])
        if sc_contested:
            parts.append(f'<div class="verifier-section-label" style="color:var(--error)">'
                         f'⚠️ Chain Steps Contested: {len(sc_contested)}</div>')
            for cr in sc_contested:
                step_num = cr.get("step", "?")
                old_val = _esc(str(cr.get("old_value", "?"))[:100])
                reason = cr.get("reason", "")
                reason_label = " (cascade)" if "cascade" in reason else ""
                parts.append(
                    f'<div class="verifier-detail" style="color:var(--error);padding:2px 0">'
                    f'Step {step_num}: <s>{old_val}</s> — cleared{reason_label}'
                    f'</div>'
                )

        parts.append('</div>')  # .spotcheck-block

    return parts


def _render_tool_block(tc: dict) -> str:
    """Render a single tool call as a collapsible tool-block."""
    tname = _esc(tc.get("tool_name", "?"))
    targs = tc.get("tool_args", {})
    toutput = tc.get("output", "")
    tc_dur = tc.get("duration_s", 0)

    args_json = json.dumps(targs, indent=2, default=str)

    # Strip base64 blobs from display
    display_output = re.sub(
        r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
        r"\1[base64 data omitted — see image below]\2",
        toutput, flags=re.DOTALL,
    )

    parts = [f'<div class="tool-block">']
    parts.append(
        f'<div class="tool-block-header" onclick="toggleTool(this)">'
        f'{_CHEVRON_SVG}'
        f'<span class="tool-fn">{tname}</span>'
        f'<span style="margin-left:auto;font-size:10px;color:var(--text-light);">{tc_dur:.2f}s</span>'
        f'</div>'
    )
    parts.append('<div class="tool-block-body">')

    # ── refine_draft: styled draft card ──────────────────────────────
    if tname == "refine_draft":
        draft_content = targs.get("content", "")
        # Parse version / char count from output: "✅ Draft v3 saved (1,639 chars)."
        _ver_m = re.search(r"Draft v(\d+) saved \(([^)]+)\)", toutput)
        is_rejected = toutput.startswith("Draft too short")

        if _ver_m:
            ver_num, char_info = _ver_m.group(1), _ver_m.group(2)
            parts.append('<div class="draft-card">')
            parts.append(f'<span class="draft-badge version">\U0001f4dd Draft v{_esc(ver_num)}</span>')
            parts.append(f'<span class="draft-meta">{_esc(char_info)}</span>')
        elif is_rejected:
            parts.append('<div class="draft-card rejected">')
            parts.append(f'<span class="draft-badge rejected">\u26a0\ufe0f Rejected</span>')
            parts.append(f'<span class="draft-meta">{_esc(toutput.split(chr(10))[0])}</span>')
        else:
            parts.append('<div class="draft-card">')
            parts.append('<span class="draft-badge version">\U0001f4dd Draft</span>')

        if draft_content:
            _dc_id = f"draft-{id(tc)}"
            preview = draft_content[:500]
            parts.append(f'<div class="draft-content">{_esc(preview)}')
            if len(draft_content) > 500:
                parts.append(f'<span id="{_dc_id}" style="display:none">{_esc(draft_content[500:])}</span>')
                parts.append(f'<span class="draft-toggle" onclick="var e=document.getElementById(\'{_dc_id}\');if(e.style.display===\'none\'){{e.style.display=\'inline\';this.textContent=\'collapse\'}}else{{e.style.display=\'none\';this.textContent=\'show full draft\'}}">&hellip; show full draft</span>')
            parts.append('</div>')  # .draft-content
        parts.append('</div>')  # .draft-card

    # ── research_complete: published / rejected draft + verification ──
    elif tname == "research_complete":
        meta = tc.get("metadata", {})
        verdict = meta.get("verification_verdict", "")
        is_approved = verdict == "APPROVED"

        if is_approved:
            parts.append('<div class="draft-card published">')
            parts.append('<span class="draft-badge published">\u2705 Published</span>')
            _dc_id = f"draft-pub-{id(tc)}"
            preview = toutput[:600]
            parts.append(f'<div class="draft-content">{_esc(preview)}')
            if len(toutput) > 600:
                parts.append(f'<span id="{_dc_id}" style="display:none">{_esc(toutput[600:])}</span>')
                parts.append(f'<span class="draft-toggle" onclick="var e=document.getElementById(\'{_dc_id}\');if(e.style.display===\'none\'){{e.style.display=\'inline\';this.textContent=\'collapse\'}}else{{e.style.display=\'none\';this.textContent=\'show full draft\'}}">&hellip; show full draft</span>')
            parts.append('</div>')  # .draft-content
            parts.append('</div>')  # .draft-card
        else:
            badge_text = '\u274c Revision Needed' if verdict == 'REVISION_NEEDED' else f'\u26a0\ufe0f {_esc(verdict or "Reviewing")}'
            parts.append('<div class="draft-card rejected">')
            parts.append(f'<span class="draft-badge rejected">{badge_text}</span>')
            if toutput.strip():
                parts.append(f'<div class="draft-content">{_esc(toutput)}</div>')
            parts.append('</div>')  # .draft-card

        # Verification metadata (always present for research_complete)
        if meta:
            parts.extend(_render_verification_meta(meta))

    # ── Generic tool rendering ───────────────────────────────────────
    else:
        # Arguments
        parts.append(f'<div style="font-size:9px;color:var(--text-light);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Arguments</div>')
        parts.append(f'<pre class="tool-args-pre">{_esc(args_json)}</pre>')

        # Output / result
        if display_output.strip():
            _lower_head = display_output.lower()[:80]
            is_error = (
                _lower_head.startswith("error:") or
                _lower_head.startswith("err:") or
                "traceback" in _lower_head
            )
            result_cls = "tool-error" if is_error else "tool-result"
            parts.append(f'<div class="{result_cls}">')
            parts.append(f'<div style="font-size:9px;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">{"Error" if is_error else "Result"}</div>')
            parts.append(f'<pre class="tool-output-pre">{_esc(display_output)}</pre>')
            parts.append('</div>')

    parts.append('</div>')  # .tool-block-body

    # Inline images
    for img in _extract_images_from_output(toutput):
        fname = _esc(img["filename"])
        ext = img["filename"].rsplit(".", 1)[-1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        parts.append(
            f'<div class="tc-img" style="padding:8px 14px;">'
            f'<div style="font-size:10px;color:var(--text-light);margin-bottom:4px;">\U0001f4ce {fname}</div>'
            f'<img src="data:{mime};base64,{img["data"]}" alt="{fname}"/>'
            f'</div>'
        )

    parts.append('</div>')  # .tool-block
    return "\n".join(parts)


def _render_spawn_card(tc: dict, depth: int) -> str:
    """Render a spawn-agent tool call as a clickable spawn card."""
    targs = tc.get("tool_args", {})
    task = _esc(targs.get("task", targs.get("prompt", "sub-agent")))
    child_trace = tc.get("child_trace", {})
    child_id = child_trace.get("trace_id", "")
    n_turns = len(child_trace.get("turns", []))
    dur = child_trace.get("duration_s", 0)
    depth_cls = "depth2" if depth >= 2 else ""
    return (
        f'<div class="spawn-card {depth_cls}" title="Click to scroll to sub-agent turns">'
        f'<div class="spawn-label">SPAWNED SUB-AGENT &middot; {n_turns} turns &middot; {dur:.1f}s</div>'
        f'<div class="spawn-task">{task[:300]}</div>'
        f'</div>'
    )


def _render_turn_card(card: dict) -> str:
    """Render a single flattened card as a turn-card HTML block."""
    turn = card["turn"]
    depth = card["depth"]
    card_id = card["id"]
    turn_num = card["turn_num"]
    agent_label = card["agent_label"]

    td_s = turn.get("duration_s", 0)
    tc_list = turn.get("tool_calls", [])
    tc_count = len(tc_list)
    prompt_tok = turn.get("prompt_tokens", 0)
    comp_tok = turn.get("completion_tokens", 0)

    # Check for reasoning
    raw_msg = turn.get("raw_assistant_message", {}) or {}
    reasoning = (raw_msg.get("reasoning_content", "") or "").strip()

    # Build badges
    badges = [_depth_badge(depth)]
    if tc_count:
        badges.append(f'<span class="turn-badge badge-tool">{tc_count} tool{"s" if tc_count != 1 else ""}</span>')
    if reasoning:
        badges.append('<span class="turn-badge badge-reasoning">reasoning</span>')

    # Title
    title = f"Turn {turn_num}"
    if depth > 0:
        title = f"{agent_label} — Turn {turn_num}"

    parts = [f'<div class="turn-card" id="{card_id}">']

    # Header
    parts.append(
        f'<div class="turn-header" onclick="toggle(this)">'
        f'<div class="turn-header-left">'
        f'{"".join(badges)}'
        f'<span class="turn-title">{title}</span>'
        f'</div>'
        f'<div class="turn-meta">'
        f'<span class="tokens-pill">'
        f'<span class="tok-in">\u2193{prompt_tok:,}</span>'
        f'<span class="tok-sep"></span>'
        f'<span class="tok-out">\u2191{comp_tok:,}</span>'
        f'</span>'
        f'<span>{td_s:.2f}s</span>'
        f'{_CHEVRON_SVG}'
        f'</div>'
        f'</div>'
    )

    # Body
    parts.append('<div class="turn-body">')

    # Reasoning section
    if reasoning:
        parts.append(
            f'<div class="turn-section">'
            f'<div class="turn-section-label">\U0001f9e0 Reasoning</div>'
            f'<div class="reasoning">{_esc(reasoning)}</div>'
            f'</div>'
        )

    # Tool calls section
    if tc_list:
        parts.append('<div class="turn-section">')
        parts.append('<div class="turn-section-label">\U0001f6e0\ufe0f Tool Calls</div>')
        for tc in tc_list:
            if tc.get("child_trace"):
                parts.append(_render_spawn_card(tc, depth + 1))
            parts.append(_render_tool_block(tc))
        parts.append('</div>')

    # Assistant text content
    acontent = turn.get("assistant_content", "")
    if acontent:
        parts.append(
            f'<div class="turn-section">'
            f'<div class="turn-section-label">\U0001f4ac Response</div>'
            f'<div class="output-block">{_esc(acontent[:2000])}</div>'
            f'</div>'
        )

    parts.append('</div>')  # .turn-body
    parts.append('</div>')  # .turn-card
    return "\n".join(parts)


def _render_chain_progress(chain_snapshot: Optional[list], prev_snapshot: Optional[list] = None) -> str:
    """Render a compact inline chain progress tracker for a root turn card.

    Shows each step as a small pill: 🔒 locked, 🔓 unlocked, ✅ resolved.
    If a step just resolved this turn (wasn't resolved in prev_snapshot), highlight it.
    If a step was resolved in prev_snapshot but is now cleared (contested by
    spot-check), show it with a ⚠️ contested indicator.
    Returns empty string if no chain data.
    """
    if not chain_snapshot:
        return ""

    # Build previous resolution state for diff detection
    prev_resolved: dict[int, str] = {}  # step_num -> resolved_value
    if prev_snapshot:
        for s in prev_snapshot:
            rv = s.get("resolved_value")
            if rv:
                prev_resolved[s.get("step", 0)] = rv

    pills = []
    has_contested = False
    for s in chain_snapshot:
        step_num = s.get("step", "?")
        resolved = s.get("resolved_value")
        lookup = s.get("lookup", "")[:60]

        # Detect contested: was resolved before, now cleared
        was_resolved = step_num in prev_resolved
        just_contested = was_resolved and not resolved

        if just_contested:
            has_contested = True
            old_val = prev_resolved.get(step_num, "?")
            cls = "chain-pill contested"
            icon = "⚠️"
            tooltip = (
                f"Step {step_num}: {lookup} — CONTESTED by spot-check "
                f"(was: {_esc(str(old_val)[:60])})"
            )
        elif resolved:
            just_resolved = step_num not in prev_resolved
            cls = "chain-pill resolved" + (" just-resolved" if just_resolved else "")
            icon = "✅"
            tooltip = f"Step {step_num}: {lookup} → {_esc(str(resolved)[:80])}"
        else:
            # Check if dependencies are met (simplified: step 1 always unlocked,
            # step N unlocked if step N-1 resolved)
            dep_met = True
            for dep in chain_snapshot:
                if dep.get("step", 0) < step_num and not dep.get("resolved_value"):
                    dep_met = False
                    break
            if dep_met:
                cls = "chain-pill unlocked"
                icon = "🔓"
            else:
                cls = "chain-pill locked"
                icon = "🔒"
            tooltip = f"Step {step_num}: {lookup}"

        pills.append(
            f'<span class="{cls}" title="{_esc(tooltip)}">'
            f'{icon} Step {step_num}'
            f'</span>'
        )

    label = "⛓ Chain:"
    if has_contested:
        label = "⛓ Chain (contested):"

    return (
        '<div class="chain-progress">'
        f'<span class="chain-progress-label">{label}</span>'
        + "".join(pills)
        + '</div>'
    )


def _render_turn_group(group: dict, prev_chain_snapshot: Optional[list] = None) -> str:
    """Render a root turn and its sub-agent children as a collapsible group.

    group = {"root": card_dict, "children": [card_dict, ...]}
    """
    root_card = group["root"]
    children = group["children"]

    parts = []

    # Render the root turn card (always visible)
    root_html = _render_turn_card(root_card)

    # Inject chain progress into root card (before closing .turn-body)
    chain_snapshot = root_card["turn"].get("chain_snapshot")
    if chain_snapshot:
        chain_progress_html = _render_chain_progress(chain_snapshot, prev_chain_snapshot)
        # Insert the chain progress after turn-header, before turn-body content
        # We inject it inside .turn-body at the top
        inject_marker = '<div class="turn-body">'
        root_html = root_html.replace(
            inject_marker,
            inject_marker + '\n' + chain_progress_html,
            1
        )

    parts.append(root_html)

    # Render collapsible sub-agent section
    if children:
        # Count unique sub-agents and total turns
        sub_agents = set()
        for c in children:
            sub_agents.add(c["agent_label"])
        n_agents = len(sub_agents)
        n_turns = len(children)

        parts.append(
            f'<details class="sub-agent-group" id="{root_card["id"]}-subs">'
            f'<summary class="sub-agent-group-summary">'
            f'<span class="sub-agent-group-icon">▶</span>'
            f' {n_agents} sub-agent{"s" if n_agents != 1 else ""}'
            f' · {n_turns} turn{"s" if n_turns != 1 else ""}'
            f'</summary>'
            f'<div class="sub-agent-group-body">'
        )
        for child_card in children:
            parts.append(_render_turn_card(child_card))
        parts.append('</div>')  # .sub-agent-group-body
        parts.append('</details>')  # .sub-agent-group

    return "\n".join(parts)


def _build_timeline(flat_cards: list, groups: Optional[list] = None) -> str:
    """Generate the sidebar timeline HTML.

    If groups are provided, show root turns as primary items with sub-agent
    counts. Otherwise fall back to the flat card list.
    """
    parts = ['<ul class="timeline">']

    if groups:
        for group in groups:
            root = group["root"]
            children = group["children"]
            card_id = root["id"]
            turn_num = root["turn_num"]
            tool_names = [tc.get("tool_name", "?") for tc in root["turn"].get("tool_calls", [])]
            primary_tool = tool_names[0] if tool_names else ""
            # Compact tool label
            tool_label = ""
            if primary_tool:
                short = primary_tool.replace("_", " ").replace("conduct research", "research")
                tool_label = f' <span class="tl-tool">{_esc(short)}</span>'

            sub_info = ""
            if children:
                n_subs = len(set(c["agent_label"] for c in children))
                sub_info = f' <span class="tl-sub-count">{n_subs}↓</span>'

            # Chain progress indicator
            chain_snap = root["turn"].get("chain_snapshot")
            chain_dots = ""
            if chain_snap:
                dots = []
                for s in chain_snap:
                    if s.get("resolved_value"):
                        dots.append('<span class="tl-chain-dot resolved">●</span>')
                    else:
                        dots.append('<span class="tl-chain-dot pending">○</span>')
                chain_dots = f' <span class="tl-chain-dots">{"".join(dots)}</span>'

            parts.append(
                f'<li class="tl-item" data-target="{card_id}">'
                f'Turn {turn_num}{tool_label}{sub_info}{chain_dots}'
                f'</li>'
            )
    else:
        current_label = None
        for card in flat_cards:
            depth = card["depth"]
            agent_label = card["agent_label"]
            turn_num = card["turn_num"]
            card_id = card["id"]

            if agent_label != current_label:
                current_label = agent_label
                parts.append(f'<li class="tl-label">{_esc(agent_label)}</li>')

            depth_cls = "" if depth == 0 else ("sub" if depth == 1 else "sub2")
            parts.append(
                f'<li class="tl-item {depth_cls}" data-target="{card_id}">Turn {turn_num}</li>'
            )

    parts.append('</ul>')
    return "\n".join(parts)


def _render_chain_panel(chain_plan: Optional[dict]) -> str:
    """Render the chain analysis panel HTML. Returns empty string if no chain data."""
    if chain_plan is None:
        return ""

    has_chain = chain_plan.get("has_chain", False)
    chain_steps = chain_plan.get("chain_steps", [])
    parallel_tasks = chain_plan.get("parallel_tasks", [])

    if not has_chain and not parallel_tasks:
        return ""

    badge_cls = "has-chain" if has_chain else "no-chain"
    badge_text = f"⛓ {len(chain_steps)} steps" if has_chain else "⚡ parallel"
    step_count_text = f"{len(chain_steps)} chain step{'s' if len(chain_steps) != 1 else ''}"
    parallel_count_text = f"{len(parallel_tasks)} parallel task{'s' if len(parallel_tasks) != 1 else ''}"

    parts = ['<div class="chain-panel">']

    # Header
    parts.append(
        f'<div class="chain-panel-header" onclick="toggleChainPanel(this)">'
        f'<div class="chain-panel-title">'
        f'<span class="chain-badge {badge_cls}">{badge_text}</span>'
        f' Chain Analysis'
        f'</div>'
        f'<div style="font-family:var(--mono);font-size:10px;color:var(--text-light);">'
        f'{step_count_text}, {parallel_count_text}'
        f'{_CHEVRON_SVG}'
        f'</div>'
        f'</div>'
    )

    # Body
    parts.append('<div class="chain-panel-body">')

    if has_chain and chain_steps:
        # Build a lookup for resolved values
        resolved_map = {}
        for s in chain_steps:
            if s.get("resolved_value"):
                resolved_map[s["placeholder"]] = s["resolved_value"]

        parts.append('<ul class="chain-steps">')
        for s in chain_steps:
            step_num = s.get("step", "?")
            lookup = s.get("lookup", "")
            depends_on = s.get("depends_on")
            placeholder = s.get("placeholder", "")
            resolved = s.get("resolved_value")

            # Substitute resolved placeholders into lookup text
            display_lookup = lookup
            for ph, val in resolved_map.items():
                display_lookup = display_lookup.replace(ph, f'<strong>{_esc(val)}</strong>')
            # Escape remaining text (but preserve our <strong> tags)
            # We need to be careful: escape first, then re-insert strong tags
            escaped_lookup = _esc(lookup)
            for ph, val in resolved_map.items():
                escaped_lookup = escaped_lookup.replace(
                    _esc(ph), f'<strong style="color:var(--success)">{_esc(val)}</strong>'
                )

            # Determine state
            if resolved:
                state_cls = "resolved"
            elif depends_on is not None:
                dep_step = next((x for x in chain_steps if x.get("step") == depends_on), None)
                if dep_step and dep_step.get("resolved_value"):
                    state_cls = "unlocked"
                else:
                    state_cls = "locked"
            else:
                state_cls = "unlocked"

            parts.append(f'<li class="chain-step {state_cls}">')
            parts.append('<div class="chain-step-connector"></div>')
            parts.append(f'<div class="chain-step-num">Step {step_num}</div>')
            parts.append(f'<div class="chain-step-lookup">{escaped_lookup}</div>')

            if depends_on is not None:
                parts.append(
                    f'<div class="chain-step-dep">depends on step {depends_on}</div>'
                )

            if resolved:
                parts.append(
                    f'<div class="chain-step-result">✅ {_esc(str(resolved))}</div>'
                )

            parts.append('</li>')
        parts.append('</ul>')

    if parallel_tasks:
        parts.append('<div class="chain-parallel">')
        parts.append('<div class="chain-parallel-title">⚡ Parallel Tasks</div>')
        for task in parallel_tasks:
            parts.append(f'<div class="chain-parallel-item">{_esc(str(task))}</div>')
        parts.append('</div>')

    if not has_chain and not chain_steps:
        parts.append(
            '<div class="chain-no-chain">'
            'No causal chain detected — all tasks can be researched in parallel.'
            '</div>'
        )

    parts.append('</div>')  # .chain-panel-body
    parts.append('</div>')  # .chain-panel
    return "\n".join(parts)


def render_trace_html(trace_dict: dict, title: str = "Dispatch Trace") -> str:
    """Generate a self-contained HTML page from a trace dict (or loaded JSON).

    Uses the trace_viewer design: sidebar timeline + main content with turn cards.
    """
    d = trace_dict
    tid = _esc(d.get("trace_id", "?"))
    model = _esc(d.get("model", "?"))
    started = _esc(d.get("started_at", ""))
    dur = d.get("duration_s", 0)
    user_input = _esc(d.get("user_input", ""))
    final_raw = d.get("final_response", "") or ""

    # Strip broken markdown image references
    final_clean = re.sub(r'!\[.*?\]\(data:image[^)]*\)', '', final_raw).strip()
    # For the <script type="text/template"> tag: escape only </script> sequences
    # so the browser doesn't prematurely close the tag.  The JS reads textContent
    # and feeds it to marked.parse(), so HTML entities must NOT be used here.
    final = final_clean.replace('</script>', '<\\/script>')

    # Collect all images produced during the episode
    all_images = _collect_all_images(d)

    # Compute recursive stats
    def _count(t):
        turns = len(t.get("turns", []))
        calls = sum(len(turn.get("tool_calls", [])) for turn in t.get("turns", []))
        prompt_tok = sum(turn.get("prompt_tokens", 0) for turn in t.get("turns", []))
        comp_tok = sum(turn.get("completion_tokens", 0) for turn in t.get("turns", []))
        subs = 0
        for turn in t.get("turns", []):
            for tc in turn.get("tool_calls", []):
                if tc.get("child_trace"):
                    subs += 1
                    ct, cc, cs, cpt, cct = _count(tc["child_trace"])
                    turns += ct; calls += cc; subs += cs
                    prompt_tok += cpt; comp_tok += cct
        return turns, calls, subs, prompt_tok, comp_tok

    total_turns, total_calls, total_subs, total_prompt_tok, total_comp_tok = _count(d)
    total_tok = total_prompt_tok + total_comp_tok
    # Token ratio percentages for the bar
    in_pct = (total_prompt_tok / total_tok * 100) if total_tok else 50
    out_pct = 100 - in_pct

    # Flatten trace into sequential cards, then group by root turn
    flat_cards = _flatten_trace(d)
    groups = _group_by_root_turn(flat_cards)

    # ── Synthesize chain snapshots for traces that predate per-turn recording ──
    # If the trace has a chain_plan but root turns lack chain_snapshot, we
    # reconstruct approximate snapshots so the inline pills and sidebar dots
    # render even for legacy traces.
    chain_plan = d.get("chain_plan")
    if chain_plan and chain_plan.get("has_chain") and chain_plan.get("chain_steps"):
        any_has_snapshot = any(
            g["root"]["turn"].get("chain_snapshot") is not None for g in groups
        )
        if not any_has_snapshot:
            _synthesize_chain_snapshots(groups, chain_plan)

    # Build sidebar timeline (root-focused with sub-agent counts)
    timeline_html = _build_timeline(flat_cards, groups=groups)

    # Build turn cards (grouped: root + collapsible sub-agents)
    cards_parts = []
    prev_chain_snapshot = None
    for group in groups:
        cards_parts.append(_render_turn_group(group, prev_chain_snapshot))
        # Track chain snapshot for diff detection
        root_snap = group["root"]["turn"].get("chain_snapshot")
        if root_snap is not None:
            prev_chain_snapshot = root_snap

    # Build chain analysis panel
    chain_html = _render_chain_panel(d.get("chain_plan"))

    # Build final response images
    images_html = "".join(
        f'<div class="tc-img" style="margin:8px 0;">'
        f'<div style="font-size:10px;color:var(--text-light);margin-bottom:4px;">\U0001f4ce {_esc(img["filename"])}</div>'
        f'<img src="data:{({"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","gif":"image/gif","svg":"image/svg+xml"}).get(img["filename"].rsplit(".",1)[-1].lower(),"image/png")};base64,{img["data"]}" '
        f'alt="{_esc(img["filename"])}" style="max-width:100%;border-radius:6px;border:1px solid var(--border);"/>'
        f'</div>'
        for img in all_images
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)} — {tid}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,100..900;1,9..144,100..900&family=JetBrains+Mono:wght@300;400;500&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>{_CSS}</style>
</head>
<body>

<div class="viewer">

  <!-- ── Sidebar ── -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-title"><span class="dot"></span> trajectorykit</div>
      <div class="sidebar-meta">{model}<br/>{started}</div>
    </div>
    {timeline_html}
  </aside>

  <!-- ── Main content ── -->
  <main class="main">

    <!-- Prompt -->
    <div class="prompt-banner">
      <div class="prompt-label">User Prompt</div>
      <div class="prompt-text">{user_input}</div>
    </div>

    <!-- Chain analysis -->
    {chain_html}

    <!-- Stats header -->
    <div class="trace-header">
      <div class="trace-header-bar">
        <div class="trace-header-left"><div class="dot"></div> Trace {tid}</div>
        <div class="trace-header-right">{model}</div>
      </div>
      <div class="trace-stats">
        <div class="trace-stat"><span class="trace-stat-val">{dur:.1f}s</span><span class="trace-stat-lbl">Duration</span></div>
        <div class="trace-stat"><span class="trace-stat-val">{total_turns}</span><span class="trace-stat-lbl">Turns</span></div>
        <div class="trace-stat"><span class="trace-stat-val">{total_subs}</span><span class="trace-stat-lbl">Sub-Agents</span></div>
        <div class="trace-stat"><span class="trace-stat-val">{total_prompt_tok:,}</span><span class="trace-stat-lbl">Input Tokens</span></div>
        <div class="trace-stat"><span class="trace-stat-val">{total_comp_tok:,}</span><span class="trace-stat-lbl">Output Tokens</span></div>
      </div>
      <div style="padding:0 16px 14px;">
        <div class="token-ratio-bar">
          <div class="token-ratio-in" style="width:{in_pct:.1f}%"></div>
          <div class="token-ratio-out" style="width:{out_pct:.1f}%"></div>
        </div>
        <div class="token-ratio-legend">
          <span><span class="swatch" style="background:var(--accent)"></span> Input {in_pct:.0f}%</span>
          <span><span class="swatch" style="background:var(--success)"></span> Output {out_pct:.0f}%</span>
          <span style="color:var(--text-mid);">{total_tok:,} total</span>
        </div>
      </div>
    </div>

    <!-- Turn cards -->
    {"".join(cards_parts)}

    <!-- Final response -->
    <div class="final-card" id="final">
      <div class="final-header">\u2705 Final Response</div>
      <div class="final-body">
        <script type="text/template" id="final-md">{final}</script>
        <div class="final-rendered" id="final-rendered"></div>
        {images_html}
      </div>
    </div>

  </main>
</div>

<script>{_JS}</script>
</body>
</html>"""


def render_trace_file(json_path: str, output_path: str = None) -> str:
    """Load a trace JSON file and render it as HTML. Returns the HTML file path."""
    with open(json_path) as f:
        data = json.load(f)
    html_str = render_trace_html(data)
    if output_path is None:
        output_path = json_path.replace(".json", ".html")
    with open(output_path, "w") as f:
        f.write(html_str)
    return output_path
