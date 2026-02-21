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


@dataclass
class EpisodeTrace:
    """Full trace of a dispatch episode, forming a tree with sub-agent traces."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
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

/* ── Responsive ── */
@media (max-width: 768px) {
  .viewer { grid-template-columns: 1fr; }
  .sidebar { position: relative; height: auto; border-right: none; border-bottom: 1px solid var(--border); }
  .main { padding: 24px 20px; }
  .trace-stats { grid-template-columns: repeat(3, 1fr); }
}
"""

_JS = """\
function toggle(header) {
  header.closest('.turn-card').classList.toggle('open');
}

function toggleTool(header) {
  header.closest('.tool-block').classList.toggle('open');
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


_CHEVRON_SVG = (
    '<svg class="chevron" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>'
)


def _depth_badge(depth: int) -> str:
    cls = "badge-root" if depth == 0 else ("badge-sub" if depth == 1 else "badge-sub2")
    label = "root" if depth == 0 else (f"sub" if depth == 1 else f"sub-{depth}")
    return f'<span class="turn-badge {cls}">{label}</span>'


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

    # Arguments
    parts.append(f'<div style="font-size:9px;color:var(--text-light);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Arguments</div>')
    parts.append(f'<pre class="tool-args-pre">{_esc(args_json)}</pre>')

    # Output / result
    if display_output.strip():
        is_error = "error" in display_output.lower()[:80] or "traceback" in display_output.lower()[:80]
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


def _build_timeline(flat_cards: list) -> str:
    """Generate the sidebar timeline HTML from the flat card list."""
    parts = ['<ul class="timeline">']
    current_label = None
    for card in flat_cards:
        depth = card["depth"]
        agent_label = card["agent_label"]
        turn_num = card["turn_num"]
        card_id = card["id"]

        # Section label when agent changes
        if agent_label != current_label:
            current_label = agent_label
            parts.append(f'<li class="tl-label">{_esc(agent_label)}</li>')

        depth_cls = "" if depth == 0 else ("sub" if depth == 1 else "sub2")
        parts.append(
            f'<li class="tl-item {depth_cls}" data-target="{card_id}">Turn {turn_num}</li>'
        )
    parts.append('</ul>')
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
    final = _esc(final_clean)

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

    # Flatten trace into sequential cards
    flat_cards = _flatten_trace(d)

    # Build sidebar timeline
    timeline_html = _build_timeline(flat_cards)

    # Build turn cards
    cards_html = "\n".join(_render_turn_card(card) for card in flat_cards)

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
    {cards_html}

    <!-- Final response -->
    <div class="final-card" id="final">
      <div class="final-header">\u2705 Final Response</div>
      <div class="final-body">
        <div class="output-block">{final}</div>
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
