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
        return path

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
            print(f"{prefix}  Final response:       {self.final_response[:200]}{'...' if len(self.final_response) > 200 else ''}")
            print(f"{prefix}{'═' * 60}")

    def to_html(self) -> str:
        """Generate a self-contained HTML page visualizing this trace."""
        d = self.to_dict()
        return render_trace_html(d)


# ─── HTML Rendering ────────────────────────────────────────────────────


_CSS = """\
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --orange: #d29922; --red: #f85149;
  --purple: #bc8cff;
  --code-bg: #1c2128; --hover: #1f262e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.5; padding: 24px; max-width: 960px; margin: 0 auto; }

/* ── Prompt banner ── */
.prompt { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
          padding: 20px 24px; margin-bottom: 20px; }
.prompt .label { color: var(--muted); font-size: .75rem; text-transform: uppercase;
                 letter-spacing: .5px; margin-bottom: 6px; }
.prompt .text { font-size: 1.05rem; }

/* ── Header / meta ── */
h1 { font-size: 1.4rem; margin-bottom: 4px; }
.meta { color: var(--muted); font-size: .85rem; margin-bottom: 16px; }
.stats { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
        padding: 10px 16px; min-width: 100px; }
.stat-val { font-size: 1.4rem; font-weight: 600; }
.stat-lbl { color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .5px; }

/* ── Final response ── */
.final { margin-top: 6px; }
.final summary { cursor: pointer; font-size: .8rem; color: var(--green); padding: 4px 0; font-weight: 600; }
.final pre { background: var(--code-bg); padding: 10px; border-radius: 4px;
  font-size: .8rem; overflow-x: auto; max-height: 400px;
  overflow-y: auto; white-space: pre-wrap; word-break: break-word; margin-top: 4px;
  border-left: 2px solid var(--green); }

/* ── Section divider ── */
.section-label { color: var(--muted); font-size: .75rem; text-transform: uppercase;
                 letter-spacing: .5px; margin: 16px 0 8px 0; padding-bottom: 4px;
                 border-bottom: 1px solid var(--border); }

/* ── Agent blocks ── */
.agent { border-left: 3px solid var(--accent); margin: 8px 0; border-radius: 0 6px 6px 0; overflow: hidden; }
.agent.depth-1 { border-left-color: var(--green); }
.agent.depth-2 { border-left-color: var(--orange); }
.agent.depth-3 { border-left-color: var(--red); }
.agent-header { padding: 8px 14px; cursor: pointer; user-select: none;
                display: flex; align-items: center; gap: 8px;
                background: var(--surface); border-bottom: 1px solid var(--border); }
.agent-header:hover { background: var(--hover); }
.agent-header .arrow { transition: transform .2s; font-size: .7rem; color: var(--muted); }
.agent-header.collapsed .arrow { transform: rotate(-90deg); }
.agent-body { padding: 0 0 0 14px; }

/* ── Turn blocks ── */
.turn { margin: 6px 0; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.turn-header { background: var(--surface); padding: 6px 12px; font-size: .85rem;
               cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
.turn-header:hover { background: var(--hover); }
.turn-body { display: none; padding: 0; }
.turn.open .turn-body { display: block; }

/* ── Tool calls ── */
.tc { padding: 10px 14px; border-top: 1px solid var(--border); }
.tc-name { font-weight: 600; color: var(--accent); }
.tc-time { color: var(--muted); font-size: .8rem; margin-left: 8px; }
.tc-args, .tc-output, .tc-reasoning { margin-top: 6px; }
.tc-args summary, .tc-output summary, .tc-reasoning summary {
  cursor: pointer; font-size: .8rem; color: var(--muted); padding: 4px 0; }
.tc-args pre, .tc-output pre, .tc-reasoning pre {
  background: var(--code-bg); padding: 10px; border-radius: 4px;
  font-size: .8rem; overflow-x: auto; max-height: 400px;
  overflow-y: auto; white-space: pre-wrap; word-break: break-word; margin-top: 4px; }
.tc-reasoning summary { color: var(--purple); }
.tc-reasoning pre { border-left: 2px solid var(--purple); }
.tc-img { margin-top: 8px; }
.tc-img img { max-width: 100%; border-radius: 6px; border: 1px solid var(--border); }

/* ── Assistant message ── */
.assistant { padding: 10px 14px; border-top: 1px solid var(--border); font-size: .9rem; }
.assistant .label { color: var(--green); font-weight: 600; font-size: .8rem; margin-bottom: 4px; }

/* ── Badges ── */
.badge { display: inline-block; font-size: .7rem; padding: 2px 6px; border-radius: 4px;
         font-weight: 600; margin-left: 6px; }
.badge-tool { background: rgba(88,166,255,.15); color: var(--accent); }
.badge-depth { background: rgba(63,185,80,.15); color: var(--green); }
.badge-reasoning { background: rgba(188,140,255,.15); color: var(--purple); }
"""

_JS = """\
function toggleAgent(el) {
  el.classList.toggle('collapsed');
  const body = el.nextElementSibling;
  body.style.display = body.style.display === 'none' ? '' : 'none';
}
function toggleTurn(el) {
  el.closest('.turn').classList.toggle('open');
}
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


def _render_turns(turns: list) -> str:
    """Render a list of turn dicts to HTML (shared by root and sub-agents)."""
    parts = []
    for turn in turns:
        tn = turn.get("turn_number", "?")
        td_s = turn.get("duration_s", 0)
        tc_count = len(turn.get("tool_calls", []))
        prompt_tok = turn.get("prompt_tokens", 0)
        comp_tok = turn.get("completion_tokens", 0)

        # Check for reasoning content
        raw_msg = turn.get("raw_assistant_message", {}) or {}
        reasoning = raw_msg.get("reasoning_content", "") or ""
        reasoning_badge = (
            f'<span class="badge badge-reasoning">reasoning</span>' if reasoning else ""
        )

        parts.append(f'<div class="turn">')
        parts.append(
            f'<div class="turn-header" onclick="toggleTurn(this)">'
            f'<span>Turn {tn}'
            f'<span class="badge badge-tool">{tc_count} tool call{"s" if tc_count != 1 else ""}</span>'
            f'{reasoning_badge}</span>'
            f'<span style="color:var(--muted);font-size:.8rem;">'
            f'{prompt_tok:,} + {comp_tok:,} tok &middot; {td_s:.2f}s</span>'
            f'</div>'
        )
        parts.append('<div class="turn-body">')

        # Reasoning content (collapsible, at top of turn body)
        if reasoning:
            parts.append(
                f'<details class="tc-reasoning" style="padding:10px 14px;">'
                f'<summary>\U0001f9e0 Reasoning</summary>'
                f'<pre>{_esc(reasoning)}</pre>'
                f'</details>'
            )

        for tc in turn.get("tool_calls", []):
            tname = _esc(tc.get("tool_name", "?"))
            targs = tc.get("tool_args", {})
            toutput = tc.get("output", "")
            tc_dur = tc.get("duration_s", 0)

            parts.append('<div class="tc">')
            parts.append(f'<span class="tc-name">{tname}</span><span class="tc-time">{tc_dur:.2f}s</span>')

            # Args (collapsed)
            args_json = json.dumps(targs, indent=2, default=str)
            parts.append(f'<details class="tc-args"><summary>Arguments</summary><pre>{_esc(args_json)}</pre></details>')

            # Output (collapsed, strip base64 blobs)
            display_output = re.sub(
                r"(--- .+? \(base64\) ---\n).+?(\n---|$)",
                r"\1[base64 data omitted — see image below]\2",
                toutput, flags=re.DOTALL
            )
            parts.append(f'<details class="tc-output"><summary>Output</summary><pre>{_esc(display_output)}</pre></details>')

            # Inline images
            for img in _extract_images_from_output(toutput):
                fname = _esc(img["filename"])
                ext = img["filename"].rsplit(".", 1)[-1].lower()
                mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
                parts.append(
                    f'<div class="tc-img">'
                    f'<div style="font-size:.8rem;color:var(--muted);margin-bottom:4px;">📎 {fname}</div>'
                    f'<img src="data:{mime};base64,{img["data"]}" alt="{fname}"/>'
                    f'</div>'
                )

            # Child trace (sub-agent)
            if tc.get("child_trace"):
                parts.append(_render_agent(tc["child_trace"]))

            parts.append('</div>')  # .tc

        # Assistant text response (non-tool-call turns)
        acontent = turn.get("assistant_content", "")
        if acontent:
            preview = _esc(acontent[:500])
            parts.append(f'<div class="assistant"><div class="label">💬 Assistant</div>{preview}</div>')

        parts.append('</div>')  # .turn-body
        parts.append('</div>')  # .turn
    return "\n".join(parts)


def _render_agent(trace_dict: dict) -> str:
    """Render a sub-agent trace to HTML (collapsible block with its own prompt)."""
    d = trace_dict
    depth = d.get("depth", 0)
    depth_cls = f"depth-{min(depth, 3)}"
    tid = _esc(d.get("trace_id", "?"))
    user_input = _esc(d.get("user_input", "")[:200])
    dur = d.get("duration_s", 0)
    n_turns = len(d.get("turns", []))
    n_tc = d.get("total_tool_calls", 0)

    parts = []
    parts.append(f'<div class="agent {depth_cls}">')
    parts.append(
        f'<div class="agent-header collapsed" onclick="toggleAgent(this)">'
        f'<span class="arrow">▼</span>'
        f'<strong>{"Sub-Agent" if depth > 0 else "Root Agent"}</strong>'
        f'<span class="badge badge-depth">depth {depth}</span>'
        f'<span style="color:var(--muted);font-size:.8rem;margin-left:auto;">'
        f'{tid} &middot; {n_turns} turns &middot; {n_tc} calls &middot; {dur:.1f}s</span>'
        f'</div>'
    )
    parts.append('<div class="agent-body" style="display:none;">')

    # Sub-agent prompt
    if user_input:
        parts.append(f'<div style="padding:8px 14px;font-size:.85rem;color:var(--muted);">Prompt: {user_input}</div>')

    # Turns
    parts.append(_render_turns(d.get("turns", [])))

    # Sub-agent final response
    final = d.get("final_response", "")
    if final:
        parts.append(
            f'<details class="final" style="margin:8px 0;" open>'
            f'<summary>✅ Sub-Agent Response</summary>'
            f'<pre>{_esc(final)}</pre>'
            f'</details>'
        )

    parts.append('</div>')  # .agent-body
    parts.append('</div>')  # .agent
    return "\n".join(parts)


def render_trace_html(trace_dict: dict, title: str = "Dispatch Trace") -> str:
    """Generate a self-contained HTML page from a trace dict (or loaded JSON).

    Layout:
      1. Prompt (top, always visible)
      2. Stats bar
      3. Trace detail (starts collapsed)
      4. Final response (bottom, always visible)
    """
    d = trace_dict
    tid = _esc(d.get("trace_id", "?"))
    model = _esc(d.get("model", "?"))
    started = _esc(d.get("started_at", ""))
    dur = d.get("duration_s", 0)
    user_input = _esc(d.get("user_input", ""))
    final_raw = d.get("final_response", "")

    # Strip broken markdown image references from the final response text
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

    # Render the root agent turns (not wrapped in an agent box)
    trace_body = _render_turns(d.get("turns", []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)} — {tid}</title>
<style>{_CSS}</style>
</head>
<body>

<!-- Prompt -->
<div class="prompt">
  <div class="label">Prompt</div>
  <div class="text">{user_input}</div>
</div>

<!-- Header & Stats -->
<h1>🔍 {_esc(title)}</h1>
<div class="meta">{tid} &middot; {model} &middot; {started}</div>
<div class="stats">
  <div class="stat"><div class="stat-val">{dur:.1f}s</div><div class="stat-lbl">Duration</div></div>
  <div class="stat"><div class="stat-val">{total_turns}</div><div class="stat-lbl">Total Turns</div></div>
  <div class="stat"><div class="stat-val">{total_calls}</div><div class="stat-lbl">Tool Calls</div></div>
  <div class="stat"><div class="stat-val">{total_subs}</div><div class="stat-lbl">Sub-Agents</div></div>
  <div class="stat"><div class="stat-val">{total_tok:,}</div><div class="stat-lbl">Tokens ({total_prompt_tok:,} prompt + {total_comp_tok:,} completion)</div></div>
</div>

<!-- Trace Detail (collapsible) -->
<div class="section-label">Trace Detail</div>
<div class="agent depth-0">
  <div class="agent-header collapsed" onclick="toggleAgent(this)">
    <span class="arrow">▼</span>
    <strong>Root Agent</strong>
    <span class="badge badge-depth">depth 0</span>
    <span style="color:var(--muted);font-size:.8rem;margin-left:auto;">
      {len(d.get("turns", []))} turns &middot; {d.get("total_tool_calls", 0)} calls &middot; {dur:.1f}s
    </span>
  </div>
  <div class="agent-body" style="display:none;">
    {trace_body}
  </div>
</div>

<!-- Final Response (always visible) -->
<details class="final" open>
  <summary>✅ Final Response</summary>
  <pre>{final}</pre>
  {''.join(
      f'<div class="tc-img" style="margin:8px 10px;">'
      f'<div style="font-size:.8rem;color:var(--muted);margin-bottom:4px;">📎 {_esc(img["filename"])}</div>'
      f'<img src="data:{({"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","gif":"image/gif","svg":"image/svg+xml"}).get(img["filename"].rsplit(".",1)[-1].lower(),"image/png")};base64,{img["data"]}" '
      f'alt="{_esc(img["filename"])}" style="max-width:100%;border-radius:6px;border:1px solid var(--border);"/>'
      f'</div>'
      for img in all_images
  )}
</details>

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
