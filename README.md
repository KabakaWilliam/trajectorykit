# 🤖 TrajectoryKit: A Research Agent Starter Pack

> A lightweight, local-first agentic framework for HuggingFace models with built-in tool calling, recursive sub-agents, and full execution tracing.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/trajectorykit)](https://pypi.org/project/trajectorykit/)

## 📥 Installation

```bash
pip install trajectorykit
```

For the full stack (vLLM serving, visualization, etc.):

```bash
pip install trajectorykit[all]
```

## 🏎💨 Quickstart

After installing, you can start using TrajectoryKit right away:

```python
from trajectorykit import dispatch

result = dispatch(
    user_input="Compare the stats of Blue Eyes White Dragon vs Dark Magician",
    turn_length=5,
    verbose=True
)

print(result["final_response"])
```

The agent writes and executes code in a sandbox, returning results with inline images:

<p align="center">
  <img src="docs/trace_example_v2.png" width="500" alt="Yu-Gi-Oh Card Stats Comparison">
</p>

## 📊 Preliminary Evaluation

TrajectoryKit was evaluated on a subset of **DeepSearchQA** (n = 62) to validate recursive delegation and tool orchestration under real-world conditions.

**Configuration**

- Model: `gpt-oss-20b(high)` (served via vLLM)
- Orchestration: Recursive delegation enabled (max depth ≤ 1)
- Tools: Web search + code sandbox fallback
- Judge: GPT-4o-mini (strict rubric-based evaluation)

### Results (Subset: n = 62)

| Model                | Fully Correct | Fully Incorrect | Correct w/ Extraneous | F1    |
|----------------------|---------------|----------------|-----------------------|-------|
| Ours (gpt-20b)      | 30.65 ± 11.48 | 45.16 ± 12.39  | 6.45 ± 6.12           | 46.44 |

---

### Important Caveats

- ~42% of questions encountered search credit limits mid-run.
- When search failed, the agent fell back to API-based retrieval via its code sandbox.
- Small sample size (n = 62) results in wide confidence intervals.
- Judge model (GPT-4o-mini) differs from Gemini-based evaluations reported elsewhere.

---

### Interpretation

This experiment was conducted to validate architectural stability rather than leaderboard performance.

Despite degraded search conditions:

- Recursive delegation remained stable (no depth explosions).
- Tool fallback mechanisms functioned as intended.
- The agent maintained non-trivial task performance.
- Structured sub-agent coordination operated reliably.

These results suggest that performance was bottlenecked primarily by retrieval constraints rather than orchestration instability.

---

## ✨ Features

- 🏠 **100% Local** — runs on your own GPU via vLLM, no API keys needed
- 🔄 **Agentic Loop** — iterative tool calling until the task is done
- 🤖 **Recursive Sub-Agents** — spawn child agents to decompose complex tasks (up to 3 levels deep)
- 💻 **Sandboxed Code Execution** — run Python (and 40+ languages) in an Apptainer sandbox with file I/O
- 🔍 **Web Search** — built-in Google search via SerpAPI
- 📊 **Full Execution Tracing** — every turn, tool call, token count, and sub-agent is recorded
- 🌐 **HTML Trace Viewer** — self-contained dark-themed trace pages with collapsible reasoning, inline images, and token stats

## 📦 Package Structure

```
src/trajectorykit/
├── __init__.py       # Public API: dispatch, EpisodeTrace, render_trace_html, render_trace_file
├── config.py         # Model, API URL, traces directory, system prompt
├── agent.py          # Core agentic loop with tool dispatch and trace building
├── tool_store.py     # Tool definitions + wrapper functions
├── tracing.py        # Dataclasses, pretty-print, JSON/HTML serialization
├── examples.py       # Ready-to-run examples
└── serve_vllm.sh     # Script to launch vLLM server
traces/               # Auto-created directory for saved traces (JSON + HTML)
```

## 🛠 Built-in Tools

| Tool | Description |
|------|-------------|
| `execute_code` | Run code in a sandboxed Apptainer container. Supports file upload (`files`) and retrieval (`fetch_files`) as base64. |
| `search_web` | Google search via SerpAPI. Returns titles, snippets, and links. |
| `get_current_time` | Returns the current date and time. |
| `add_numbers` | Adds two numbers (demo tool). |
| `spawn_agent` | Spawns a recursive sub-agent with its own tool access and trace. |

## 🔁 Recursive Sub-Agents

Dispatch can spawn child agents to handle subtasks independently. Each sub-agent runs its own agentic loop with full tool access and produces its own trace, nested inside the parent's trace tree.

```python
from trajectorykit import dispatch

result = dispatch(
    user_input=(
        "How has the market price of the card 'Blue-Eyes White Dragon' moved over the last years?"
    ),
    turn_length=5,
    max_tokens=4096
)

result["trace"].pretty_print()
```

Max recursion depth is controlled by `MAX_RECURSION_DEPTH` in `config.py` (default: 3).

## 📜 Tracing

Every `dispatch()` call returns a full execution trace in `result["trace"]`.

### Pretty-print to terminal

```python
result["trace"].pretty_print()
```

```
🏁 Agent [root]  trace_id=154b9f2e
  Input: What is the current time? What is 123 + 456?
  Duration: 8.42s | Turns: 2 | Tool calls: 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ┌─ Turn 1 (4.00s)
  │  🔧 get_current_time({}) [0.01s]
  │     → 2026-02-19 21:04:29
  │  🔧 add_numbers({"a": 123, "b": 456}) [0.00s]
  │     → 579
  └─
  ...
📊 Episode Summary:
  Prompt tokens:        3,621
  Completion tokens:    686
  Total tokens:         4,307
```

### Save to disk

```python
path = result["trace"].save()
# Writes traces/trace_20260219_210429_154b9f2e.json
# Also writes traces/trace_20260219_210429_154b9f2e.html
```

Traces are saved to the `traces/` directory automatically.

### HTML trace viewer

The HTML file is a self-contained dark-themed page with:
- **Prompt** at the top (always visible)
- **Stats bar** — duration, turns, tool calls, sub-agents, total tokens (prompt + completion)
- **Collapsible trace detail** — starts collapsed, expand to drill into turns
- **Reasoning dropdowns** — model chain-of-thought shown per turn
- **Inline images** — plots and generated images rendered directly
- **Final response** at the bottom (always visible)

You can also render any trace JSON to HTML:

```python
from trajectorykit import render_trace_file

render_trace_file("traces/trace_20260219_210429_154b9f2e.json")
```

## ⚙️ Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-8B` | HuggingFace model served by vLLM |
| `VLLM_API_URL` | `http://localhost:3030/v1` | vLLM OpenAI-compatible endpoint |
| `MAX_RECURSION_DEPTH` | `3` | Max sub-agent nesting depth |
| `TRACES_DIR` | `<repo_root>/traces/` | Where `.json` and `.html` traces are saved |

### Environment Variables

The `search_web` tool requires a [SerpAPI](https://serpapi.com/) key. Add it to a `.env` file in your project root:

```bash
SERP_API_KEY=your_serpapi_key_here
```

You can get a free API key at [serpapi.com](https://serpapi.com/). Web search will be disabled if this key is not set.

## 🚀 Running

### 1. Install conda env 
```bash
conda env create -f environment. yml.
```

### 2. Start the Apptainer sandbox (code execution)

```bash
bash src/trajectorykit/apptainer.sh
```

This pulls and runs the [SandboxFusion](https://github.com/volcengine/sandbox-fusion) container, exposing a code execution API on `http://localhost:8080`.

### 3. Start vLLM

```bash
bash src/trajectorykit/serve_vllm.sh
```

### 4. Run an example

```bash
cd src && python -m trajectorykit.examples
```

### API

```python
result = dispatch(
    user_input="Your task here",
    turn_length=5,          # Max turns (None = unlimited)
    verbose=True,           # Print turn-by-turn output
    max_tokens=2000,        # Max tokens per generation
    temperature=0.7,        # Sampling temperature
)

# result keys:
result["final_response"]   # str — the model's final answer
result["turns"]            # int — number of turns taken
result["tool_calls"]       # int — total tool calls made
result["messages"]         # list — full conversation history
result["trace"]            # EpisodeTrace — full execution tree
```

## 📚 Citation

If you use TrajectoryKit in your research or project, please cite it:

```bibtex
@software{trajectorykit2026,
  title={TrajectoryKit: A Research Agent Starter Pack},
  author={Lugoloobi, William},
  year={2026},
  url={https://github.com/KabakaWilliam/TrajectoryKit}
}
```

### License

MIT
