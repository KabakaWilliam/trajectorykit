# 🤖 TrajectoryKit

> A local-first agentic framework for running LLM agents with recursive sub-agents, sandboxed code execution, and full execution tracing — all from a single YAML config.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

---

## Why TrajectoryKit?

Most agent frameworks are cloud-first and opaque. TrajectoryKit is designed for researchers who want full control:

- **One config, one command.** A single YAML defines the model, GPU assignment, vLLM flags, sandbox container, evaluation dataset, and judge — then `orchestrate.py` handles the rest.
- **Recursive delegation.** The root agent spawns sub-agents that each get a fresh context window, their own tool access, and a recorded trace — no context pollution.
- **RLM-inspired memory.** Long tool outputs are stored in an external `MemoryStore` and accessed programmatically via code execution, not via attention. The LLM writes programs to query its own research history.
- **Every token is traced.** Full execution trees (turns, tool calls, sub-agent trees, token counts, latencies) are saved as JSON and rendered as interactive HTML.

---

## Quickstart

```bash
# 1. Install
conda env create -f environment.yml
conda activate pika
pip install -e .

# 2. Run an experiment end-to-end (sandbox + vLLM + eval + judge)
python orchestrate.py --config configs/experiments/gpt_oss_deepsearchqa.yaml
```

Or start services and run the eval separately:

```bash
# Start services only
python orchestrate.py --config configs/experiments/gpt_oss_deepsearchqa.yaml --services-only

# Run eval (includes LLM judge if judge.enabled: true in config)
python evals/eval.py --config configs/experiments/gpt_oss_deepsearchqa.yaml

# Run eval with parallel workers (3-5x speedup)
python evals/eval.py --config configs/experiments/gpt_oss_deepsearchqa.yaml --workers 4
```

### Programmatic use

```python
from trajectorykit import dispatch

result = dispatch(
    user_input="Compare the populations of Tokyo and New York City",
    turn_length=10,
    verbose=True,
)

print(result["final_response"])
result["trace"].save()  # → traces/trace_YYYYMMDD_HHMMSS_uuid.json + .html
```

<p align="center">
  <img src="docs/trace_example_v2.png" width="500" alt="Agent trace example">
</p>

### Live Demo

> [**▶ Walk through a real agent trace**](https://kabakawilliam.github.io/trajectorykit/demo_trace.html)
>
> Click through the sidebar timeline to see how the agent searches the web, reads pages,
> executes Python in a sandbox, and synthesizes an answer across 15+ turns.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  orchestrate.py                                          │
│  Reads YAML → starts Apptainer sandbox → starts vLLM    │
│  → runs eval.py → runs LLM judge                        │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  eval.py  (parallel dispatch runner)                     │
│                                                          │
│  Loads dataset → ThreadPoolExecutor(workers=N) →         │
│  dispatch() per question → results.parquet → LLM judge   │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  agent.py  (dispatch loop)                               │
│                                                          │
│  system prompt → LLM → tool_calls → dispatch_tool_call   │
│       ▲                                    │             │
│       └────────── tool results ◄───────────┘             │
│                                                          │
│  Depth 0 (root):  orchestrator prompt, full tools        │
│  Depth 1+ (sub):  worker prompt, no spawn_agent          │
│  Synthesis:       synthesizer prompt, code + final_answer│
│                                                          │
│  Resilience:                                             │
│  • Degeneration detection → auto synthesis fallback      │
│  • Malformed final_answer → 3-layer recovery             │
│  • Consecutive search ban → forces sub-agent delegation  │
│  • Empty answer → memory-backed synthesis pipeline       │
└──────────────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  tool_store.py                                           │
│                                                          │
│  search_web      Serper → auto-fallback to DuckDuckGo   │
│  fetch_url       Browser-grade headers, cookie jar,      │
│                  block detection, Jina fallback,          │
│                  PDF auto-detection + text extraction     │
│  read_pdf        PDF download + pypdf text extraction    │
│  execute_code    Sandboxed via Sandbox Fusion API        │
│  spawn_agent     Recursive sub-agent (fresh context)     │
│  final_answer    Terminates the agent loop               │
│  search_available_tools  Self-introspection              │
│  get_current_time / add_numbers  Utility tools           │
└──────────────────────────────────────────────────────────┘
```

### Key design decisions

| Concern | Approach |
|---------|----------|
| Long context | `MemoryStore` captures full tool outputs externally; synthesis sub-agent queries them via `execute_code` |
| Web fetch resilience | Browser-grade User-Agent + headers, cookie jar, block detection with retries, Jina Reader fallback for stubborn sites |
| PDF handling | Content-type + magic-byte detection before parsing; HTML masquerading as PDF handled gracefully |
| Search resilience | Primary backend (Serper) with automatic DuckDuckGo fallback; consecutive search ban forces delegation |
| Model degeneration | Detects 3+ turns with no tool calls (< 200 chars); breaks to fresh synthesis sub-agent instead of accepting garbage |
| Malformed `final_answer` | 3-layer recovery: regex extraction → reject + retry → findings-backed synthesis fallback |
| Empty `final_answer` | Nudge echoes model's own text; structured backfill scans prior responses; falls back to synthesis pipeline |
| Budget management | Progressive turn warnings at 5/3/2/1 remaining; last turn restricts tools to `final_answer` only |
| Parallel eval | `ThreadPoolExecutor` with configurable workers; questions are embarrassingly parallel (stateless HTTP I/O) |
| Trace fidelity | Every turn records: assistant content, tool calls with args/output/duration, child traces, token counts |

---

## Experiment Configs

Everything is defined in a single YAML:

```yaml
# configs/experiments/gpt_oss_deepsearchqa.yaml

model:
  name: "openai/gpt-oss-20b"
  api_url: "http://localhost:3030/v1"

vllm:
  port: 3030
  gpu_devices: [2]
  gpu_memory_utilization: 0.9
  tool_call_parser: "openai"
  extra_args: ["--async-scheduling"]

sandbox:
  url: "http://localhost:8080/run_code"
  port: 8080
  sif_image: "sandbox-fusion_server-20250609.sif"
  docker_uri: "docker://volcengine/sandbox-fusion:server-20250609"

agent:
  max_recursion_depth: 1
  sub_agent_turn_budget: 15

dataset:
  name: "google/deepsearchqa"
  split: "eval"
  sample_n: 100
  seed: 42

eval:
  turn_length: null          # unlimited
  reasoning_effort: "high"
  workers: 4                 # parallel dispatch calls (1 = sequential)

judge:
  enabled: true
  model: "gemini-2.5-flash"  # or "gpt-4.1-mini"
  threads: 5
  # rpm: 5                   # uncomment to throttle (Gemini free tier)
```

Run it:

```bash
python orchestrate.py --config configs/experiments/gpt_oss_deepsearchqa.yaml
```

The orchestrator will:
1. Pull the Apptainer SIF if not cached, start the sandbox container
2. Launch vLLM with the correct model, GPU, parser, and flags
3. Wait for both services to be healthy
4. Run the eval on the configured dataset (parallel if `workers > 1`)
5. Run the LLM judge and save detailed ratings
6. Leave services running for further use

---

## Project Structure

```
trajectorykit/
├── orchestrate.py                    # One-command experiment runner
├── serve_vllm_oss.sh                 # Manual vLLM launch script
├── environment.yml                   # Conda environment
├── pyproject.toml                    # Package metadata
│
├── configs/
│   ├── default.yaml                  # Fallback config
│   ├── experiments/
│   │   ├── gpt_oss_deepsearchqa.yaml # GPT-OSS-20B experiment
│   │   └── qwen3_deepsearchqa.yaml   # Qwen3-8B experiment
│   └── prompts/
│       ├── orchestrator.txt          # Root agent system prompt
│       ├── worker.txt                # Sub-agent system prompt
│       └── synthesizer.txt           # Synthesis sub-agent prompt
│
├── src/trajectorykit/
│   ├── __init__.py                   # Public API
│   ├── agent.py                      # Agentic loop + dispatch
│   ├── config.py                     # YAML config loader
│   ├── tool_store.py                 # Tool definitions + wrappers
│   ├── tracing.py                    # Trace dataclasses + HTML renderer
│   ├── memory.py                     # MemoryStore (compressed tool output storage)
│   └── utils.py                      # Shared utilities
│
├── evals/
│   ├── eval.py                       # Dataset evaluation runner (parallel)
│   ├── llm_judge.py                  # LLM-as-judge grading (OpenAI + Gemini)
│   ├── traces_to_parquet.py          # Build results parquet from trace JSONs
│   └── recover_parquet.py            # Rebuild results from trace JSONs (legacy)
│
├── data/                             # Evaluation outputs (parquets, traces)
├── traces/                           # Ad-hoc trace storage
└── docs/                             # Screenshots and diagrams
```

---

## Tools

| Tool | Description |
|------|-------------|
| `execute_code` | Run code via Sandbox Fusion API (Python + 40 languages). Supports file upload/download as base64. |
| `search_web` | Web search via Serper (primary) with automatic DuckDuckGo fallback. |
| `fetch_url` | Fetch a web page with browser-grade headers, cookie jar, block detection, Jina Reader fallback, and automatic PDF detection. |
| `read_pdf` | Download and extract text from a PDF via pypdf. |
| `spawn_agent` | Spawn a recursive sub-agent with fresh context and its own trace. |
| `final_answer` | Submit the final answer (terminates the loop). |
| `search_available_tools` | Self-introspection — list tools or get full schema for any tool. |
| `get_current_time` | Current date and time. |
| `add_numbers` | Add two numbers (demo/testing). |

---

## Tracing

Every `dispatch()` call produces a full execution trace.

### Terminal output

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
📊 Episode Summary:
  Prompt tokens:        3,621
  Completion tokens:    686
  Total tokens:         4,307
```

### JSON + HTML

```python
result["trace"].save()
# → traces/trace_20260219_210429_154b9f2e.json
# → traces/trace_20260219_210429_154b9f2e.html
```

The HTML viewer is self-contained with:
- Collapsible turn-by-turn trace detail
- Reasoning content (chain-of-thought) dropdowns
- Inline rendered images (base64)
- Token stats and latency per turn

---

## Evaluation Pipeline

The evaluation pipeline is fully automated:

```bash
# End-to-end: services → eval → judge
python orchestrate.py --config configs/experiments/gpt_oss_deepsearchqa.yaml
```

**What happens:**

1. `eval.py` runs the agent on each dataset sample in parallel (`workers` threads), saving per-item traces and a `results.parquet`
2. `llm_judge.py` grades each `(question, response)` pair using an LLM judge (OpenAI or Gemini — auto-detected from model name)
3. Outputs: `results.parquet`, `results_judge_ratings.json` with per-item and aggregate metrics

**Standalone usage:**

```bash
# Eval only (no judge)
python evals/eval.py --config configs/experiments/gpt_oss_deepsearchqa.yaml

# Eval with parallel workers
python evals/eval.py --config configs/experiments/gpt_oss_deepsearchqa.yaml --workers 4

# Judge only (on existing results — OpenAI)
python evals/llm_judge.py --results data/google_deepsearchqa/gpt_oss_20b/results.parquet --judge-model gpt-4.1-mini

# Judge only (Gemini)
python evals/llm_judge.py --results data/google_deepsearchqa/gpt_oss_20b/results.parquet --judge-model gemini-2.5-flash

# Judge with rate limiting (Gemini free tier, 5 RPM)
python evals/llm_judge.py --results ... --judge-model gemini-2.5-flash --rpm 5

# Build results parquet from trace JSONs (no config needed)
python evals/traces_to_parquet.py --traces data/google_deepsearchqa/gpt_oss_20b/traces/
```

### LLM Judge

The judge supports **multiple providers with automatic detection**:

| Model prefix | Provider | Env var | Notes |
|---|---|---|---|
| `gpt-*`, `o1-*`, etc. | OpenAI | `OPENAI_API_KEY` | Default provider |
| `gemini-*` | Gemini | `GEMINI_API_KEY` | Supports `response_mime_type: "application/json"` for structured output |

Switching judges is as simple as changing `judge.model` in your experiment YAML. The Gemini integration includes:
- Thread-safe rate limiter (`--rpm N`) for free-tier compliance
- Server-suggested `retryDelay` parsing from 429 errors
- Omit `--rpm` for unlimited throughput (paid tier)

### Preliminary Results

Evaluated on **DeepSearchQA** (n = 100, 93–96 evaluated) with `gpt-oss-20b` served via vLLM:

| Metric | GPT-4.1-mini judge | Gemini-2.5-Flash judge |
|--------|-------------------|----------------------|
| Fully Correct | 46.88 ± 9.98% | 44.09 ± 10.09% |
| Fully Incorrect | 31.25 ± 9.27% | 34.41 ± 9.66% |
| Correct w/ Extraneous | 5.21 ± 4.44% | 5.38 ± 4.58% |
| Precision | 63.89% | 61.04% |
| Recall | 61.14% | 58.69% |
| F1 | 61.08% | 58.30% |

> **Caveats:** Wide confidence intervals at n = 100. Both judges agree closely, suggesting stable grading. This validates architectural correctness, not leaderboard performance.

---

## Configuration

Settings are loaded from YAML with fallback chain: explicit path → `TRAJECTORYKIT_CONFIG` env var → `configs/default.yaml` → hardcoded defaults.

### Key config sections

| Section | Purpose |
|---------|---------|
| `model` | Model name and API URL |
| `vllm` | Server port, GPU devices, parser, memory utilization, extra args |
| `sandbox` | Sandbox Fusion URL, port, SIF image, Docker URI |
| `agent` | Recursion depth, sub-agent turn budget, safety margin |
| `model_profiles` | Per-model settings (context window, temperature, reasoning effort) |
| `prompts` | Paths to system prompt files (orchestrator, worker, synthesizer) |
| `dataset` | HuggingFace dataset name, split, sample size, seed |
| `eval` | Turn length, reasoning effort, verbosity, parallel workers |
| `judge` | Enable/disable, judge model, thread count, RPM limit |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `SERPER_API_KEY` | Primary web search (Serper.dev). Falls back to DuckDuckGo if not set or credits exhausted. |
| `OPENAI_API_KEY` | LLM judge when using OpenAI models (e.g. `gpt-4.1-mini`). |
| `GEMINI_API_KEY` | LLM judge when using Gemini models (e.g. `gemini-2.5-flash`). |
| `SERP_API_KEY` | Legacy SerpAPI key (alternative search backend). |

---

## `dispatch()` API

```python
from trajectorykit import dispatch

result = dispatch(
    user_input="Your task here",
    turn_length=10,           # Max turns (None = unlimited)
    verbose=True,             # Print turn-by-turn output
    temperature=0.7,          # Sampling temperature
    model="openai/gpt-oss-20b",
    reasoning_effort="high",  # For models that support it
    example_id="q0042",       # Optional ID for tracing
)

result["final_response"]   # str — the agent's final answer
result["turns"]            # int — number of turns taken
result["tool_calls"]       # int — total tool calls made
result["messages"]         # list — full conversation history
result["trace"]            # EpisodeTrace — full execution tree
```

---

## Citation

```bibtex
@software{trajectorykit2026,
  title={TrajectoryKit: A Local-First Agentic Framework},
  author={Lugoloobi, William},
  year={2026},
  url={https://github.com/KabakaWilliam/TrajectoryKit}
}
```

## License

MIT
