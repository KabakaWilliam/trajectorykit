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
│  agent.py  (dispatch loop)                               │
│                                                          │
│  system prompt → LLM → tool_calls → dispatch_tool_call   │
│       ▲                                    │             │
│       └────────── tool results ◄───────────┘             │
│                                                          │
│  Depth 0 (root):  orchestrator prompt, full tools        │
│  Depth 1+ (sub):  worker prompt, no spawn_agent          │
│  Synthesis:       synthesizer prompt, code + final_answer│
└──────────────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  tool_store.py                                           │
│                                                          │
│  search_web      Serper → auto-fallback to DuckDuckGo   │
│  fetch_url       HTTP fetch + HTML → text extraction     │
│  read_pdf        PDF download + text extraction          │
│  execute_code    Sandboxed execution via Apptainer       │
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
| Search resilience | Primary backend (Serper) with automatic DuckDuckGo fallback on credit/auth errors |
| Empty `final_answer` | Smarter nudge echoes the model's own text back; structured backfill scans prior responses |
| Budget management | Progressive turn warnings at 5/3/2/1 remaining; last turn restricts tools to `final_answer` only |
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
  reasoning_parser: null
  extra_args: ["--async-scheduling"]
  env:
    TIKTOKEN_CACHE_DIR: "/VData/resources/huggingface/tiktoken-cache"

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

judge:
  enabled: true
  model: "gpt-4.1-mini"
  threads: 5
```

Run it:

```bash
python orchestrate.py --config configs/experiments/gpt_oss_deepsearchqa.yaml
```

The orchestrator will:
1. Pull the Apptainer SIF if not cached, start the sandbox container
2. Launch vLLM with the correct model, GPU, parser, and flags
3. Wait for both services to be healthy
4. Run the eval on the configured dataset
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
│   ├── utils.py                      # Shared utilities
│   └── apptainer.sh                  # Sandbox container pull + run
│
├── evals/
│   ├── eval.py                       # Dataset evaluation runner
│   ├── llm_judge.py                  # LLM-as-judge grading
│   └── recover_parquet.py            # Rebuild results from trace JSONs
│
├── data/                             # Evaluation outputs (parquets, traces)
├── traces/                           # Ad-hoc trace storage
└── docs/                             # Screenshots and diagrams
```

---

## Tools

| Tool | Description |
|------|-------------|
| `execute_code` | Run code in an Apptainer sandbox (Python + 40 languages). Supports file upload/download as base64. |
| `search_web` | Web search via Serper (primary) with automatic DuckDuckGo fallback. |
| `fetch_url` | Fetch a web page and extract readable text. |
| `read_pdf` | Download and extract text from a PDF. |
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

1. `eval.py` runs the agent on each dataset sample, saving per-item traces and a `results.parquet`
2. `llm_judge.py` grades each `(question, response)` pair using an OpenAI judge model
3. Outputs: `results.parquet`, `results_judge_ratings.json` with per-item and aggregate metrics

**Standalone usage:**

```bash
# Just eval (no judge)
python evals/eval.py --config configs/experiments/gpt_oss_deepsearchqa.yaml

# Just judge (on existing results)
python evals/llm_judge.py --results data/google_deepsearchqa/gpt_oss_20b/results.parquet

# Recover parquet from trace JSONs (if eval crashed mid-run)
python evals/recover_parquet.py
```

### Preliminary Results

Evaluated on **DeepSearchQA** (n = 62) with `gpt-oss-20b` served via vLLM, judged by GPT-4.1-mini:

| Metric | Value |
|--------|-------|
| Fully Correct | 30.65 ± 11.48% |
| Fully Incorrect | 45.16 ± 12.39% |
| Correct w/ Extraneous | 6.45 ± 6.12% |
| F1 | 46.44% |

> **Caveats:** ~42% of samples hit search credit limits mid-run (pre-DuckDuckGo fallback). Small sample size (n = 62) gives wide CIs. This validates architectural stability, not leaderboard performance.

---

## Configuration

Settings are loaded from YAML with fallback chain: explicit path → `TRAJECTORYKIT_CONFIG` env var → `configs/default.yaml` → hardcoded defaults.

### Key config sections

| Section | Purpose |
|---------|---------|
| `model` | Model name and API URL |
| `vllm` | Server port, GPU devices, parser, memory utilization, extra args |
| `sandbox` | Sandbox URL, port, SIF image, Docker URI |
| `agent` | Recursion depth, sub-agent turn budget, safety margin |
| `model_profiles` | Per-model settings (context window, temperature, reasoning effort) |
| `prompts` | Paths to system prompt files (orchestrator, worker, synthesizer) |
| `dataset` | HuggingFace dataset name, split, sample size, seed |
| `eval` | Turn length, reasoning effort, verbosity |
| `judge` | Enable/disable, judge model, thread count |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `SERPER_API_KEY` | Primary web search (Serper.dev). Falls back to DuckDuckGo if not set or credits exhausted. |
| `OPENAI_API_KEY` | LLM judge (GPT-4.1-mini by default). Only needed if `judge.enabled: true`. |
| `SERP_API_KEY` | Legacy SerpAPI key (alternative search backend, set `SEARCH_BACKEND=serpapi`). |

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
