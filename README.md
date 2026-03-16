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
│  → runs eval → runs LLM judge / RACE scorer             │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  agent.py → agent_state.py → runner.py → nodes.py       │
│                                                          │
│  agent.py:       dispatch() entry, config_path reload,   │
│                  pre-dispatch chain analysis (chain.py)   │
│  agent_state.py: AgentState dataclass + create_state()   │
│  runner.py:      turn loop, 5 cycle gates, plan inject,  │
│                  history compaction, synthesis pipeline   │
│  nodes.py:       tool handlers, 4-stage verification     │
│  plan.py:        ResearchPlan + PlanTask tracking         │
│  config.py:      YAML loader, prompt loader, 30+ keys    │
│                                                          │
│  Depth 0 (root):  orchestrator prompt, root tools        │
│  Depth 1+ (sub):  worker prompt, worker tools            │
│  Synthesis:       synthesizer prompt, code + final_answer│
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  tool_store.py  (4,200+ lines)                           │
│                                                          │
│  ROOT TOOLS (orchestrator only):                         │
│  conduct_research  Delegate task → sub-agent             │
│  refine_draft      Write/replace the full draft report   │
│  read_draft        Read previous draft versions/feedback │
│  research_complete Publish draft (triggers verification) │
│  summarize_webpage Fetch URL + LLM-focused summary       │
│  think             Reasoning scratchpad                  │
│                                                          │
│  WORKER TOOLS (sub-agents):                              │
│  search_web        Serper → Exa → DuckDuckGo fallback   │
│  fetch_url         Direct → Jina → Exa → Wayback chain  │
│  read_page         Paginate cached page text (no I/O)    │
│  extract_tables    HTML tables → structured JSON         │
│  read_pdf          PDF download + text extraction        │
│  wikipedia_lookup  MediaWiki API → article/section/info  │
│  fetch_cached      Wayback Machine archived pages        │
│  execute_code      Sandboxed execution via Apptainer     │
│  recall_memory     Retrieve stored tool outputs by key   │
│  spawn_agent       Recursive sub-agent (fresh context)   │
│  final_answer      Terminates the agent loop             │
│  think             Reasoning scratchpad                  │
│  search_available_tools  Self-introspection              │
└──────────────────────────────────────────────────────────┘
```

### Agent Flow

The full lifecycle of a `dispatch()` call, from question to verified answer:

```
                        User Question
                             │
                             ▼
              ┌──────────────────────────────┐
              │     ⛓ CHAIN ANALYSIS         │
              │     (pre-dispatch, root only) │
              │     [External/Local]          │
              │  (e.g. GPT-5.4, Claude)      │
              │                              │
              │  LLM detects causal chains:  │
              │    Step 1: find X → {step_1} │
              │    Step 2: Y of {step_1}     │
              │         → {step_2}           │
              │    Step 3: Z of {step_2}     │
              │         → final answer       │
              │                              │
              │  → ChainPlan on state        │
              │  → Seeds ResearchPlan        │
              └──────────┬───────────────────┘
                         │
                         ▼
     ┌───────────────────────────────────────────────┐
     │            🔄 RESEARCH LOOP                   │
     │            (runner.py)                         │
     │     [External/Local] ↔ Interchangeable        │
     │                                               │
     │  ┌─────────────────────────────────────────┐  │
     │  │ Each turn:                              │  │
     │  │   LLM → tool calls → execute → results │  │
     │  │   [External/Local model routes here]   │  │
     │  │         ▲                     │         │  │
     │  │         └─────────────────────┘         │  │
     │  └─────────────────────────────────────────┘  │
     │                                               │
     │  Cycle enforcement (5 gates):                 │
     │    G1: can't publish without a draft          │
     │    G2: can't re-publish without revising      │
     │    G3: can't draft without researching         │
     │        (1 round for QA, 2 for report mode)    │
     │    G4: must follow plan when active            │
     │    G5: report mode — must research after first │
     │        draft before publishing                │
     │                                               │
     │  History compaction:                          │
     │    • Evicts old messages when threshold hit   │
     │    • Preserves system prompt + recent turns   │
     │    • Configurable threshold/interval/window   │
     │                                               │
     │  Chain enforcement:                           │
     │    • Chain plan injected at turns 0–2         │
     │    • Tasks with unresolved placeholders       │
     │      are hard-blocked                         │
     │    • Sub-agent results matched to chain       │
     │      steps → placeholders resolved            │
     │                                               │
     │  ┌─────────────────────────────────────────┐  │
     │  │ Sub-agents (depth 1+)                   │  │
     │  │  • Fresh context, worker prompt         │  │
     │  │  • Full tools: search, fetch, read_pdf, │  │
     │  │    wikipedia, extract_tables, code,     │  │
     │  │    read_page, recall_memory             │  │
     │  │  • Return findings via final_answer     │  │
     │  │  • [External/Local] at root             │  │
     │  │    (local recommended to minimize cost) │  │
     │  └─────────────────────────────────────────┘  │
     │                                               │
     │  Root tools: conduct_research, refine_draft,  │
     │    read_draft, think, summarize_webpage,       │
     │    research_complete                           │
     └──────────────────┬────────────────────────────┘
                        │
               research_complete()
                        │
                        ▼
     ┌───────────────────────────────────────────────┐
     │       🛡️ VERIFICATION PIPELINE (4 stages)     │
     │                                               │
     │  Stage 0 ── Format Gate                       │
     │  │  Report mode: ## Executive Summary +       │
     │  │    ## Sources                              │
     │  │  QA mode: **Final Answer:** + **Sources:** │
     │  │  Fail → back to research loop              │
     │  ▼                                            │
     │  Stage 1 ── Verifier LLM                      │
     │  │  [External/Local] ↔ Interchangeable       │
     │  │  9 criteria: adequacy (hard-stop),         │
     │  │  language, depth, insight, comprehensiveness│
     │  │  section quality, citations, coherence,    │
     │  │  conflicts, gaps                           │
     │  │  Fail → back to loop with feedback         │
     │  ▼                                            │
     │  Stage 2 ── Spot-Check (sub-agent powered)    │
     │                                               │
     │  ┌─────────────────────────────────────────┐  │
     │  │ Step 1: EXTRACT                         │  │
     │  │  [External/Local] ↔ Interchangeable     │  │
     │  │  LLM extracts checkable claims from     │  │
     │  │  draft. Chain plan injected as hint.     │  │
     │  │  Claim #1 = core answer (mandatory).    │  │
     │  └────────────────┬────────────────────────┘  │
     │                   ▼                           │
     │  ┌─────────────────────────────────────────┐  │
     │  │ Step 2: VERIFY (parallel sub-agents)    │  │
     │  │  ** Sub-agents: Local models only **    │  │
     │  │  (spawned at root, separate context)    │  │
     │  │                                         │  │
     │  │  Each claim → 1 verification sub-agent  │  │
     │  │  (3 turns, fresh context, full tools)   │  │
     │  │                                         │  │
     │  │  The sub-agent can:                     │  │
     │  │    • Search the web                     │  │
     │  │    • Fetch & read cited source URLs     │  │
     │  │    • Open PDFs, gov sites, .edu pages   │  │
     │  │    • Use wikipedia_lookup               │  │
     │  │    • Think & try different queries       │  │
     │  │    • Report: sources, assessment,       │  │
     │  │      corrections                        │  │
     │  │                                         │  │
     │  │  Up to 4 sub-agents run in parallel     │  │
     │  └────────────────┬────────────────────────┘  │
     │                   ▼                           │
     │  ┌─────────────────────────────────────────┐  │
     │  │ Step 3: COMPARE (LLM judge)             │  │
     │  │  [External/Local] ↔ Interchangeable     │  │
     │  │  Claims + sub-agent evidence reports    │  │
     │  │  → 4-point core answer check            │  │
     │  │  → Chain coherence verification         │  │
     │  │  → SPOT_CHECK_PASSED / FAILED           │  │
     │  └─────────────────────────────────────────┘  │
     │                                               │
     │  Refusal path:                                │
     │    No claims → refusal detected →             │
     │    sub-agent investigates independently →      │
     │    REFUSAL_JUSTIFIED / REFUSAL_CHALLENGED     │
     │                                               │
     │  Stage 3 ── Citation Audit                    │
     │  │  [External/Local] ↔ Interchangeable       │
     │  │  Extracts [N]→URL pairs from Sources       │
     │  │  Fetches each cited URL (cache-first)      │
     │  │  LLM judges: does cited URL support the    │
     │  │    specific claim attributed to it?         │
     │  │  ≥3 UNSUPPORTED citations → FAIL           │
     │  ▼                                            │
     │  Post-processing:                             │
     │    Linkification: [N] → [[N]](url)            │
     │                                               │
     │  Fail → back to research loop with feedback   │
     │  Pass → publish                               │
     └──────────────────┬────────────────────────────┘
                        │
                        ▼
              ┌──────────────────────────────┐
              │     ✅ PUBLISHED ANSWER      │
              │                              │
              │  Final response returned     │
              │  Trace: JSON + HTML          │
              │  Chain panel shows step      │
              │  resolution in HTML viewer   │
              └──────────────────────────────┘
```

### Key design decisions

| Concern | Approach |
|---------|----------|
| Model interchangeability | Most LLM steps (chain analysis, research loop, verification stages 1/3) accept external (GPT-5.4, Claude) or local models as drop-in replacements. Sub-agents use independent model configuration; verification sub-agents (spawned at root) may use local models to minimize cost. Configure via `model.name` and `model.api_url` in YAML. |
| Causal chains | Pre-dispatch LLM detects dependency chains → `ChainPlan` enforces sequential resolution with placeholder blocking |
| Verification | 4-stage pipeline: format gate → verifier LLM (insight criterion) → sub-agent spot-check → citation audit (URL faithfulness) |
| Cycle prevention | 5 runner gates: must draft before publish, must revise before re-publish, must research before draft, must follow plan, must research after first draft (report mode) |
| Context management | History compaction evicts old messages when threshold hit, keeping system prompt + recent turns. `MemoryStore` captures full tool outputs externally; synthesis sub-agent queries them via `execute_code` |
| Search resilience | Serper (primary) → Exa.ai neural search → DuckDuckGo — three-tier automatic fallback on credit/auth/rate errors |
| Fetch resilience | Direct HTTP → Jina Reader (headless Chrome, GET then POST) → Exa contents API → Wayback Machine — four-tier fallback with domain-aware routing |
| Draft workflow | Root orchestrator writes via `refine_draft`, publishes via `research_complete`. Draft versions tracked. Verifier pushes back on shallow/low-insight reports. |
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
  sub_agent_turn_budget: 10

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
│   │   ├── gpt_oss_deepsearchqa.yaml          # GPT-OSS-20B on DeepSearchQA
│   │   ├── gpt_oss_deep_research_bench.yaml   # GPT-OSS-20B on Deep Research Bench
│   │   └── qwen3_deepsearchqa.yaml            # Qwen3-8B on DeepSearchQA
│   └── prompts/
│       ├── orchestrator.txt          # Root agent system prompt
│       ├── orchestrator_bench.txt    # Bench-mode orchestrator (report format)
│       ├── worker.txt                # Sub-agent system prompt
│       ├── synthesizer.txt           # Synthesis sub-agent prompt
│       ├── chain_analysis.txt        # Chain detection prompt
│       ├── verifier.txt              # Stage 1 verification prompt
│       ├── verifier_bench.txt        # Bench-mode verifier (insight criterion)
│       ├── spotcheck_extract.txt     # Stage 2 — claim extraction
│       ├── spotcheck_compare.txt     # Stage 2 — evidence comparison
│       ├── spotcheck_compare_bench.txt # Bench-mode spot-check compare
│       ├── spotcheck_refusal.txt     # Stage 2 — refusal challenge
│       └── citation_audit.txt        # Stage 3 — citation faithfulness
│
├── src/trajectorykit/
│   ├── __init__.py                   # Public API (dispatch, EpisodeTrace, render_trace_html)
│   ├── agent.py                      # Entry point: dispatch() + chain analysis
│   ├── agent_state.py                # AgentState dataclass + create_state()
│   ├── runner.py                     # Turn loop, 5 cycle gates, history compaction
│   ├── nodes.py                      # Tool handlers + 4-stage verification pipeline
│   ├── chain.py                      # Causal chain analysis (ChainPlan/ChainStep)
│   ├── plan.py                       # ResearchPlan + PlanTask tracking
│   ├── config.py                     # YAML config loader + 30+ configurable keys
│   ├── tool_store.py                 # Tool definitions (4,200+ lines)
│   ├── tracing.py                    # Trace dataclasses + HTML renderer
│   ├── memory.py                     # MemoryStore (compressed tool output storage)
│   ├── symbolic.py                   # Symbolic reference compression
│   ├── utils.py                      # Shared utilities
│   └── apptainer.sh                  # Sandbox container pull + run
│
├── deep_research_bench/              # 📊 Evaluation Benchmark
│   ├── README.md                     # Benchmark setup, results, and evaluation framework
│   ├── deepresearch_bench_race.py    # RACE (Reference-based Adaptive Criteria) evaluator
│   ├── rewrite_articles.py           # ✨ Post-processing rewrite with GPT-5.4/Claude
│   ├── run_benchmark.sh              # End-to-end benchmark runner
│   ├── data/
│   │   ├── deep_research_bench/      # 100 benchmark queries (22 domains)
│   │   ├── test_data/                # Agent outputs (JSONL format)
│   │   └── criteria_data/            # Evaluation rubrics
│   ├── results/race/                 # RACE eval results per model
│   └── results/fact/                 # FACT eval results per model
│
├── evals/
│   ├── eval.py                       # DeepSearchQA evaluation runner
│   ├── eval_deep_research_bench.py   # Deep Research Bench eval (RACE + FACT)
│   ├── llm_judge.py                  # LLM-as-judge grading
│   ├── recover_parquet.py            # Rebuild results from trace JSONs
│   └── traces_to_parquet.py          # Convert trace JSONs → parquet
│
├── data/                             # Evaluation outputs (parquets, traces)
├── traces/                           # Ad-hoc trace storage
└── docs/                             # Screenshots and diagrams
```

---

## Tools

### Root tools (orchestrator only)

| Tool | Description |
|------|-------------|
| `conduct_research` | Delegate a research task to a sub-agent with its own context window, tools, and trace. |
| `refine_draft` | Write or replace the entire draft report. Each call overwrites the previous version. |
| `read_draft` | Read previous draft versions and verifier feedback history. |
| `research_complete` | Publish the draft — triggers the 4-stage verification pipeline. |
| `summarize_webpage` | Fetch a URL and return an LLM-generated summary focused on a specific topic. Faster than `conduct_research` for targeted reads. |
| `think` | Reasoning scratchpad — output is not shown to the user. |
| `search_available_tools` | Self-introspection — list tools or get full schema for any tool. |

### Worker tools (sub-agents)

| Tool | Description |
|------|-------------|
| `search_web` | Web search via Serper → Exa.ai → DuckDuckGo (three-tier automatic fallback). |
| `fetch_url` | Fetch a web page with four-tier fallback: direct HTTP → Jina Reader → Exa contents → Wayback Machine. Supports `css_selector` for targeted extraction and `extract="table"` mode. |
| `read_page` | Paginate through cached page text from a previous `fetch_url` call — no network I/O. |
| `extract_tables` | Extract HTML tables from a URL as structured JSON (list of row-dicts). |
| `read_pdf` | Download and extract text from a PDF. |
| `wikipedia_lookup` | Look up a Wikipedia article directly via the MediaWiki API. Returns article text, section list, and structured infobox data. |
| `fetch_cached` | Fetch an archived version of a URL from the Wayback Machine. |
| `execute_code` | Run code in an Apptainer sandbox (Python + 40 languages). Supports file upload/download as base64. |
| `recall_memory` | Retrieve stored tool outputs by key. Query within compressed memory for specific data. |
| `spawn_agent` | Spawn a recursive sub-agent with fresh context and its own trace. |
| `final_answer` | Submit the final answer (terminates the loop). |
| `think` | Reasoning scratchpad. |

---

## Tracing

Every `dispatch()` call produces a full execution trace.

### Terminal output

```python
result["trace"].pretty_print()
```

```
🏁 Agent [root]  trace_id=a3f7c912
  Input: What country has the largest population in Africa?
  Duration: 22.14s | Turns: 3 | Tool calls: 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ┌─ Turn 1 (6.21s)
  │  🔧 search_web({"query": "largest population Africa"}) [1.03s]
  │     → 1. Nigeria — 223.8 million (2024) ...
  │  🔧 wikipedia_lookup({"title": "Nigeria", "section": "Demographics"}) [0.84s]
  │     → Population: 223,804,632 (2024 estimate) ...
  └─
  ┌─ Turn 2 (4.12s)
  │  🔧 extract_tables({"url": "https://en.wikipedia.org/wiki/..."}) [1.22s]
  │     → [{"Country": "Nigeria", "Population": "223804632", ...}, ...]
  └─
  ┌─ Turn 3 (3.80s)
  │  🔧 final_answer({"answer": "Nigeria, with ~224 million people"}) [0.00s]
  └─
📊 Episode Summary:
  Prompt tokens:       12,450
  Completion tokens:    1,820
  Total tokens:        14,270
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

## 🏆 Deep Research Bench Evaluation

TrajectoryKit is evaluated on **[Deep Research Bench](deep_research_bench/README.md)**, a comprehensive benchmark with 100 PhD-level research tasks across 22 domains. The framework includes a **post-processing enhancement step** using GPT-5.4 with high reasoning effort:

### Results with Post-Processing

| Metric | Baseline | With Rewrite | Improvement |
|--------|----------|--------------|-------------|
| Comprehensiveness | 0.4543 | **0.5078** | +11.8% |
| Insight/Depth | 0.4373 | **0.5472** | +25.2% |
| Instruction Following | 0.4825 | **0.5215** | +8.1% |
| Readability | 0.4861 | **0.5213** | +7.2% |
| **Overall Score** | **0.4597** | **0.5266** | **+14.6%** |

### Post-Processing Pipeline

After the agent generates a draft report, an optional **high-reasoning rewrite step** enhances quality:

```bash
# Rewrite with GPT-5.4 (high reasoning, 128k tokens)
python deep_research_bench/rewrite_articles.py \
  -i traces/research_results.jsonl \
  -o rewrite_research_results.jsonl \
  -c 5

# Or use Anthropic Claude Opus
python deep_research_bench/rewrite_articles.py \
  --provider anthropic \
  -i traces/research_results.jsonl \
  -o rewrite_research_results.jsonl
```

**Key enhancements applied:**
- Quantify vague claims with specific metrics and benchmarks
- Deepen entity and case study coverage
- Reduce scaffolding and eliminate redundancy
- Execute frameworks with worked examples
- Ground risks in real-world incidents
- Specify regulatory and standards content
- Update stale reference points
- Build consolidated comparison tables
- Strengthen causal reasoning
- Improve source quality framing

For detailed evaluation methodology and benchmark setup, see [Deep Research Bench README](deep_research_bench/README.md).

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

### Results

Evaluated on **DeepSearchQA** (n = 100, eval split) with `gpt-oss-20b` (131K context) served via vLLM, judged by GPT-4.1-mini:

| Metric | Value |
|--------|-------|
| Fully Correct | 46.88 ± 9.98% (45/96) |
| Fully Incorrect | 31.25 ± 9.27% (30/96) |
| Correct w/ Extraneous | 5.21 ± 4.44% (5/96) |
| Precision | 63.89% |
| Recall | 61.14% |
| F1 | 61.08% |

96 of 100 items received valid judge ratings (1 empty model response, 3 invalid rater responses).

> **Note:** DeepSearchQA questions are multi-hop research tasks requiring the agent to search, synthesize, and reason over multiple sources. The dataset often expects multi-part answers (e.g. lists of names, dates, or figures).

---

## Configuration

Settings are loaded from YAML with fallback chain: explicit path → `TRAJECTORYKIT_CONFIG` env var → `configs/default.yaml` → hardcoded defaults.

### Key config sections

| Section | Purpose |
|---------|---------|
| `model` | Model name and API URL |
| `vllm` | Server port, GPU devices, parser, memory utilization, extra args |
| `sandbox` | Sandbox URL, port, SIF image, Docker URI |
| `agent` | Recursion depth, sub-agent turn budget, safety margin, draft format, verification flags, compaction settings |
| `model_profiles` | Per-model settings (context window, temperature, reasoning effort) |
| `prompts` | Paths to system prompt files (orchestrator, worker, synthesizer) |
| `dataset` | HuggingFace dataset name, split, sample size, seed |
| `eval` | Turn length, reasoning effort, verbosity |
| `judge` | Enable/disable, judge model, thread count |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `SERPER_API_KEY` | Primary web search (Serper.dev). Falls back to Exa → DuckDuckGo if not set or credits exhausted. |
| `EXA_API_KEY` | Exa.ai neural search (fallback) + Exa contents API (fetch fallback). Highly recommended. |
| `JINA_API_KEY` | Jina Reader API for fetching JS-heavy / paywalled pages. Optional but improves fetch success rate. |
| `OPENAI_API_KEY` | OpenAI API key for LLM judge (GPT-4.1-mini) and post-processing rewrite (GPT-5.4). Only needed for these features. |
| `ANTHROPIC_API_KEY` | Anthropic API key for alternative post-processing rewrite (Claude Opus). Optional. |
| `SERP_API_KEY` | Legacy SerpAPI key (alternative search backend, set `SEARCH_BACKEND=serpapi`). |
| `GOOGLE_API_KEY` | Gemini API key. Required only when using Gemini models as the LLM judge. |

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
    config_path="configs/experiments/gpt_oss_deep_research_bench.yaml",  # Optional config override
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
