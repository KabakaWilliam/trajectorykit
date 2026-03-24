# Development Guide - TrajectoryKit

This guide covers working with TrajectoryKit's internal components and optional external tools.

## Repository Structure

```
trajectorykit/                       # Main repository (your installation)
├── post_process_chain.py           # ⚙️ Batch refinement script (at root level)
├── src/                            # Core agent harness
├── configs/                        # Experiment configurations
├── evals/                          # Evaluation scripts
├── docs/                           # Documentation
├── README.md                       # Main README
└── deep_research_bench/            # ⚙️ Cloned external tool [optional]
    ├── rewrite_articles.py        # Refinement prompts & logic
    ├── POST_PROCESS_CHAIN_README.md
    ├── data/                      # Benchmark datasets
    └── utils/                     # Utility functions
```

**Note:** Clone `deep_research_bench` as a separate repository when you need batch post-processing. The `post_process_chain.py` script is at the TrajectoryKit root and imports prompts from `deep_research_bench/rewrite_articles.py`.

## Working with Refinement Prompts

The refinement pipeline consists of four sequential passes defined in `deep_research_bench/rewrite_articles.py`:

### 1. REWRITE_PROMPT_OG
**Purpose:** Foundation enhancement — quantify claims, fill gaps, strengthen causal reasoning.

**Key directives:**
- Replace vague language with specific numbers/benchmarks
- Add missing entity coverage (vendors, methodologies, case studies)
- Reduce scaffold text (methodology exposition, meta-commentary)
- Execute frameworks with worked examples
- Ground risks in real incidents
- Build consolidated comparison tables

**Length target:** ~50% longer than input

### 2. REFINEMENT_PROMPT_V2
**Purpose:** Comprehensive depth — mechanistic understanding and grounding.

**Key directives:**
- Ensure full task scope coverage (substantively, not just mentioned)
- Close coverage gaps within existing categories
- Strengthen structural clarity with section context
- Add missing analytical depth (boundary conditions, alternatives, related constraints)
- Ground abstract claims with concrete examples
- Deepen comparative analysis and mechanistic reasoning

**Target:** Maintain or slightly grow (+5-10% from input)

### 3. REFINEMENT_PROMPT_V3
**Purpose:** Natural narrative clarity — calibrated signposting and readability.

**Key directives:**
- Natural paragraph structure with topic sentences
- Calibrated signposting (use ONLY where connections aren't obvious)
- Light section framing (1-2 sentences per section opening/closing)
- Enhance scannability where sections are complex
- Strengthen connections between ideas
- Verify completeness against original task requirements

### 4. REFINEMENT_PROMPT_V3_B
**Purpose:** Strategic scannability — finding-based organization and decision relevance.

**Key directives:**
- Decompose task requirements (internal planning only, not output)
- Map current content to requirements
- Add missing coverage with finding-level synthesis
- Strengthen mechanistic depth with worked decision examples
- Use finding-based subheadings (not generic categories)
- Mandatory strong topic sentences surfacing actual findings
- Calibrated signposting with section framing
- Final requirement validation checkpoint

## Using the Post-Processing Chain

### Installation

```bash
# Clone deep_research_bench for the refinement prompts
git clone https://github.com/KabakaWilliam/deep_research_bench.git
```

### Basic Usage

From TrajectoryKit root:

```bash
# Process a directory of drafts
python post_process_chain.py \
  -i ./my_drafts/ \
  -o ./refined/ \
  -c 3

# Process JSONL file
python post_process_chain.py \
  -i articles.jsonl \
  -o refined_articles.jsonl

# Test mode (single record)
python post_process_chain.py \
  -i articles.jsonl \
  -o test_output.jsonl \
  --test
```

### Configuration

**Environment variables:**
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # Optional: for Anthropic
```

Or use a `.env` file in the repo root.

**Command-line options:**
```
-i, --input           Input: JSONL file or directory of text files
-o, --output          Output path: JSONL file or directory
-c, --concurrency     Number of concurrent workers (default: 2)
--provider {openai,anthropic}  LLM provider (default: openai)
--test                Process only first record
```

### Input/Output Formats

**JSONL format:**
```json
{"id": "article1", "prompt": "What are X trends?", "article": "Article text..."}
{"id": "article2", "prompt": "How does Y work?", "article": "Article text..."}
```

**Directory format:**
- Input: `drafts/*.md` or `drafts/*.txt`
- Output: `refined/*.md` (or `.txt`)
- Filename (without extension) used as article ID

### Output Structure

**JSONL output:**
```json
{
  "id": "article1",
  "prompt": "original question",
  "article": "fully refined text",
  "chain_results": [
    {"step": "REWRITE_OG", "success": true, "error": null},
    {"step": "REFINEMENT_V2", "success": true, "error": null},
    {"step": "REFINEMENT_V3", "success": true, "error": null},
    {"step": "REFINEMENT_V3_B", "success": true, "error": null}
  ],
  "success": true
}
```

Each article includes detailed chain results showing success/failure at each refinement stage.

## Extending the Refinement Chain

To add new refinement stages, work in the external `deep_research_bench` repository:

### Step 1: Define a New Prompt

In `deep_research_bench/rewrite_articles.py`, add:

```python
REFINEMENT_PROMPT_V4 = """
You are an expert research report editor specializing in [new focus].
Your task is to [specific enhancement goal].

<ORIGINAL RESEARCH QUESTION>
{question}
</ORIGINAL RESEARCH QUESTION>

[Your detailed enhancement instructions...]

ORIGINAL REPORT:
{article}"""
```

### Step 2: Update the Post-Processing Chain

In `deep_research_bench/post_process_chain.py`, add to the `REWRITE_CHAIN` list:

```python
REWRITE_CHAIN = [
    {"name": "REWRITE_OG", "prompt": REWRITE_PROMPT_OG},
    {"name": "REFINEMENT_V2", "prompt": REFINEMENT_PROMPT_V2},
    {"name": "REFINEMENT_V3", "prompt": REFINEMENT_PROMPT_V3},
    {"name": "REFINEMENT_V3_B", "prompt": REFINEMENT_PROMPT_V3_B},
    {"name": "REFINEMENT_V4", "prompt": REFINEMENT_PROMPT_V4},  # Add here
]
```

### Step 3: Test

Run in test mode to validate on a single record:

```bash
cd ../deep_research_bench
python post_process_chain.py \
  -i test_article.jsonl \
  -o test_output.jsonl \
  --test
```

Check the output's `chain_results` to verify your new stage succeeded.

## Performance Notes

### API Costs and Token Usage

Each article goes through **4 passes**, so API costs scale accordingly:

| Provider | Model | Pass | Max Tokens | Est. Cost/1K Pages |
|----------|-------|------|-----------|-------------------|
| OpenAI | GPT-5.4 + reasoning: high | 4× | 128K | ~$40-50 |
| Anthropic | Claude-Opus-4.6 | 4× | 128K | ~$30-40 |
| Local | GPT-OSS-20B | 1× per pass | N/A | Free (GPU time) |

### Concurrency Recommendations

- **Default (2 workers):** Safe, minimal rate-limiting risk
- **3-5 workers:** Good balance for batch processing
- **>5 workers:** Monitor for API rate limits; consider slowing requests

## Troubleshooting

### Import Errors

If post-processing script fails to import, verify `rewrite_articles.py` is accessible:

```bash
cd ../deep_research_bench
python -c "from rewrite_articles import *; print('✓ Imports OK')"
```

### API Key Issues

Check environment variables:

```bash
echo "OpenAI API Key set: [$([ -n $OPENAI_API_KEY ] && echo 'yes' || echo 'no')]"
echo "Anthropic API Key set: [$([ -n $ANTHROPIC_API_KEY ] && echo 'yes' || echo 'no')]"
```

### Slow Performance

**Approach 1:** Reduce token limits in `deep_research_bench/rewrite_articles.py`

```python
PROVIDER_CONFIG["openai"]["max_completion_tokens"] = 64000  # Down from 128K
```

**Approach 2:** Use a faster provider for initial passes, GPT-5.4 for final pass

**Approach 3:** Reduce concurrency to lower API queue depth

### Partial Failures

If a stage fails, the script falls back to the previous stage's output and logs the error in `chain_results`. Check the error message:

```json
{"step": "REFINEMENT_V2", "success": false, "error": "Rate limit exceeded"}
```

## Project Integration

### Using Post-Processing in Your Workflow

**Example: Research → Refine → Evaluate**

Assuming you've cloned `deep_research_bench` as a sibling directory:

```bash
# 1. Generate draft articles
python orchestrate.py --config configs/experiments/gpt_oss_deep_research_bench.yaml

# 2. Refine outputs with external post-processing chain
python ../deep_research_bench/post_process_chain.py \
  -i data/google_deepsearchqa/gpt_oss_20b/traces/*.json \
  -o refined_results.jsonl

# 3. Re-evaluate refined articles
python evals/llm_judge.py --results refined_results.jsonl
```

### Optional: Create an Alias

To make the command shorter, create a shell alias:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias post_process='python ../deep_research_bench/post_process_chain.py'

# Then use directly:
post_process -i drafts/ -o refined/
```

Or use as a Git submodule (optional):

```bash
# In trajectorykit root:
git submodule add https://github.com/KabakaWilliam/deep_research_bench.git tools/deep_research_bench

# Then reference it:
python tools/deep_research_bench/post_process_chain.py -i drafts/ -o refined/
```

## Contributing to Deep Research Bench

When adding new refinement stages or improving existing prompts to the external repository:

1. **Work in the external repository** — Make changes in the `deep_research_bench` repo
2. **Document the change** — Add rationale to prompt docstring
3. **Test on benchmark** — Verify improvement on Deep Research Bench (eval/evaluate.py)
4. **Add to chain** — Update post_process_chain.py with new stage
5. **Report performance** — Include scores before/after in PR description
6. **Push upstream** — Create a PR to the main `deep_research_bench` repository

---

For questions on the refinement pipeline or post-processing, see the [main README](README.md#-post-processing-chain-batch-article-refinement) or visit the external [deep_research_bench](https://github.com/KabakaWilliam/deep_research_bench) repository.
