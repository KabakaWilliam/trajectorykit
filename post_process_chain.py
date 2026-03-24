#!/usr/bin/env python3
"""
Post-processing script that applies a chain of rewrite prompts to draft articles.
Chain: REWRITE_PROMPT_OG → REFINEMENT_PROMPT_V2 → REFINEMENT_PROMPT_V3 → REFINEMENT_PROMPT_V3_B

Processes either:
1. A directory of text files (each draft in a separate file) — no research question needed
2. A JSONL file (one JSON record per line with 'article' and 'prompt' fields)

Usage:
    # Process a directory of text files (standalone articles)
    python post_process_chain.py -i ./drafts/ -o ./refined/ -c 3
    
    # Process JSONL file with per-record research questions
    python post_process_chain.py -i articles.jsonl -o refined_articles.jsonl
    
    # Test mode
    python post_process_chain.py -i articles.jsonl -o test_output.jsonl --test
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY', '')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')

from openai import OpenAI
from tqdm import tqdm

# Add deep_research_bench to path to import prompts
sys.path.insert(0, str(Path(__file__).parent / 'deep_research_bench'))

from rewrite_articles import (
    REWRITE_PROMPT_OG,
    REFINEMENT_PROMPT_V2,
    REFINEMENT_PROMPT_V3,
    REFINEMENT_PROMPT_V3_B,
    PROVIDER_CONFIG,
)

# Chain of prompts to apply sequentially
REWRITE_CHAIN = [
    {"name": "REWRITE_OG", "prompt": REWRITE_PROMPT_OG},
    {"name": "REFINEMENT_V2", "prompt": REFINEMENT_PROMPT_V2},
    {"name": "REFINEMENT_V3", "prompt": REFINEMENT_PROMPT_V3},
    {"name": "REFINEMENT_V3_B", "prompt": REFINEMENT_PROMPT_V3_B},
]


def rewrite_article_single_pass(client, article_content, provider, provider_config, question, prompt_template):
    """Apply a single rewrite pass using the given prompt template."""
    try:
        prompt = prompt_template.format(
            question=question or "",
            article=article_content
        )
        
        config = provider_config[provider]
        model = config["model"]
        
        request_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_completion_tokens", config.get("max_tokens", 8192)),
        }
        
        # Add reasoning_effort for OpenAI if supported
        if provider == "openai" and "reasoning_effort" in config:
            request_params["reasoning_effort"] = config["reasoning_effort"]
        
        response = client.chat.completions.create(**request_params)
        rewritten = response.choices[0].message.content
        
        return {
            "success": True,
            "article": rewritten,
            "error": None,
        }
    except Exception as e:
        error_msg = f"Rewrite failed: {str(e)}"
        return {
            "success": False,
            "article": article_content,  # Return original on failure
            "error": error_msg,
        }


def apply_rewrite_chain(client, article_content, provider, provider_config, question):
    """Apply the full chain of rewrites sequentially."""
    current_article = article_content
    chain_results = []
    
    for i, rewrite_step in enumerate(REWRITE_CHAIN):
        step_name = rewrite_step["name"]
        prompt_template = rewrite_step["prompt"]
        
        result = rewrite_article_single_pass(
            client,
            current_article,
            provider,
            provider_config,
            question,
            prompt_template,
        )
        
        chain_results.append({
            "step": step_name,
            "success": result["success"],
            "error": result["error"],
        })
        
        current_article = result["article"]
        
        if not result["success"]:
            print(f"  ⚠ Step {i+1} ({step_name}) failed: {result['error']}")
        else:
            print(f"  ✓ Step {i+1} ({step_name}) completed")
    
    return {
        "article": current_article,
        "chain_results": chain_results,
        "success": all(r["success"] for r in chain_results),
    }


def load_input_data(input_path):
    """Load articles from either a JSONL file or directory of text files."""
    records = []
    
    input_path = Path(input_path)
    
    if input_path.suffix == ".jsonl":
        # Load from JSONL file
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        records.append({
                            "id": data.get("id", "unknown"),
                            "prompt": data.get("prompt", ""),
                            "article": data.get("article", ""),
                            "source": "jsonl",
                        })
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line")
    elif input_path.is_dir():
        # Load from directory of text files
        for file_path in sorted(input_path.glob("*.md")) + sorted(input_path.glob("*.txt")):
            with open(file_path, "r") as f:
                content = f.read()
                records.append({
                    "id": file_path.stem,
                    "prompt": "",  # No prompt for directory mode
                    "article": content,
                    "source": "directory",
                })
    else:
        raise ValueError(f"Input must be a JSONL file or directory: {input_path}")
    
    return records


def save_output_data(output_path, records, is_jsonl=True):
    """Save processed records to output file or directory."""
    output_path = Path(output_path)
    
    if is_jsonl:
        # Write to JSONL file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps({
                    "id": record["id"],
                    "prompt": record["prompt"],
                    "article": record["article"],
                    "chain_results": record.get("chain_results", []),
                    "success": record.get("success", True),
                }) + "\n")
    else:
        # Write to directory
        output_path.mkdir(parents=True, exist_ok=True)
        for record in records:
            file_path = output_path / f"{record['id']}.md"
            with open(file_path, "w") as f:
                f.write(record["article"])


def main():
    parser = argparse.ArgumentParser(
        description="Post-process drafts through a chain of rewrite prompts"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input: JSONL file or directory of text files (*.md, *.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path: JSONL file or directory (auto-detected based on input)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=2,
        help="Number of concurrent rewrite chains (default: 2)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider: openai (default) or anthropic",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first record",
    )
    args = parser.parse_args()

    # Validate provider and API key
    if args.provider == "openai":
        if not openai_api_key:
            print("Error: OPENAI_API_KEY not set")
            sys.exit(1)
        api_key = openai_api_key
    elif args.provider == "anthropic":
        if not anthropic_api_key:
            print("Error: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        api_key = anthropic_api_key

    # Initialize client
    client_kwargs = {"api_key": api_key}
    base_url = PROVIDER_CONFIG[args.provider].get("base_url")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Determine input/output format
    input_path = Path(args.input)
    output_path = Path(args.output)
    is_jsonl_input = input_path.suffix == ".jsonl"
    is_jsonl_output = output_path.suffix == ".jsonl"

    # Load input data
    print(f"Loading input from: {args.input}")
    records = load_input_data(args.input)
    print(f"Loaded {len(records)} records")

    # Test mode
    if args.test:
        records = records[:1]
        args.concurrency = 1
        print("TEST MODE: Processing 1 record only")

    # Process records with thread pool
    print(f"\nProvider: {args.provider}")
    print(f"Model: {PROVIDER_CONFIG[args.provider]['model']}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {args.output}")
    print(f"Rewrite chain: {' → '.join([s['name'] for s in REWRITE_CHAIN])}\n")

    output_lock = threading.Lock()

    def process_record(record):
        """Apply full rewrite chain to a record."""
        article_id = record["id"]
        print(f"\nProcessing: {article_id}")
        
        result = apply_rewrite_chain(
            client,
            record["article"],
            args.provider,
            PROVIDER_CONFIG,
            record["prompt"],
        )
        
        # Update record with result
        record["article"] = result["article"]
        record["chain_results"] = result["chain_results"]
        record["success"] = result["success"]
        
        return record

    # Execute with thread pool
    processed_records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_record, record): record["id"] for record in records}

        with tqdm(total=len(records), desc="Processing articles") as pbar:
            for future in as_completed(futures):
                try:
                    processed_record = future.result()
                    processed_records.append(processed_record)
                except Exception as e:
                    article_id = futures[future]
                    print(f"Error processing {article_id}: {e}")
                finally:
                    pbar.update(1)

    # Save output
    print(f"\nSaving results to: {args.output}")
    save_output_data(args.output, processed_records, is_jsonl=is_jsonl_output)

    # Print summary
    success_count = sum(1 for r in processed_records if r.get("success", False))
    print(f"\n✓ Completed: {success_count}/{len(processed_records)} successful")
    if success_count < len(processed_records):
        print(f"⚠ Failed: {len(processed_records) - success_count} (fallback to original)")


if __name__ == "__main__":
    main()
