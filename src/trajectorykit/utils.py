import json
import logging
import os
import threading
import time
import traceback
import uuid
import re
import requests

from typing import Any, Optional, Dict, List
# from tools import TOOL_REGISTRY

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)

# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


# -----------------------------
# Tool calling helpers
# -----------------------------

def _decode_new_tokens(tokenizer, inputs, outputs) -> str:
    """Decode only the newly generated tokens, not the prompt."""
    gen = outputs[0, inputs.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

def execute_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> str:
    from tools import TOOL_REGISTRY
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return f"Error: Tool '{tool_name}' not found"

    try:
        return str(fn(**tool_args))
    except Exception as e:
        return f"Error executing {tool_name}: {e}"



def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract <tool_call>{...}</tool_call> blocks and parse JSON inside.
    Supports multiple tool calls in one model turn.
    """
    tool_calls: List[Dict[str, Any]] = []
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"

    for m in re.finditer(pattern, text, flags=re.DOTALL):
        blob = m.group(1).strip()
        try:
            tool_calls.append(json.loads(blob))
        except json.JSONDecodeError as e:
            # Keep going; malformed tool-call shouldn't crash the loop.
            print(f"[warn] Failed to parse tool call JSON: {e}\n{blob}")

    return tool_calls