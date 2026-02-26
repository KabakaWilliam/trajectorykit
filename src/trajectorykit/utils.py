import json
import logging
import os
import threading
import time
import traceback
import uuid
import re
import requests
import random
import threading

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List


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

# --- simple TTL cache ---
@dataclass
class CacheEntry:
    value: str
    expires_at: float

class TTLCache:
    def __init__(self, ttl_seconds: int = 1800, max_items: int = 512):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._data: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            ent = self._data.get(key)
            if not ent:
                return None
            if ent.expires_at < now:
                self._data.pop(key, None)
                return None
            return ent.value

    def set(self, key: str, value: str):
        now = time.time()
        with self._lock:
            if len(self._data) >= self.max_items:
                # naive eviction: drop a random key
                self._data.pop(next(iter(self._data)), None)
            self._data[key] = CacheEntry(value=value, expires_at=now + self.ttl)

# --- global rate limiter (token bucket-ish) ---
class RateLimiter:
    def __init__(self, min_interval_s: float = 1.0):
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
            self._next_allowed = time.time() + self.min_interval_s

# Shared singletons (process-wide)
_ddg_cache = TTLCache(ttl_seconds=1800, max_items=1024)
_ddg_rate = RateLimiter(min_interval_s=1.2)  # tune: 1.0–2.0s usually helps a lot
_ddg_sem = threading.Semaphore(3)            # cap concurrency
_inflight: Dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()


def _normalize_query(q: str) -> str:
    return " ".join(q.strip().lower().split())
