"""
Configuration for the trajectorykit agent.

Loads settings from a YAML config file. All module-level names are preserved
for backward compatibility — existing imports like
    from .config import MODEL_NAME, SYSTEM_PROMPT, get_model_profile
continue to work unchanged.

Config resolution order:
    1. load_config(path) — explicit call (e.g., from eval.py)
    2. TRAJECTORYKIT_CONFIG env var
    3. configs/default.yaml (relative to repo root)
    4. Hardcoded fallback defaults
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ── Locate repo root (two levels up from this file) ─────────────────────
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent

# ── Hardcoded fallback (used if no YAML found) ──────────────────────────
_FALLBACK_CONFIG = {
    "model": {
        "name": "Qwen/Qwen3-8B",
        "api_url": "http://localhost:3030/v1",
    },
    "agent": {
        "max_recursion_depth": 1,
        "sub_agent_turn_budget": 15,
        "token_safety_margin": 256,
    },
    "sandbox": {
        "url": "http://localhost:8080/run_code",
    },
    "traces": {
        "dir": "traces/",
    },
    "prompts": {
        "orchestrator": "configs/prompts/orchestrator.txt",
        "worker": "configs/prompts/worker.txt",
    },
    "model_profiles": {
        "Qwen/Qwen3-8B": {
            "context_window": 32768,
            "supports_reasoning_effort": False,
            "default_temperature": 0.7,
        },
        "openai/gpt-oss-20b": {
            "context_window": 131072,
            "supports_reasoning_effort": True,
            "default_reasoning_effort": "high",
            "default_temperature": 1.0,
        },
    },
    "default_profile": {
        "context_window": 32768,
        "supports_reasoning_effort": False,
        "default_temperature": 0.7,
    },
    "dataset": {
        "name": "google/deepsearchqa",
        "split": "eval",
        "sample_n": 50,
        "seed": 42,
    },
    "eval": {
        "turn_length": None,
        "reasoning_effort": "high",
        "verbose": True,
        "output_dir": None,
    },
}


# ── Internal state ──────────────────────────────────────────────────────
_config: dict = {}
_loaded = False


def _resolve_path(relative: str) -> str:
    """Resolve a path relative to the repo root."""
    return str(_REPO_ROOT / relative)


def _load_prompt(filepath: str) -> str:
    """Load a prompt template from a text file, substituting {current_date}."""
    resolved = _resolve_path(filepath)
    try:
        raw = Path(resolved).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompt file not found: {resolved}\n"
            f"Expected at: {filepath} (relative to repo root {_REPO_ROOT})"
        )
    return raw.replace("{current_date}", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))


def load_config(path: Optional[str] = None) -> dict:
    """Load configuration from a YAML file.

    Args:
        path: Path to a YAML config file. If None, tries:
              1. TRAJECTORYKIT_CONFIG env var
              2. configs/default.yaml (relative to repo root)
              3. Hardcoded fallback

    Returns:
        The loaded config dict.

    Side effects:
        Updates all module-level constants (MODEL_NAME, SYSTEM_PROMPT, etc.)
        so that existing imports see the new values.
    """
    global _config, _loaded

    # Resolve config path
    if path is None:
        path = os.environ.get("TRAJECTORYKIT_CONFIG")
    if path is None:
        default_path = _REPO_ROOT / "configs" / "default.yaml"
        if default_path.exists():
            path = str(default_path)

    # Load YAML or use fallback
    if path is not None:
        resolved_path = Path(path)
        if not resolved_path.is_absolute():
            resolved_path = _REPO_ROOT / resolved_path
        with open(resolved_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    else:
        _config = _FALLBACK_CONFIG.copy()

    _loaded = True
    _update_module_constants()
    return _config


def get_config() -> dict:
    """Return the current config, loading defaults if not yet loaded."""
    if not _loaded:
        load_config()
    return _config


def _update_module_constants():
    """Populate module-level constants from the loaded config.

    This is what makes `from .config import MODEL_NAME` work —
    the module globals are updated in-place after every load_config() call.
    """
    global MODEL_NAME, VLLM_API_URL, SANDBOX_FUSION_URL
    global MAX_RECURSION_DEPTH, SUB_AGENT_TURN_BUDGET, TOKEN_SAFETY_MARGIN
    global CONTEXT_WINDOW, TRACES_DIR
    global MODEL_PROFILES, _DEFAULT_PROFILE
    global SYSTEM_PROMPT, WORKER_PROMPT, SYNTHESIZER_PROMPT
    global DATASET_CONFIG, EVAL_CONFIG

    c = _config

    # Model
    MODEL_NAME = c["model"]["name"]
    VLLM_API_URL = c["model"]["api_url"]

    # Agent
    MAX_RECURSION_DEPTH = c["agent"]["max_recursion_depth"]
    SUB_AGENT_TURN_BUDGET = c["agent"]["sub_agent_turn_budget"]
    TOKEN_SAFETY_MARGIN = c["agent"]["token_safety_margin"]

    # Sandbox
    SANDBOX_FUSION_URL = c["sandbox"]["url"]

    # Traces
    traces_dir = c["traces"]["dir"]
    TRACES_DIR = _resolve_path(traces_dir)

    # Model profiles
    MODEL_PROFILES = c.get("model_profiles", {})
    _DEFAULT_PROFILE = c.get("default_profile", {
        "context_window": 32768,
        "supports_reasoning_effort": False,
        "default_temperature": 0.7,
    })

    # Derived
    CONTEXT_WINDOW = get_model_profile(MODEL_NAME)["context_window"]

    # Prompts
    orchestrator_path = c.get("prompts", {}).get("orchestrator", "configs/prompts/orchestrator.txt")
    worker_path = c.get("prompts", {}).get("worker", "configs/prompts/worker.txt")
    synthesizer_path = c.get("prompts", {}).get("synthesizer", "configs/prompts/synthesizer.txt")
    SYSTEM_PROMPT = _load_prompt(orchestrator_path)
    WORKER_PROMPT = _load_prompt(worker_path)
    SYNTHESIZER_PROMPT = _load_prompt(synthesizer_path)

    # Dataset & eval (new — used by eval.py)
    DATASET_CONFIG = c.get("dataset", {})
    EVAL_CONFIG = c.get("eval", {})


def get_model_profile(model: str) -> dict:
    """Return the profile for a model, falling back to defaults for unknown models."""
    if not _loaded:
        load_config()
    return MODEL_PROFILES.get(model, _DEFAULT_PROFILE)


# ── Auto-load on import ─────────────────────────────────────────────────
# This ensures backward compatibility: importing config.py immediately
# populates all module-level constants.
load_config()