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
        "symbolic_references": True,
        "symbolic_threshold": 500,
        "plan_state": True,
        "plan_inject_interval": 3,
        "draft_report": True,
        "history_compaction_enabled": True,
        "history_compaction_msg_threshold": 30,
        "history_compaction_min_interval": 4,
        "history_compaction_recent_turns": 3,
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


def _load_prompt(filepath: str, **extra_vars) -> str:
    """Load a prompt template from a text file, substituting template variables.

    Built-in variables: {current_date}
    Additional variables can be passed via **extra_vars and will be
    substituted as {key} → str(value) in the prompt text.
    """
    resolved = _resolve_path(filepath)
    try:
        raw = Path(resolved).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompt file not found: {resolved}\n"
            f"Expected at: {filepath} (relative to repo root {_REPO_ROOT})"
        )
    text = raw.replace("{current_date}", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
    for key, value in extra_vars.items():
        text = text.replace("{" + key + "}", str(value))
    return text


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
    global SYMBOLIC_REFERENCES, SYMBOLIC_THRESHOLD, PLAN_STATE, PLAN_INJECT_INTERVAL, DRAFT_REPORT, DRAFT_FORMAT
    global VERIFY_BEFORE_PUBLISH, VERIFIER_PROMPT, VERIFIER_EXTERNAL_PROMPT, VERIFIER_MODEL, VERIFIER_API_URL, VERIFIER_API_KEY, VERIFIER_TEMPERATURE
    global VERIFIER_STAGE1_PROVIDER, VERIFIER_STAGE3_PROVIDER
    global SPOT_CHECK_ENABLED, SPOT_CHECK_CLAIMS, SPOTCHECK_EXTRACT_PROMPT, SPOTCHECK_COMPARE_PROMPT, SPOTCHECK_REFUSAL_PROMPT
    global MAX_VERIFICATION_REJECTIONS, MAX_SPOT_CHECK_REJECTIONS
    global CITATION_AUDIT_ENABLED, CITATION_AUDIT_PROMPT
    global CHAIN_ANALYSIS_ENABLED, CHAIN_ANALYSIS_PROMPT
    global HISTORY_COMPACTION_ENABLED, HISTORY_COMPACTION_MSG_THRESHOLD
    global HISTORY_COMPACTION_MIN_INTERVAL, HISTORY_COMPACTION_RECENT_TURNS

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

    # Prompts — inject budget so templates like {sub_agent_turn_budget} resolve
    _budget = SUB_AGENT_TURN_BUDGET
    _prompt_vars = {
        "sub_agent_turn_budget": _budget,
        "worker_checkpoint_turn": _budget // 2,
        "worker_answer_turn": max(_budget - 2, _budget // 2 + 1),
    }
    orchestrator_path = c.get("prompts", {}).get("orchestrator", "configs/prompts/orchestrator.txt")
    worker_path = c.get("prompts", {}).get("worker", "configs/prompts/worker.txt")
    synthesizer_path = c.get("prompts", {}).get("synthesizer", "configs/prompts/synthesizer.txt")
    SYSTEM_PROMPT = _load_prompt(orchestrator_path, **_prompt_vars)
    WORKER_PROMPT = _load_prompt(worker_path, **_prompt_vars)
    SYNTHESIZER_PROMPT = _load_prompt(synthesizer_path, **_prompt_vars)

    # Agent feature flags
    agent_cfg = c.get("agent", {})
    SYMBOLIC_REFERENCES = agent_cfg.get("symbolic_references", True)
    SYMBOLIC_THRESHOLD = agent_cfg.get("symbolic_threshold", 500)
    PLAN_STATE = agent_cfg.get("plan_state", True)
    PLAN_INJECT_INTERVAL = agent_cfg.get("plan_inject_interval", 3)
    DRAFT_REPORT = agent_cfg.get("draft_report", True)
    DRAFT_FORMAT = agent_cfg.get("draft_format", "qa")  # "qa" (Final Answer/Sources/Details) or "report" (Title/Executive Summary/Sections/Sources)
    VERIFY_BEFORE_PUBLISH = agent_cfg.get("verify_before_publish", True)

    # Verifier model — defaults to the main model/endpoint if not set
    verifier_cfg = c.get("verifier", {})
    VERIFIER_MODEL = verifier_cfg.get("model", None)       # None → use state.model
    VERIFIER_API_URL = verifier_cfg.get("api_url", None)   # None → use VLLM_API_URL
    VERIFIER_API_KEY = verifier_cfg.get("api_key", None) or os.getenv("OPENAI_API_KEY", "")
    VERIFIER_TEMPERATURE = verifier_cfg.get("temperature", None)  # None → use state.temperature

    # Per-stage provider: "self" = local model, "external" = verifier.model
    VERIFIER_STAGE1_PROVIDER = verifier_cfg.get("stage1_provider", "self")
    VERIFIER_STAGE3_PROVIDER = verifier_cfg.get("stage3_provider", "self")

    # Verifier prompts
    verifier_path = c.get("prompts", {}).get("verifier", "configs/prompts/verifier.txt")
    try:
        VERIFIER_PROMPT = _load_prompt(verifier_path)
    except FileNotFoundError:
        VERIFIER_PROMPT = ""  # gracefully degrade if file missing

    verifier_external_path = c.get("prompts", {}).get("verifier_external", "configs/prompts/verifier_external.txt")
    try:
        VERIFIER_EXTERNAL_PROMPT = _load_prompt(verifier_external_path)
    except FileNotFoundError:
        VERIFIER_EXTERNAL_PROMPT = ""

    # Spot-check (Stage 2 verification)
    SPOT_CHECK_ENABLED = agent_cfg.get("spot_check_enabled", False)
    SPOT_CHECK_CLAIMS = agent_cfg.get("spot_check_claims", 3)
    MAX_VERIFICATION_REJECTIONS = agent_cfg.get("max_verification_rejections", 5)
    MAX_SPOT_CHECK_REJECTIONS = agent_cfg.get("max_spot_check_rejections", 5)
    spotcheck_extract_path = c.get("prompts", {}).get("spotcheck_extract", "configs/prompts/spotcheck_extract.txt")
    spotcheck_compare_path = c.get("prompts", {}).get("spotcheck_compare", "configs/prompts/spotcheck_compare.txt")
    try:
        SPOTCHECK_EXTRACT_PROMPT = _load_prompt(spotcheck_extract_path)
    except FileNotFoundError:
        SPOTCHECK_EXTRACT_PROMPT = ""
    try:
        SPOTCHECK_COMPARE_PROMPT = _load_prompt(spotcheck_compare_path)
    except FileNotFoundError:
        SPOTCHECK_COMPARE_PROMPT = ""
    spotcheck_refusal_path = c.get("prompts", {}).get("spotcheck_refusal", "configs/prompts/spotcheck_refusal.txt")
    try:
        SPOTCHECK_REFUSAL_PROMPT = _load_prompt(spotcheck_refusal_path)
    except FileNotFoundError:
        SPOTCHECK_REFUSAL_PROMPT = ""

    # Citation audit (Stage 3 verification — citation faithfulness)
    CITATION_AUDIT_ENABLED = agent_cfg.get("citation_audit_enabled", False)
    citation_audit_path = c.get("prompts", {}).get("citation_audit", "configs/prompts/citation_audit.txt")
    try:
        CITATION_AUDIT_PROMPT = _load_prompt(citation_audit_path)
    except FileNotFoundError:
        CITATION_AUDIT_PROMPT = ""

    # History compaction (root context window management)
    HISTORY_COMPACTION_ENABLED = agent_cfg.get("history_compaction_enabled", True)
    HISTORY_COMPACTION_MSG_THRESHOLD = agent_cfg.get("history_compaction_msg_threshold", 30)
    HISTORY_COMPACTION_MIN_INTERVAL = agent_cfg.get("history_compaction_min_interval", 4)
    HISTORY_COMPACTION_RECENT_TURNS = agent_cfg.get("history_compaction_recent_turns", 3)

    # Chain analysis (pre-dispatch decomposition)
    CHAIN_ANALYSIS_ENABLED = agent_cfg.get("chain_analysis_enabled", False)
    chain_analysis_path = c.get("prompts", {}).get("chain_analysis", "configs/prompts/chain_analysis.txt")
    try:
        CHAIN_ANALYSIS_PROMPT = _load_prompt(chain_analysis_path)
    except FileNotFoundError:
        CHAIN_ANALYSIS_PROMPT = ""

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