#!/usr/bin/env python3
"""
Orchestrate: one-command experiment runner.

Reads an experiment YAML, ensures the Apptainer sandbox and vLLM server
are running with the correct arguments, then launches the eval (+judge).

Usage:
    python orchestrate.py --config configs/experiments/gpt_oss_deepsearchqa.yaml
    python orchestrate.py --config configs/experiments/qwen3_deepsearchqa.yaml

What it does:
    1. Pulls the Apptainer SIF (if not cached) and starts the sandbox container
    2. Starts vLLM with model-specific args (parser, GPU, memory, etc.)
    3. Waits for both services to be healthy
    4. Runs evals/eval.py (which now also runs the LLM judge if configured)
    5. Leaves services running for further use

Config keys used (beyond the existing eval config):

    vllm:
      port: 3030                          # default: 3030
      gpu_devices: [2]                     # CUDA_VISIBLE_DEVICES
      gpu_memory_utilization: 0.9          # default: 0.9
      tool_call_parser: "openai"           # required
      reasoning_parser: null               # optional (e.g. "deepseek_r1")
      extra_args: ["--async-scheduling"]   # any other vllm flags
      env:                                 # extra env vars for vLLM process
        TIKTOKEN_CACHE_DIR: "/VData/resources/huggingface/tiktoken-cache"

    sandbox:
      url: "http://localhost:8080/run_code"
      port: 8080                           # default: 8080
      sif_image: "sandbox-fusion_server-20250609.sif"
      docker_uri: "docker://volcengine/sandbox-fusion:server-20250609"
      sif_dir: "."                         # where to store/find the .sif
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_VLLM_PORT = 3030
DEFAULT_SANDBOX_PORT = 8080
DEFAULT_GPU_MEM_UTIL = 0.9
DEFAULT_SIF_IMAGE = "sandbox-fusion_server-20250609.sif"
DEFAULT_DOCKER_URI = "docker://volcengine/sandbox-fusion:server-20250609"
HEALTH_CHECK_TIMEOUT = 300   # seconds to wait for each service
HEALTH_CHECK_INTERVAL = 3    # seconds between checks


def load_experiment_config(path: str) -> dict:
    """Load the experiment YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Service health checks ────────────────────────────────────────────────────

def _wait_for_http(url: str, name: str, timeout: int = HEALTH_CHECK_TIMEOUT) -> bool:
    """Poll a URL until it returns 200 or timeout is reached."""
    print(f"⏳ Waiting for {name} at {url} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code == 200:
                print(f"✅ {name} is ready")
                return True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.TimeoutException):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    print(f"❌ {name} did not become ready within {timeout}s")
    return False


def _is_port_responding(port: int) -> bool:
    """Quick check if anything responds on localhost:port."""
    try:
        r = httpx.get(f"http://localhost:{port}/", timeout=3)
        return True
    except Exception:
        return False


# ── Apptainer sandbox ────────────────────────────────────────────────────────

def ensure_sandbox(cfg: dict) -> subprocess.Popen | None:
    """Pull SIF if needed, start the Apptainer sandbox, return the process.
    
    Returns None if the sandbox is already running.
    """
    sandbox_cfg = cfg.get("sandbox", {})
    port = sandbox_cfg.get("port", DEFAULT_SANDBOX_PORT)
    sif_image = sandbox_cfg.get("sif_image", DEFAULT_SIF_IMAGE)
    docker_uri = sandbox_cfg.get("docker_uri", DEFAULT_DOCKER_URI)
    sif_dir = sandbox_cfg.get("sif_dir", ".")

    sif_path = Path(sif_dir) / sif_image

    # Check if already running
    if _is_port_responding(port):
        print(f"✅ Sandbox already running on port {port}")
        return None

    # Pull SIF if not present
    if not sif_path.exists():
        print(f"📦 Pulling Apptainer image: {docker_uri}")
        print(f"   → {sif_path}")
        subprocess.run(
            ["apptainer", "pull", str(sif_path), docker_uri],
            check=True,
        )
        print(f"✅ SIF pulled: {sif_path}")
    else:
        print(f"✅ SIF exists: {sif_path}")

    # Start the container
    cmd = [
        "apptainer", "run",
        "--containall", "--cleanenv", "--writable-tmpfs",
        str(sif_path),
    ]
    print(f"🚀 Starting sandbox: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc


# ── vLLM server ──────────────────────────────────────────────────────────────

def ensure_vllm(cfg: dict) -> subprocess.Popen | None:
    """Start the vLLM server with model-specific arguments from the config.
    
    Returns None if vLLM is already running.
    """
    vllm_cfg = cfg.get("vllm", {})
    model_name = cfg["model"]["name"]
    port = vllm_cfg.get("port", DEFAULT_VLLM_PORT)

    # Check if already running
    if _is_port_responding(port):
        print(f"✅ vLLM already running on port {port}")
        return None

    # Build command
    gpu_devices = vllm_cfg.get("gpu_devices", [0])
    gpu_mem_util = vllm_cfg.get("gpu_memory_utilization", DEFAULT_GPU_MEM_UTIL)
    tool_call_parser = vllm_cfg.get("tool_call_parser", "openai")
    reasoning_parser = vllm_cfg.get("reasoning_parser")
    extra_args = vllm_cfg.get("extra_args", [])

    cmd = [
        "vllm", "serve", model_name,
        "--enable-auto-tool-choice",
        "--tool-call-parser", tool_call_parser,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    cmd.extend(extra_args)

    # Build environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in gpu_devices)

    # Apply any extra env vars from config
    extra_env = vllm_cfg.get("env", {})
    for k, v in extra_env.items():
        env[k] = str(v)
        # Also ensure directories exist for path-like env vars
        if "DIR" in k or "PATH" in k:
            os.makedirs(str(v), exist_ok=True)

    print(f"🚀 Starting vLLM:")
    print(f"   Model:  {model_name}")
    print(f"   GPUs:   {gpu_devices}")
    print(f"   Port:   {port}")
    print(f"   Parser: {tool_call_parser}" + (f" + {reasoning_parser}" if reasoning_parser else ""))
    print(f"   Cmd:    {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc


# ── Run eval ──────────────────────────────────────────────────────────────────

def run_eval(config_path: str) -> int:
    """Launch evals/eval.py with the given config. Returns exit code."""
    cmd = [sys.executable, "evals/eval.py", "--config", config_path]
    print(f"\n{'═' * 60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'═' * 60}\n")
    result = subprocess.run(cmd)
    return result.returncode


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate: set up services and run an experiment end-to-end",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--skip-sandbox", action="store_true",
        help="Skip starting the Apptainer sandbox (assume it's already running)",
    )
    parser.add_argument(
        "--skip-vllm", action="store_true",
        help="Skip starting vLLM (assume it's already running)",
    )
    parser.add_argument(
        "--services-only", action="store_true",
        help="Only start services (sandbox + vLLM), don't run the eval",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)

    vllm_cfg = cfg.get("vllm", {})
    sandbox_cfg = cfg.get("sandbox", {})
    vllm_port = vllm_cfg.get("port", DEFAULT_VLLM_PORT)
    sandbox_port = sandbox_cfg.get("port", DEFAULT_SANDBOX_PORT)

    sandbox_proc = None
    vllm_proc = None

    # ── Start services ────────────────────────────────────────────────────
    print(f"{'═' * 60}")
    print(f"  Orchestrating experiment")
    print(f"  Config: {args.config}")
    print(f"  Model:  {cfg['model']['name']}")
    print(f"{'═' * 60}\n")

    # 1. Sandbox
    if not args.skip_sandbox:
        sandbox_proc = ensure_sandbox(cfg)
        sandbox_url = sandbox_cfg.get("url", f"http://localhost:{sandbox_port}/run_code")
        # Health check — use the base URL, not /run_code
        sandbox_base = sandbox_url.rsplit("/", 1)[0]
        if not _wait_for_http(f"http://localhost:{sandbox_port}/", "Sandbox"):
            print("❌ Aborting — sandbox failed to start")
            if sandbox_proc:
                sandbox_proc.terminate()
            sys.exit(1)
    else:
        print("⏩ Skipping sandbox setup (--skip-sandbox)")

    # 2. vLLM
    if not args.skip_vllm:
        vllm_proc = ensure_vllm(cfg)
        # vLLM exposes /v1/models when ready
        if not _wait_for_http(f"http://localhost:{vllm_port}/v1/models", "vLLM"):
            print("❌ Aborting — vLLM failed to start")
            if vllm_proc:
                vllm_proc.terminate()
            if sandbox_proc:
                sandbox_proc.terminate()
            sys.exit(1)
    else:
        print("⏩ Skipping vLLM setup (--skip-vllm)")

    # ── Run eval ──────────────────────────────────────────────────────────
    if args.services_only:
        print("\n✅ Services are running. Use --services-only was set, skipping eval.")
        print(f"   vLLM:    http://localhost:{vllm_port}/v1")
        print(f"   Sandbox: http://localhost:{sandbox_port}/")
        print(f"\n   To run eval manually:")
        print(f"   python evals/eval.py --config {args.config}")
        return

    exit_code = run_eval(args.config)

    if exit_code == 0:
        print(f"\n✅ Experiment completed successfully")
    else:
        print(f"\n❌ Experiment failed with exit code {exit_code}")

    # Services are left running for further use
    print(f"\n💡 Services left running:")
    print(f"   vLLM:    http://localhost:{vllm_port}/v1")
    print(f"   Sandbox: http://localhost:{sandbox_port}/")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
