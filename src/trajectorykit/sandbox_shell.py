"""
agent_shell.py
==============
Programmatic shell interface for an Apptainer sandbox.

Usage (as a library):
    from agent_shell import SandboxShell

    with SandboxShell() as shell:
        result = shell.run("echo hello")
        print(result.stdout)

Usage (interactive demo):
    python3 agent_shell.py
"""

from __future__ import annotations

import atexit
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Global registry of all live SandboxShell instances for cleanup
_LIVE_SHELLS: list[SandboxShell] = []
_LIVE_SHELLS_LOCK = threading.Lock()


def _cleanup_all_shells():
    """atexit handler: kill all live sandbox shells."""
    with _LIVE_SHELLS_LOCK:
        for sh in _LIVE_SHELLS:
            try:
                sh._force_kill()
            except Exception:
                pass
        _LIVE_SHELLS.clear()

atexit.register(_cleanup_all_shells)


# ── Config ────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent.resolve()

SANDBOX_IMAGE   = os.getenv("SANDBOX_IMAGE",   str(_HERE / "agent_sandbox.sif"))
WORKSPACE_DIR   = os.getenv("WORKSPACE_DIR",   str(_HERE / "workspace"))
LOGS_DIR        = os.getenv("LOGS_DIR",        str(_HERE / "logs"))
DEFAULT_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))   # seconds per command


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class CommandResult:
    command:    str
    stdout:     str
    stderr:     str
    exit_code:  int
    duration:   float          # seconds
    timed_out:  bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def __str__(self) -> str:
        parts = [f"$ {self.command}"]
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append(f"[stderr] {self.stderr.rstrip()}")
        status = "OK" if self.success else f"exit={self.exit_code}"
        if self.timed_out:
            status = "TIMED OUT"
        parts.append(f"[{status} | {self.duration:.2f}s]")
        return "\n".join(parts)


# ── Core shell class ──────────────────────────────────────────────────────────

class SandboxShell:
    """
    A stateful shell session running inside an Apptainer sandbox.

    - The container filesystem is read-only (the .sif image is never modified).
    - /workspace is bind-mounted from the host, providing persistent shared storage.
    - Each SandboxShell instance gets an isolated /tmp inside the container.
    - The session is reset simply by creating a new SandboxShell instance.
    """

    def __init__(
        self,
        image:       str            = SANDBOX_IMAGE,
        workspace:   str            = WORKSPACE_DIR,
        timeout:     int            = DEFAULT_TIMEOUT,
        network:     bool           = False,     # disabled by default — enable if needed
        extra_binds: list[str]      = None,      # e.g. ["/data:/data:ro"]
        env:         dict[str, str] = None,      # extra env vars inside container
        log_file:    Optional[str]  = None,
    ):
        self.image       = Path(image)
        self.workspace   = Path(workspace)
        self.timeout     = timeout
        self.network     = network
        self.extra_binds = extra_binds or []
        self.extra_env   = env or {}
        self.session_id  = uuid.uuid4().hex[:8]
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        # Per-session workspace subdirectory
        self.session_workspace = self.workspace / f"session_{self.session_id}"
        self.session_workspace.mkdir(parents=True, exist_ok=True)

        # Logging
        Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
        log_path = log_file or str(Path(LOGS_DIR) / f"session_{self.session_id}.log")
        self._log = open(log_path, "a", buffering=1)

        self._start_shell()
        
        # Register for global cleanup
        with _LIVE_SHELLS_LOCK:
            _LIVE_SHELLS.append(self)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _build_apptainer_cmd(self) -> list[str]:
        cmd = ["apptainer", "exec"]

        # Isolation flags
        cmd += [
            "--contain",          # Don't auto-mount host dirs
            "--no-home",          # Don't mount host $HOME into container
            "--no-privs",         # Drop any extra privileges
            "--cleanenv",         # Start with a clean environment
        ]

        # Network
        if not self.network:
            cmd += ["--net", "--network", "none"]

        # Bind mounts: only the session workspace
        cmd += ["--bind", f"{self.session_workspace}:/workspace"]
        for b in self.extra_binds:
            cmd += ["--bind", b]

        # Core environment — redirect HOME and pip cache away from the host
        # home directory, which doesn't exist inside the container
        core_env = {
            "HOME": "/workspace",
            "PIP_CACHE_DIR": "/tmp/pip-cache",
            "PIP_NO_WARN_SCRIPT_LOCATION": "1",
            "TERM": "dumb",
        }
        core_env.update(self.extra_env)  # caller env takes precedence
        for k, v in core_env.items():
            cmd += ["--env", f"{k}={v}"]

        # Use non-interactive bash (-s reads from stdin, no prompt, no job control
        # messages). Do NOT use -i here — interactive mode causes the `Apptainer>`
        # prompt to echo our sentinel lines back as output, breaking exit-code
        # detection and making every command appear to run twice.
        cmd += [str(self.image), "/bin/bash", "--norc", "--noprofile", "-s"]
        return cmd

    def _start_shell(self) -> None:
        """Spawn the persistent bash process inside the container."""
        if not self.image.exists():
            raise FileNotFoundError(
                f"Sandbox image not found: {self.image}\n"
                f"Run `bash setup_apptainer.sh` first."
            )

        cmd = self._build_apptainer_cmd()
        self._log_event("SESSION_START", f"cmd={shlex.join(cmd)}")

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,           # Unbuffered
            preexec_fn=os.setsid,  # New process group for clean kill
        )

        # Brief wait to confirm the shell started
        time.sleep(0.3)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"Container process exited immediately (code {self._proc.returncode}).\n"
                f"Check that Apptainer is installed and the image is valid."
            )

    @property
    def is_alive(self) -> bool:
        """Return True if the shell process is still running."""
        return self._proc is not None and self._proc.poll() is None

    def ensure_alive(self) -> None:
        """Restart the shell process if it has died.

        The session workspace is bind-mounted from the host, so all files
        survive a restart — only in-memory shell state (env vars, cwd) is lost.
        After restart the working directory is reset to /workspace.
        """
        if self.is_alive:
            return
        self._log_event("RESTART", "Shell process died — restarting")
        # Clean up the old process (might be a zombie)
        if self._proc is not None:
            try:
                self._proc.wait(timeout=1)
            except Exception:
                pass
        self._start_shell()
        # Restore working directory to /workspace after restart
        try:
            self._proc.stdin.write("cd /workspace\n")
            self._proc.stdin.flush()
            time.sleep(0.1)
        except Exception:
            pass
        self._log_event("RESTART", "Shell restarted successfully")

    def close(self) -> None:
        """Terminate the shell session and clean up."""
        # Unregister from global cleanup
        with _LIVE_SHELLS_LOCK:
            try:
                _LIVE_SHELLS.remove(self)
            except ValueError:
                pass
        self._force_kill()
        self._log_event("SESSION_END", f"session={self.session_id}")
        try:
            self._log.close()
        except Exception:
            pass
    
    def _force_kill(self) -> None:
        """Kill the shell process and all its children (SIGTERM then SIGKILL)."""
        if self._proc and self._proc.poll() is None:
            try:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    pgid = os.getpgid(self._proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    self._proc.wait(timeout=2)
                except Exception:
                    # Last resort: kill just the process
                    try:
                        self._proc.kill()
                        self._proc.wait(timeout=2)
                    except Exception:
                        pass
    
    def __del__(self) -> None:
        """Safety net: kill shell if object is garbage-collected without close()."""
        try:
            if self._proc and self._proc.poll() is None:
                self._force_kill()
        except Exception:
            pass
    
    @classmethod
    def kill_orphans(cls) -> int:
        """Find and kill any orphaned apptainer exec processes.
        
        Useful after a crash or Ctrl-C that didn't trigger atexit.
        Returns the number of processes killed.
        """
        import subprocess as _sp
        killed = 0
        try:
            result = _sp.run(
                ["pgrep", "-f", "apptainer exec.*agent_sandbox"],
                capture_output=True, text=True, timeout=5
            )
            for pid_str in result.stdout.strip().split("\n"):
                pid_str = pid_str.strip()
                if pid_str and pid_str.isdigit():
                    pid = int(pid_str)
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        pass
        except Exception:
            pass
        return killed

    def __enter__(self) -> SandboxShell:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Command execution ─────────────────────────────────────────────────────

    def run(
        self,
        command:  str,
        timeout:  Optional[int] = None,
    ) -> CommandResult:
        """
        Execute a shell command inside the sandbox and return a CommandResult.

        The command runs in the persistent shell session, so state (cwd, env vars,
        shell functions) carries over between calls — just like a real terminal.

        Args:
            command: Shell command string (passed to bash -c)
            timeout: Override the default per-command timeout (seconds)

        Returns:
            CommandResult with stdout, stderr, exit_code, duration, timed_out
        """
        timeout = timeout if timeout is not None else self.timeout

        with self._lock:
            self.ensure_alive()
            return self._execute(command, timeout)

    def _execute(self, command: str, timeout: int) -> CommandResult:
        # Unique sentinel printed to stdout once the command finishes.
        # We do NOT use a stderr sentinel — stderr is read in a separate
        # thread with no synchronisation requirement, avoiding the race
        # that caused exit=2 when the shell was in interactive mode.
        sentinel = f"__SANDBOX_DONE_{uuid.uuid4().hex}__"

        # Wrap the command so we can capture the exit code cleanly:
        #   1. Run the user command
        #   2. Save $? immediately (before anything else can clobber it)
        #   3. Print sentinel + exit code to stdout
        #   4. Also tee stderr so callers see it, but don't rely on it for sync
        wrapped = (
            f"{command}\n"
            f"__ec__=$?\n"
            f"echo '{sentinel}' $__ec__\n"
        )

        self._log_event("CMD", command)
        t_start = time.monotonic()
        timed_out = False

        try:
            self._proc.stdin.write(wrapped)
            self._proc.stdin.flush()
        except BrokenPipeError:
            # Shell died between ensure_alive() and write — restart and retry once
            self._log_event("RESTART", "BrokenPipeError during write — restarting")
            try:
                self.ensure_alive()
                sentinel = f"__SANDBOX_DONE_{uuid.uuid4().hex}__"
                wrapped = (
                    f"{command}\n"
                    f"__ec__=$?\n"
                    f"echo '{sentinel}' $__ec__\n"
                )
                self._proc.stdin.write(wrapped)
                self._proc.stdin.flush()
            except Exception:
                return CommandResult(command, "", "Shell process died and could not be restarted", -1, 0.0)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        exit_code = -1
        stdout_done = threading.Event()

        # ── stdout reader — stops at sentinel, extracts exit code ─────────────
        def read_stdout():
            nonlocal exit_code
            for line in self._proc.stdout:
                if sentinel in line:
                    m = re.search(r"(\d+)\s*$", line)
                    if m:
                        exit_code = int(m.group(1))
                    stdout_done.set()
                    return
                stdout_lines.append(line)
            stdout_done.set()

        # ── stderr reader — purely collects, never blocks on sentinel ─────────
        def read_stderr():
            # Read whatever stderr lines are available up until stdout signals
            # done (plus a small grace period). We can't iterate blocking here
            # because stderr has no sentinel, so we poll with a short sleep.
            deadline = time.monotonic() + timeout + 2
            while time.monotonic() < deadline:
                if stdout_done.is_set() and not self._proc.stderr.readable():
                    break
                # Non-blocking read via select
                import select
                ready, _, _ = select.select([self._proc.stderr], [], [], 0.05)
                if ready:
                    line = self._proc.stderr.readline()
                    if line:
                        stderr_lines.append(line)
                elif stdout_done.is_set():
                    # stdout finished and no more stderr data — we're done
                    break

        t_out = threading.Thread(target=read_stdout, daemon=True)
        t_err = threading.Thread(target=read_stderr, daemon=True)
        t_out.start()
        t_err.start()

        stdout_done.wait(timeout=timeout)
        t_err.join(timeout=1)  # give stderr reader a moment to drain

        if not stdout_done.is_set():
            timed_out = True
            try:
                # Send Ctrl-C to interrupt the running command
                self._proc.stdin.write("\x03\n")
                self._proc.stdin.flush()
            except Exception:
                pass

        duration = time.monotonic() - t_start
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        result = CommandResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration=duration,
            timed_out=timed_out,
        )
        self._log_event("RESULT", f"exit={exit_code} duration={duration:.2f}s timed_out={timed_out}")
        if stdout.strip():
            self._log_event("STDOUT", stdout.rstrip())
        if stderr.strip():
            self._log_event("STDERR", stderr.rstrip())
        return result

    def run_script(self, script: str, timeout: int = 60) -> CommandResult:
        """Write a multi-line script to a temp file and execute it."""
        script_path = f"/tmp/agent_script_{uuid.uuid4().hex[:6]}.sh"
        write_result = self.run(
            f"cat > {script_path} << 'HEREDOC_EOF'\n{script}\nHEREDOC_EOF",
            timeout=5,
        )
        if not write_result.success:
            return write_result
        return self.run(f"bash {script_path}", timeout=timeout)

    def upload(self, local_path: str, container_path: str = "/workspace/") -> None:
        """Copy a file from the host into the session workspace."""
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(local_path)
        import shutil
        dest_rel = container_path.lstrip("/workspace/")
        dest = self.session_workspace / (dest_rel or src.name)
        shutil.copy2(src, dest)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, detail: str) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._log.write(f"[{ts}] [{self.session_id}] [{event_type}] {detail}\n")


# ── Interactive demo ──────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("  Apptainer Agent Sandbox — Interactive Demo")
    print("=" * 60)
    print()

    with SandboxShell(timeout=15, network=True) as shell:
        demo_commands = [
            ("System info",   "uname -a && cat /etc/os-release | head -5"),
            ("Python",        "python3 -c \"import sys; print(sys.version)\""),
            ("pip packages",  "pip3 list --format=columns | head -15"),
            ("Node.js",       "node -e \"console.log('Node ' + process.version)\""),
            ("Git",           "git --version"),
            ("Workspace",     "ls -la /workspace/"),
            ("Isolation",     "curl https://example.com 2>&1 | head -3 || echo 'Network blocked ✓'"),
            ("State persists","export MY_VAR=hello && echo $MY_VAR"),
            ("Across calls",  "echo $MY_VAR"),  # Should still be set
        ]

        for label, cmd in demo_commands:
            print(f"\n{'─'*50}")
            print(f"  Test: {label}")
            print(f"{'─'*50}")
            result = shell.run(cmd)
            print(result)

    print("\n" + "=" * 60)
    print("  Demo complete. Session reset — all container state gone.")
    print("=" * 60)


if __name__ == "__main__":
    demo()