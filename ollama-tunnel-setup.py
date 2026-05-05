#!/usr/bin/env python3
"""
Professional Adaptive Ollama + Cloudflare Tunnel (Fixed Self‑Kill)
- Auto-detects architecture, downloads correct binaries only
- Selects best model based on total memory
- Launches Ollama, pulls model, exposes via Cloudflare Tunnel
- Handles cleanup gracefully on exit (SIGINT, SIGTERM, normal)
- Robust error handling, no more accidental self‑termination
"""

import os
import sys
import time
import signal
import atexit
import shutil
import logging
import subprocess
import urllib.request
import platform
import re
import socket
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
OLLAMA_BIN = Path("./ollama_bin")
OLLAMA_PORT = 11434
CF_BIN = Path("./cf_bin")
CF_LOG = Path("cf.log")
OLLAMA_LOG = Path("ollama.log")

# Models (threshold_GB, model_name)
MODELS = [
    (2, "qwen2.5-coder:3b-instruct-q6_K"),
    (4, "qwen2.5-coder:7b-instruct-q4_K_M"),
    (8, "qwen2.5-coder:7b-instruct-q6_K"),
    (12, "qwen2.5-coder:14b-instruct-q4_K_M"),
    (24, "qwen2.5-coder:14b-instruct-q6_K"),
    (40, "qwen2.5-coder:32b-instruct-q4_K_M"),
]

# Global process handles for cleanup
processes = []

# ----------------------------------------------------------------------
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
def safe_kill(proc):
    """Terminate a subprocess gracefully, then force kill if needed."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            proc.kill()
            proc.wait(timeout=2)
        except ProcessLookupError:
            pass
    except Exception:
        pass


def cleanup():
    """Terminate all background processes and remove log files."""
    log.info("Cleaning up...")
    for proc in processes:
        safe_kill(proc)
    for f in [OLLAMA_LOG, CF_LOG]:
        if f.exists():
            f.unlink()


def signal_handler(signum, frame):
    """Handle SIGINT / SIGTERM gracefully."""
    log.warning(f"Received signal {signum}, exiting...")
    cleanup()
    sys.exit(0)


def register_cleanup():
    """Register cleanup on normal exit and signals."""
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# ----------------------------------------------------------------------
def download(url, dest, desc="file"):
    """Download a file with descriptive error message."""
    log.info(f"Downloading {desc} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(f"Failed to download {desc}: {e}")


def test_binary(path):
    """Check if binary is executable and responds to --version."""
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_ollama():
    """Download correct Ollama binary for the current architecture."""
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        url = "https://ollama.com/download/ollama-linux-amd64"
        desc = "ollama (amd64)"
    elif arch in ("aarch64", "arm64"):
        url = "https://ollama.com/download/ollama-linux-arm64"
        desc = "ollama (arm64)"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    download(url, OLLAMA_BIN, desc)
    OLLAMA_BIN.chmod(0o755)
    if not test_binary(OLLAMA_BIN):
        OLLAMA_BIN.unlink(missing_ok=True)
        raise RuntimeError("Ollama binary is not executable (check glibc version?)")
    log.info("Ollama binary ready.")


def get_cloudflared():
    """Download correct cloudflared binary for the current architecture."""
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        desc = "cloudflared (amd64)"
    elif arch in ("aarch64", "arm64"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        desc = "cloudflared (arm64)"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    download(url, CF_BIN, desc)
    CF_BIN.chmod(0o755)
    if not test_binary(CF_BIN):
        CF_BIN.unlink(missing_ok=True)
        raise RuntimeError("cloudflared binary is not executable")
    log.info("cloudflared binary ready.")


# ----------------------------------------------------------------------
def total_memory_gb():
    """Detect total system RAM + GPU VRAM in GB."""
    ram_gb = 0
    vram_gb = 0

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_kb = int(line.split()[1])
                    ram_gb = ram_kb / (1024 * 1024)
                    break
    except Exception:
        log.warning("Cannot read /proc/meminfo, assuming 2 GB RAM.")
        ram_gb = 2.0

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode()
        vram_mb = sum(int(x) for x in out.strip().splitlines())
        vram_gb = vram_mb / 1024.0
    except Exception:
        log.info("No NVIDIA GPU detected or nvidia-smi not found.")

    total = ram_gb + vram_gb
    if total < 1.0:
        total = 2.0  # absolute minimum fallback
    log.info(f"Detected total memory: {total:.1f} GB (RAM {ram_gb:.1f} + VRAM {vram_gb:.1f})")
    return total


def select_model(mem_gb):
    """Select largest model that fits safely (85% of total memory)."""
    usable = mem_gb * 0.85
    selected = None
    for threshold, model in MODELS:
        if usable >= threshold:
            selected = model
        else:
            break
    if selected is None:
        selected = MODELS[0][1]
        log.warning("Very low memory, using smallest model.")
    log.info(f"Selected model: {selected} (usable {usable:.1f} GB)")
    return selected


# ----------------------------------------------------------------------
def port_in_use(port):
    """Check if a TCP port is listening on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port_process(port):
    """Kill any process currently listening on the given port."""
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        time.sleep(1)
    except Exception:
        pass


def wait_for_ollama(timeout=60):
    """Wait until Ollama API is reachable."""
    log.info("Waiting for Ollama server to start...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    log.info("Ollama server is ready.")
                    return True
        except Exception:
            time.sleep(2)
    return False


def pull_model(model_name):
    """Pull the Ollama model, with disk space check."""
    log.info(f"Pulling model {model_name}...")
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024**3)
        if free_gb < 5:
            raise RuntimeError(f"Low disk space ({free_gb:.1f} GB). Need at least 5 GB free.")
    except Exception as e:
        raise RuntimeError(f"Disk space check failed: {e}")

    result = subprocess.run(
        [str(OLLAMA_BIN), "pull", model_name],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull model {model_name}")


def start_ollama_server():
    """Launch Ollama server process."""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{OLLAMA_PORT}"
    env["OLLAMA_ORIGINS"] = "*"

    log_file = open(OLLAMA_LOG, "w")
    proc = subprocess.Popen(
        [str(OLLAMA_BIN), "serve"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setpgrp,
    )
    processes.append(proc)
    return proc


def start_cloudflared():
    """Start cloudflared tunnel process."""
    log_file = open(CF_LOG, "w")
    proc = subprocess.Popen(
        [str(CF_BIN), "tunnel", "--url", f"http://127.0.0.1:{OLLAMA_PORT}"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp,
    )
    processes.append(proc)
    return proc


def extract_tunnel_url(timeout=40):
    """Extract the trycloudflare.com URL from the tunnel log."""
    log.info("Waiting for Cloudflare tunnel URL...")
    start = time.time()
    while time.time() - start < timeout:
        if not CF_LOG.exists():
            time.sleep(1)
            continue
        try:
            text = CF_LOG.read_text()
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
            if match:
                url = match.group()
                log.info("Tunnel URL obtained.")
                return url
        except Exception:
            pass
        # If cloudflared process died early, report error
        for p in processes:
            if p is not None and p.poll() is not None:
                raise RuntimeError("Cloudflared process exited unexpectedly.")
        time.sleep(2)
    raise RuntimeError("Timeout waiting for tunnel URL.")


# ----------------------------------------------------------------------
def print_security_warning(url):
    """Print a highly visible security warning."""
    red = "\033[91m" if sys.stdout.isatty() else ""
    bold = "\033[1m" if sys.stdout.isatty() else ""
    reset = "\033[0m" if sys.stdout.isatty() else ""
    border = "!" * 70
    msg = (
        f"\n{red}{bold}{border}\n"
        f"⚠️  IMPORTANT SECURITY WARNING ⚠️\n"
        f"Your Ollama API is now PUBLICLY accessible at:\n{url}\n"
        f"Anyone with this link can use your models without authentication.\n"
        f"Keep this URL private and stop the script when done.\n"
        f"{border}{reset}\n"
    )
    print(msg, file=sys.stderr)


# ----------------------------------------------------------------------
def main():
    register_cleanup()
    try:
        # 1. Prepare environment (NO self-kill!)
        log.info("Preparing environment...")
        if port_in_use(OLLAMA_PORT):
            log.info(f"Port {OLLAMA_PORT} is occupied, freeing it...")
            kill_port_process(OLLAMA_PORT)
            if port_in_use(OLLAMA_PORT):
                raise RuntimeError(f"Port {OLLAMA_PORT} still in use. Please free it manually.")

        # Remove old logs
        for f in [OLLAMA_LOG, CF_LOG]:
            if f.exists():
                f.unlink()

        # 2. Ensure Ollama binary exists (download if missing/broken)
        if not (OLLAMA_BIN.exists() and test_binary(OLLAMA_BIN)):
            if OLLAMA_BIN.exists():
                OLLAMA_BIN.unlink()
            get_ollama()

        # 3. Start Ollama server
        log.info("Starting Ollama server...")
        start_ollama_server()
        if not wait_for_ollama():
            raise RuntimeError("Ollama server failed to start (check ollama.log)")

        # 4. Detect memory and select model
        mem_gb = total_memory_gb()
        model = select_model(mem_gb)

        # 5. Pull model (if needed)
        pull_model(model)

        # 6. Ensure cloudflared binary
        if not (CF_BIN.exists() and test_binary(CF_BIN)):
            if CF_BIN.exists():
                CF_BIN.unlink()
            get_cloudflared()

        # 7. Start Cloudflare tunnel
        log.info("Starting Cloudflare tunnel...")
        start_cloudflared()
        url = extract_tunnel_url()

        # 8. Output and warn
        print(url, flush=True)
        print_security_warning(url)

        # 9. Keep alive until interrupted
        log.info("Services running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
