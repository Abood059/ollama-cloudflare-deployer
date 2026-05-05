#!/usr/bin/env python3
"""
Professional Adaptive Ollama + Cloudflare Tunnel Deployer
- Safely kills old instances without self-termination
- Tries official Ollama download, falls back to stable GitHub binary
- Selects best model based on total memory
- Exposes API via TryCloudflare tunnel
- Clean exit with clear English logging
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

# Fallback stable Ollama binary (amd64)
OLLAMA_FALLBACK_URL = "https://github.com/ollama/ollama/releases/download/v0.1.32/ollama-linux-amd64"

# Model list
MODELS = [
    (2, "qwen2.5-coder:3b-instruct-q6_K"),
    (4, "qwen2.5-coder:7b-instruct-q4_K_M"),
    (8, "qwen2.5-coder:7b-instruct-q6_K"),
    (12, "qwen2.5-coder:14b-instruct-q4_K_M"),
    (24, "qwen2.5-coder:14b-instruct-q6_K"),
    (40, "qwen2.5-coder:32b-instruct-q4_K_M"),
]

# Background process handles
processes = []

# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ======================================================================
# Helper functions
# ======================================================================
def safe_kill(proc):
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


def cleanup():
    log.info("Cleaning up...")
    for proc in processes:
        safe_kill(proc)
    for f in [OLLAMA_LOG, CF_LOG]:
        if f.exists():
            f.unlink()


def signal_handler(signum, frame):
    log.warning(f"Received signal {signum}, exiting...")
    cleanup()
    sys.exit(0)


def register_cleanup():
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def download_file(url, dest, desc="file", min_size_mb=10):
    """Download using curl with browser User-Agent, verify minimum size."""
    log.info(f"Downloading {desc} from {url}...")
    try:
        result = subprocess.run(
            [
                "curl", "-L", "--progress-bar",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "-o", str(dest),
                url
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl exited with code {result.returncode}")
    except FileNotFoundError:
        raise RuntimeError("curl is required. Install it: sudo apt install curl")

    size_mb = dest.stat().st_size / (1024*1024) if dest.exists() else 0
    if size_mb < min_size_mb:
        raise RuntimeError(f"Downloaded {desc} is too small ({size_mb:.1f} MB), probably an error page.")
    log.info(f"{desc} downloaded ({size_mb:.1f} MB).")


def test_binary(path):
    """Check if a file is executable and outputs version info."""
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def safe_pkill(proc_name):
    """Kill processes by name, excluding current PID."""
    current_pid = os.getpid()
    try:
        subprocess.run(
            f"pkill -f '{proc_name}' --exclude-pids {current_pid}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ======================================================================
# Binary acquisition
# ======================================================================
def get_ollama():
    """
    Try official Ollama download (tar.zst), extract.
    If it fails, download the stable binary directly from GitHub.
    """
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        official_url = "https://ollama.com/download/ollama-linux-amd64.tar.zst"
        fallback_url = OLLAMA_FALLBACK_URL  # direct binary
        desc = "ollama (amd64)"
    elif arch in ("aarch64", "arm64"):
        official_url = "https://ollama.com/download/ollama-linux-arm64.tar.zst"
        fallback_url = OLLAMA_FALLBACK_URL  # not ideal for arm, but keep as safety
        desc = "ollama (arm64)"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    # Step 1: try official archive
    archive = Path("ollama_temp.tar.zst")
    try:
        download_file(official_url, archive, desc, min_size_mb=50)
        log.info("Extracting Ollama from archive...")
        # Use zstd if available, else fallback to standard tar
        try:
            subprocess.run(["tar", "-I", "zstd", "-xf", str(archive), "-C", "."],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            subprocess.run(["tar", "-xf", str(archive), "-C", "."],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Locate extracted ollama binary
        extracted = None
        for root, dirs, files in os.walk("."):
            if "ollama" in files:
                extracted = Path(root) / "ollama"
                break
        if extracted is None:
            raise RuntimeError("ollama binary not found after extraction.")
        
        # Move to final name
        if extracted != OLLAMA_BIN:
            extracted.rename(OLLAMA_BIN)
        OLLAMA_BIN.chmod(0o755)
        archive.unlink(missing_ok=True)

        if test_binary(OLLAMA_BIN):
            log.info("Ollama binary from official source is ready.")
            return
        else:
            log.warning("Official binary not functional, switching to fallback.")
            OLLAMA_BIN.unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Official download/extraction failed: {e}")
        archive.unlink(missing_ok=True)

    # Step 2: fallback direct binary
    log.info("Attempting fallback download (stable GitHub release)...")
    download_file(fallback_url, OLLAMA_BIN, desc, min_size_mb=100)  # direct binary > 100MB
    OLLAMA_BIN.chmod(0o755)
    if not test_binary(OLLAMA_BIN):
        OLLAMA_BIN.unlink(missing_ok=True)
        raise RuntimeError("Fallback Ollama binary also not executable.")
    log.info("Ollama binary from fallback is ready.")


def get_cloudflared():
    """Download cloudflared with curl."""
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        desc = "cloudflared (amd64)"
    elif arch in ("aarch64", "arm64"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        desc = "cloudflared (arm64)"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    download_file(url, CF_BIN, desc, min_size_mb=5)
    CF_BIN.chmod(0o755)
    if not test_binary(CF_BIN):
        CF_BIN.unlink(missing_ok=True)
        raise RuntimeError("cloudflared binary is not executable")
    log.info("cloudflared binary ready.")


# ======================================================================
# System detection
# ======================================================================
def total_memory_gb():
    ram = 0
    vram = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) / (1024*1024)
                    break
    except Exception:
        ram = 2.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5
        ).decode()
        vram = sum(int(x) for x in out.strip().splitlines()) / 1024.0
    except Exception:
        pass
    total = ram + vram
    if total < 1.0:
        total = 2.0
    log.info(f"Total memory: {total:.1f} GB (RAM: {ram:.1f}, VRAM: {vram:.1f})")
    return total


def select_model(mem_gb):
    usable = mem_gb * 0.85
    selected = None
    for th, model in MODELS:
        if usable >= th:
            selected = model
        else:
            break
    if selected is None:
        selected = MODELS[0][1]
    log.info(f"Selected model: {selected} (usable memory {usable:.1f} GB)")
    return selected


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port_process(port):
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
        )
        time.sleep(1)
    except Exception:
        pass


# ======================================================================
# Service management
# ======================================================================
def wait_for_ollama(timeout=60):
    log.info("Waiting for Ollama server...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=2)
            log.info("Ollama ready.")
            return True
        except Exception:
            time.sleep(2)
    return False


def pull_model(model_name):
    log.info(f"Pulling model: {model_name}")
    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024**3)
    if free_gb < 5:
        raise RuntimeError(f"Insufficient disk space ({free_gb:.1f} GB). Need >= 5 GB.")
    subprocess.run(
        [str(OLLAMA_BIN), "pull", model_name],
        stdout=sys.stdout, stderr=subprocess.STDOUT, check=True
    )


def start_ollama_server():
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{OLLAMA_PORT}"
    env["OLLAMA_ORIGINS"] = "*"
    log_file = open(OLLAMA_LOG, "w")
    proc = subprocess.Popen(
        [str(OLLAMA_BIN), "serve"],
        stdout=log_file, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setpgrp
    )
    processes.append(proc)


def start_cloudflared():
    log_file = open(CF_LOG, "w")
    proc = subprocess.Popen(
        [str(CF_BIN), "tunnel", "--url", f"http://127.0.0.1:{OLLAMA_PORT}"],
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp
    )
    processes.append(proc)


def extract_tunnel_url(timeout=40):
    log.info("Waiting for tunnel URL...")
    start = time.time()
    while time.time() - start < timeout:
        if CF_LOG.exists():
            text = CF_LOG.read_text()
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
            if match:
                return match.group()
        for p in processes:
            if p and p.poll() is not None:
                raise RuntimeError("cloudflared died unexpectedly.")
        time.sleep(2)
    raise RuntimeError("Timeout waiting for tunnel URL.")


def print_warning(url):
    border = "!" * 70
    msg = f"""
{border}
SECURITY WARNING: Your Ollama API is PUBLICLY accessible at:
{url}
Anyone with this link can use your models without authentication.
Keep it private and stop the script when done.
{border}
"""
    print(msg, file=sys.stderr)


# ======================================================================
# Main
# ======================================================================
def main():
    register_cleanup()
    try:
        # Clean old instances
        log.info("Preparing environment (stopping old instances)...")
        safe_pkill("ollama")
        safe_pkill("cloudflared")
        if port_in_use(OLLAMA_PORT):
            log.info("Freeing port 11434...")
            kill_port_process(OLLAMA_PORT)
            if port_in_use(OLLAMA_PORT):
                raise RuntimeError("Port 11434 still in use.")
        for f in [OLLAMA_LOG, CF_LOG]:
            f.unlink(missing_ok=True)

        # Get Ollama binary (with fallback)
        if not (OLLAMA_BIN.exists() and test_binary(OLLAMA_BIN)):
            if OLLAMA_BIN.exists():
                OLLAMA_BIN.unlink()
            get_ollama()

        # Start Ollama server
        log.info("Starting Ollama server...")
        start_ollama_server()
        if not wait_for_ollama():
            raise RuntimeError("Ollama server did not start. Check ollama.log")

        # Model selection & pull
        mem = total_memory_gb()
        model = select_model(mem)
        pull_model(model)

        # Cloudflared
        if not (CF_BIN.exists() and test_binary(CF_BIN)):
            if CF_BIN.exists():
                CF_BIN.unlink()
            get_cloudflared()
        log.info("Starting Cloudflare tunnel...")
        start_cloudflared()
        url = extract_tunnel_url()

        # Output
        print(url, flush=True)
        print_warning(url)

        log.info("Services running. Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
