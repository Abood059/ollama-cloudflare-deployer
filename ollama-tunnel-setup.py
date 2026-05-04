#!/usr/bin/env python3
"""
Professional Ollama + Cloudflare Tunnel Deployment Script
---------------------------------------------
- Auto-detects RAM and GPU VRAM
- Recommends a suitable Qwen2.5-Coder model
- Installs and runs Ollama server
- Exposes the server via Cloudflare Tunnel
- Provides a single public URL on success
"""

import os
import sys
import time
import json
import shutil
import logging
import subprocess
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict, List

# ======================== CONFIGURATION ========================
OLLAMA_BIN = "./ollama"
OLLAMA_PORT = 11434
OLLAMA_HOST = f"127.0.0.1:{OLLAMA_PORT}"
CLOUDFLARED_BIN = "./cloudflared"
CLOUDFLARED_LOG = "cloudflared.log"

# Base models to recommend based on memory tiers (GB)
MODEL_RECOMMENDATIONS = [
    (2, "qwen2.5-coder:3b-instruct-q6_K"),     # 2GB total
    (4, "qwen2.5-coder:7b-instruct-q4_K_M"),    # 4GB total
    (8, "qwen2.5-coder:7b-instruct-q6_K"),      # 8GB total
    (12, "qwen2.5-coder:14b-instruct-q4_K_M"),  # 12GB total
    (24, "qwen2.5-coder:14b-instruct-q6_K"),    # 24GB total
    (40, "qwen2.5-coder:32b-instruct-q4_K_M"),  # 40GB total
]

# Quiet logging - only errors and critical messages
logging.basicConfig(level=logging.ERROR, format="")
logger = logging.getLogger(__name__)

# ======================== HARDWARE DETECTION ========================
def get_total_ram_gb() -> float:
    """Return total system RAM in GB using psutil. Fallback to /proc/meminfo."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback for minimal environments without psutil
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        return mem_kb / (1024 * 1024)
        except Exception:
            pass
        return 0.0


def get_total_vram_gb() -> float:
    """Return total GPU VRAM in GB if NVIDIA GPU is present, else 0.0."""
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            # Take the first GPU's total memory
            vram_mib = float(result.stdout.strip().split("\n")[0])
            return vram_mib / 1024
    except Exception:
        pass
    return 0.0


def get_available_memory_gb() -> Tuple[float, float, float]:
    """
    Returns (ram_gb, vram_gb, total_gb) where total_gb = ram_gb + vram_gb.
    Used to recommend the largest model that comfortably fits.
    """
    ram = get_total_ram_gb()
    vram = get_total_vram_gb()
    # RAM is the primary constraint; VRAM is bonus.
    total = ram + vram
    return ram, vram, total

# ======================== MODEL RECOMMENDATION ========================
def recommend_model(total_memory_gb: float) -> str:
    """
    Recommend a Qwen2.5-Coder model based on total available memory.
    Returns the model name string.
    """
    recommended = MODEL_RECOMMENDATIONS[0][1]  # smallest as fallback
    for threshold, model in MODEL_RECOMMENDATIONS:
        if total_memory_gb >= threshold:
            recommended = model
        else:
            break
    return recommended

# ======================== SYSTEM COMMAND HELPERS ========================
def run_cmd(cmd: List[str], check: bool = False, timeout: int = 300, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command, optionally check return code, with timeout."""
    try:
        if capture:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)
        else:
            return subprocess.run(cmd, timeout=timeout, check=check)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    except subprocess.CalledProcessError as e:
        if check:
            raise RuntimeError(f"Command failed with exit {e.returncode}: {e.stderr}")
        return e


def download_file(url: str, dest: str) -> None:
    """Download a file via urllib (works everywhere without wget/curl)."""
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def wait_for_ollama(max_retries: int = 30, delay: float = 2) -> bool:
    """Poll Ollama health endpoint until it responds."""
    for _ in range(max_retries):
        try:
            req = urllib.request.Request(f"http://{OLLAMA_HOST}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(delay)
    return False

# ======================== SETUP FUNCTIONS ========================
def kill_process(name: str) -> None:
    """Kill a process by name (ignore errors)."""
    try:
        subprocess.run(["pkill", "-f", name], capture_output=True, check=False)
    except Exception:
        pass


def clean_environment() -> None:
    """Remove stale processes and binaries."""
    kill_process("ollama")
    kill_process("cloudflared")
    for f in [OLLAMA_BIN, CLOUDFLARED_BIN, CLOUDFLARED_LOG]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


def download_ollama() -> None:
    """Download Ollama binary from official release URL."""
    url = "https://ollama.com/download/ollama-linux-amd64"
    download_file(url, OLLAMA_BIN)
    os.chmod(OLLAMA_BIN, 0o755)


def start_ollama_server() -> None:
    """Launch ollama serve in the background."""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    env["OLLAMA_ORIGINS"] = "*"

    with open("ollama.log", "w") as log:
        subprocess.Popen(["./ollama", "serve"], stdout=log, stderr=log, env=env)

    if not wait_for_ollama():
        raise RuntimeError("Ollama server failed to start within timeout")


def pull_model_with_retry(model: str, max_retries: int = 3) -> None:
    """Pull model with retries on network errors."""
    for attempt in range(max_retries):
        try:
            result = run_cmd(["./ollama", "pull", model], timeout=600, check=False)
            if result.returncode == 0:
                return
            # If pull failed, wait and retry
            time.sleep(5 * (attempt + 1))
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    raise RuntimeError(f"Failed to pull model '{model}' after {max_retries} attempts")


def download_cloudflared() -> None:
    """Download cloudflared binary."""
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    download_file(url, CLOUDFLARED_BIN)
    os.chmod(CLOUDFLARED_BIN, 0o755)


def start_cloudflared_tunnel() -> Optional[str]:
    """
    Launch cloudflared tunnel and extract the public URL from its log.
    Returns the URL string or None if not found.
    """
    # Ensure clean log file
    if os.path.exists(CLOUDFLARED_LOG):
        os.remove(CLOUDFLARED_LOG)

    with open(CLOUDFLARED_LOG, "w") as log:
        subprocess.Popen(
            ["./cloudflared", "tunnel", "--url", f"http://127.0.0.1:{OLLAMA_PORT}"],
            stdout=log, stderr=log
        )

    # Wait for the log to contain a 'trycloudflare.com' URL
    for _ in range(15):
        time.sleep(2)
        if not os.path.exists(CLOUDFLARED_LOG):
            continue
        try:
            with open(CLOUDFLARED_LOG, "r") as f:
                content = f.read()
                # Search for a public URL pattern
                import re
                matches = re.findall(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', content)
                if matches:
                    return matches[-1]  # most recent URL
        except Exception:
            pass
    return None

# ======================== MAIN ORCHESTRATION ========================
def main():
    try:
        # 1. Hardware detection
        ram_gb, vram_gb, total_gb = get_available_memory_gb()
        if total_gb <= 0:
            raise RuntimeError("Could not determine system memory")

        # 2. Model recommendation
        recommended_model = recommend_model(total_gb)

        # 3. Clean and prepare
        clean_environment()
        download_ollama()
        start_ollama_server()
        pull_model_with_retry(recommended_model)

        # 4. Expose via tunnel
        download_cloudflared()
        public_url = start_cloudflared_tunnel()

        if public_url:
            print(f"{public_url}")
        else:
            raise RuntimeError("Could not establish Cloudflare tunnel")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
