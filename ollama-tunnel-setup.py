#!/usr/bin/env python3
"""
Ollama + Cloudflare Tunnel – Production‑Ready Deployer
- Installs essential system tools (zstd, curl)
- Installs Ollama via official script (no manual extraction)
- Installs cloudflared via official .deb package (avoids binary issues)
- Automatically selects the optimal model based on total memory
- Exposes the API securely through a Cloudflare TryCloudflare tunnel
- Safe cleanup on exit, no self‑termination, clean English output
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
OLLAMA_PORT = 11434
CF_LOG = Path("cf.log")
OLLAMA_LOG = Path("ollama.log")

MODELS = [
    (2, "qwen2.5-coder:3b-instruct-q6_K"),
    (4, "qwen2.5-coder:7b-instruct-q4_K_M"),
    (8, "qwen2.5-coder:7b-instruct-q6_K"),
    (12, "qwen2.5-coder:14b-instruct-q4_K_M"),
    (24, "qwen2.5-coder:14b-instruct-q6_K"),
    (40, "qwen2.5-coder:32b-instruct-q4_K_M"),
]

# Global handles for cleanup
processes = []
INSTALLED_OLLAMA = None   # path to the real ollama binary

# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ======================================================================
# Helper utilities
# ======================================================================
def safe_kill(proc):
    """Terminate a subprocess gracefully, then force kill."""
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
    """Kill all background processes and remove log files."""
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

def safe_pkill(proc_name):
    """Kill processes by name, excluding the current script PID."""
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

def download_file(url, dest, desc="file", min_size_mb=10):
    """
    Download a file using curl with a browser User‑Agent.
    Raises an error if the file is too small (likely a download error).
    """
    log.info(f"Downloading {desc} from {url} ...")
    try:
        subprocess.run(
            [
                "curl", "-L", "--progress-bar", "--retry", "3",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "-o", str(dest),
                url
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("curl is required. Install it with: sudo apt install curl")
    size_mb = dest.stat().st_size / (1024*1024) if dest.exists() else 0
    if size_mb < min_size_mb:
        raise RuntimeError(f"Downloaded {desc} is too small ({size_mb:.1f} MB) – probably an error page.")
    log.info(f"{desc} downloaded ({size_mb:.1f} MB).")

def ensure_system_tools():
    """Install zstd and curl if they are missing (important for Colab)."""
    log.info("Checking/installing system tools (zstd, curl)...")
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["apt-get", "install", "-y", "-qq", "zstd", "curl"], check=True)
        log.info("System tools ready.")
    except Exception as e:
        log.warning(f"Could not install zstd/curl via apt: {e}")

def test_binary(path):
    """Return True if the path is an executable file."""
    return Path(path).is_file() and os.access(path, os.X_OK)

# ======================================================================
# Ollama installation – official script, no manual archive extraction
# ======================================================================
def get_ollama():
    """
    Install Ollama using the official install script.
    Sets the global INSTALLED_OLLAMA to the real binary path.
    """
    global INSTALLED_OLLAMA
    log.info("Installing Ollama via official install script...")
    try:
        subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True, check=True, executable="/bin/bash",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log.info("Ollama installed successfully via official script.")
    except subprocess.CalledProcessError:
        raise RuntimeError("Official Ollama install script failed.")

    # Locate the installed binary (usually /usr/local/bin/ollama or /usr/bin/ollama)
    possible_paths = ["/usr/local/bin/ollama", "/usr/bin/ollama"]
    installed = None
    for p in possible_paths:
        if os.path.exists(p):
            installed = p
            break
    if installed is None:
        installed = shutil.which("ollama")
    if installed is None:
        raise RuntimeError("Cannot find installed ollama binary.")

    # Resolve symlinks to get the real file
    installed_real = os.path.realpath(installed)
    if not os.path.exists(installed_real):
        raise RuntimeError(f"Resolved path {installed_real} does not exist.")
    if not test_binary(installed_real):
        raise RuntimeError(f"Installed ollama at {installed_real} is not executable.")

    INSTALLED_OLLAMA = installed_real
    log.info(f"Using Ollama binary: {INSTALLED_OLLAMA}")

# ======================================================================
# Cloudflared installation – official .deb package (avoids binary issues)
# ======================================================================
def get_cloudflared():
    """Download and install cloudflared using the official .deb package."""
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
        desc = "cloudflared (amd64 deb)"
    elif arch in ("aarch64", "arm64"):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb"
        desc = "cloudflared (arm64 deb)"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    deb_path = Path("./cloudflared.deb")
    download_file(url, deb_path, desc, min_size_mb=5)

    log.info(f"Installing {desc}...")
    try:
        subprocess.run(["dpkg", "-i", str(deb_path)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        log.warning("dpkg failed, attempting to fix dependencies...")
        subprocess.run(["apt-get", "install", "-f", "-y"], check=True)
        subprocess.run(["dpkg", "-i", str(deb_path)], check=True)
    finally:
        deb_path.unlink(missing_ok=True)

    cloudflared_path = shutil.which("cloudflared")
    if not cloudflared_path:
        raise RuntimeError("cloudflared not found after .deb installation.")
    log.info(f"cloudflared installed at: {cloudflared_path}")

# ======================================================================
# System detection
# ======================================================================
def total_memory_gb():
    """Detect total RAM + GPU VRAM in GB."""
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
            timeout=5,
        ).decode()
        vram = sum(int(x) for x in out.strip().splitlines()) / 1024.0
    except Exception:
        pass
    total = ram + vram
    if total < 1.0:
        total = 2.0
    log.info(f"Total memory: {total:.1f} GB (RAM {ram:.1f} + VRAM {vram:.1f})")
    return total

def select_model(mem_gb):
    """Choose the largest model that fits in 85% of total memory."""
    usable = mem_gb * 0.85
    selected = None
    for th, model in MODELS:
        if usable >= th:
            selected = model
        else:
            break
    if selected is None:
        selected = MODELS[0][1]
    log.info(f"Selected model: {selected} (usable {usable:.1f} GB)")
    return selected

def port_in_use(port):
    """Check if a TCP port is open on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def kill_port_process(port):
    """Kill any process listening on the given port using fuser."""
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
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
        [INSTALLED_OLLAMA, "pull", model_name],
        stdout=sys.stdout, stderr=subprocess.STDOUT, check=True
    )

def start_ollama_server():
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{OLLAMA_PORT}"
    env["OLLAMA_ORIGINS"] = "*"
    log_file = open(OLLAMA_LOG, "w")
    proc = subprocess.Popen(
        [INSTALLED_OLLAMA, "serve"],
        stdout=log_file, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setpgrp,
    )
    processes.append(proc)

def start_cloudflared():
    cloudflared_cmd = shutil.which("cloudflared")
    if not cloudflared_cmd:
        raise RuntimeError("cloudflared not found in PATH.")
    log_file = open(CF_LOG, "w")
    proc = subprocess.Popen(
        [cloudflared_cmd, "tunnel", "--url", f"http://127.0.0.1:{OLLAMA_PORT}"],
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp,
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
        # Monitor cloudflared process
        for p in processes:
            if p and p.poll() is not None:
                raise RuntimeError("cloudflared process died unexpectedly.")
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
        ensure_system_tools()

        log.info("Preparing environment (stopping old instances)...")
        safe_pkill("ollama")
        safe_pkill("cloudflared")
        if port_in_use(OLLAMA_PORT):
            log.info("Freeing port 11434...")
            kill_port_process(OLLAMA_PORT)
            if port_in_use(OLLAMA_PORT):
                raise RuntimeError("Port 11434 still in use after kill attempt.")

        for f in [OLLAMA_LOG, CF_LOG]:
            f.unlink(missing_ok=True)

        # Install/verify Ollama
        if INSTALLED_OLLAMA is None or not test_binary(INSTALLED_OLLAMA):
            get_ollama()

        # Start Ollama server
        log.info("Starting Ollama server...")
        start_ollama_server()
        if not wait_for_ollama():
            raise RuntimeError("Ollama server did not start. Check ollama.log")

        # Memory detection & model pull
        mem = total_memory_gb()
        model = select_model(mem)
        pull_model(model)

        # Install cloudflared if missing
        if not shutil.which("cloudflared"):
            get_cloudflared()
        else:
            log.info("cloudflared already installed.")

        log.info("Starting Cloudflare tunnel...")
        start_cloudflared()
        url = extract_tunnel_url()

        print(url, flush=True)
        print_warning(url)

        log.info("Services running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
