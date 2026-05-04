#!/usr/bin/env python3
"""
Adaptive Ollama + Cloudflare Tunnel for Colab (T4) and any Linux
Auto-detects ARM64/AMD64, falls back to working binary.
"""

import os
import sys
import time
import signal
import subprocess
import urllib.request
import platform
import re

# Config
OLLAMA_BIN = "./ollama_bin"
OLLAMA_PORT = 11434
CF_BIN = "./cf_bin"
CF_LOG = "cf.log"
OLLAMA_LOG = "ollama.log"

# Models by total RAM+VRAM (GB)
MODELS = [
    (2, "qwen2.5-coder:3b-instruct-q6_K"),
    (4, "qwen2.5-coder:7b-instruct-q4_K_M"),
    (8, "qwen2.5-coder:7b-instruct-q6_K"),
    (12, "qwen2.5-coder:14b-instruct-q4_K_M"),
    (24, "qwen2.5-coder:14b-instruct-q6_K"),
    (40, "qwen2.5-coder:32b-instruct-q4_K_M"),
]

# ------------------------------------------------------------
def run(cmd):
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def kill(name):
    for p in subprocess.getoutput(f"pgrep -f '{name}'").split():
        if p != str(os.getpid()):
            os.kill(int(p), signal.SIGKILL)

def download(url, dst):
    urllib.request.urlretrieve(url, dst)

def test_binary(path):
    try:
        return subprocess.run([path, "--version"], capture_output=True, timeout=5).returncode == 0
    except:
        return False

def get_ollama():
    arch = platform.machine()
    if arch in ["x86_64", "amd64"]:
        candidates = [("amd64", "https://ollama.com/download/ollama-linux-amd64"),
                      ("arm64", "https://ollama.com/download/ollama-linux-arm64")]
    else:
        candidates = [("arm64", "https://ollama.com/download/ollama-linux-arm64"),
                      ("amd64", "https://ollama.com/download/ollama-linux-amd64")]
    for name, url in candidates:
        download(url, OLLAMA_BIN)
        os.chmod(OLLAMA_BIN, 0o755)
        if test_binary(OLLAMA_BIN):
            return True
        os.remove(OLLAMA_BIN)
    raise RuntimeError("No working Ollama binary")

def get_cf():
    arch = platform.machine()
    if arch in ["x86_64", "amd64"]:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    else:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
    download(url, CF_BIN)
    os.chmod(CF_BIN, 0o755)
    if not test_binary(CF_BIN):
        raise RuntimeError("Cloudflared not executable")

def total_memory_gb():
    ram = 0
    try:
        import psutil
        ram = psutil.virtual_memory().total / (1024**3)
    except:
        try:
            for l in open("/proc/meminfo"):
                if l.startswith("MemTotal:"):
                    ram = int(l.split()[1]) / (1024*1024)
                    break
        except:
            ram = 2.0
    vram = 0
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], timeout=5).decode()
        vram = float(out.strip().split()[0]) / 1024
    except:
        pass
    return ram + vram

def wait_for_ollama():
    for _ in range(30):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    return True
        except:
            time.sleep(2)
    return False

# ------------------------------------------------------------
def main():
    try:
        # Install psutil if missing
        try:
            import psutil
        except:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil"])

        # Cleanup
        kill("ollama")
        kill("cloudflared")
        run(f"fuser -k {OLLAMA_PORT}/tcp 2>/dev/null")
        for f in [OLLAMA_BIN, CF_BIN, OLLAMA_LOG, CF_LOG]:
            if os.path.exists(f):
                os.remove(f)

        # Get Ollama
        get_ollama()
        # Start server
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"0.0.0.0:{OLLAMA_PORT}"
        env["OLLAMA_ORIGINS"] = "*"
        with open(OLLAMA_LOG, "w") as log:
            subprocess.Popen([OLLAMA_BIN, "serve"], stdout=log, stderr=log, env=env)
        if not wait_for_ollama():
            raise RuntimeError("Ollama server not ready")

        # Pull model
        mem = total_memory_gb()
        model = next((m for th,m in MODELS if mem >= th), MODELS[0][1])
        if subprocess.run([OLLAMA_BIN, "pull", model]).returncode != 0:
            raise RuntimeError(f"Pull failed: {model}")

        # Cloudflared
        get_cf()
        with open(CF_LOG, "w") as f:
            subprocess.Popen([CF_BIN, "tunnel", "--url", f"http://127.0.0.1:{OLLAMA_PORT}"],
                             stdout=f, stderr=f)

        # Get URL
        url = None
        for _ in range(15):
            time.sleep(2)
            if os.path.exists(CF_LOG):
                with open(CF_LOG) as f:
                    m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', f.read())
                    if m:
                        url = m.group()
                        break
        if url:
            print(url)
        else:
            raise RuntimeError("No tunnel URL")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
