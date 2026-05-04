#!/usr/bin/env python3
"""
Ollama + Cloudflare Tunnel deployment script
Safe process cleanup, automatic model selection, quiet output.
"""

import os
import sys
import time
import signal
import subprocess
import urllib.request
import urllib.error
import re
import tempfile
import shutil
from pathlib import Path

# ======================== CONFIG ========================
OLLAMA_BIN = "./ollama_tmp"          # اسم مؤقت لتجنب قتله بالخطأ
OLLAMA_PORT = 11434
CLOUDFLARED_BIN = "./cloudflared_tmp"
CLOUDFLARED_LOG = "cf_log.txt"
OLLAMA_LOG = "ollama_log.txt"

# قائمة النماذج حسب الذاكرة (بالـ GB)
MODEL_TIERS = [
    (2, "qwen2.5-coder:3b-instruct-q6_K"),
    (4, "qwen2.5-coder:7b-instruct-q4_K_M"),
    (8, "qwen2.5-coder:7b-instruct-q6_K"),
    (12, "qwen2.5-coder:14b-instruct-q4_K_M"),
    (24, "qwen2.5-coder:14b-instruct-q6_K"),
    (40, "qwen2.5-coder:32b-instruct-q4_K_M"),
]

# ======================== HELPER FUNCTIONS ========================
def run_cmd(cmd, check=False, timeout=300, capture=True):
    """Run a shell command, return result."""
    try:
        if capture:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, check=check)
        else:
            return subprocess.run(cmd, shell=True, timeout=timeout, check=check)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out: {cmd}")
    except subprocess.CalledProcessError as e:
        if check:
            raise RuntimeError(f"Command failed: {e}")
        return e

def safe_kill(proc_name):
    """Kill process by name without killing self."""
    # تجنب قتل العملية الحالية إذا كان اسمها يحوي proc_name
    try:
        # استخدم pgrep للحصول على PIDs ثم قتلهم
        pids = subprocess.getoutput(f"pgrep -f '{proc_name}'").strip().split()
        mypid = str(os.getpid())
        for pid in pids:
            if pid != mypid:
                os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass

def download_file(url, dest):
    """Download file with urllib."""
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

def get_total_memory_gb():
    """Get total RAM + GPU VRAM safely."""
    ram = 0
    try:
        import psutil
        ram = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # fallback
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram = int(line.split()[1]) / (1024 * 1024)
                        break
        except:
            ram = 2.0  # تخمين آمن
    vram = 0
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            vram = float(result.stdout.strip().split()[0]) / 1024
    except:
        pass
    return ram + vram

def recommend_model(total_gb):
    for threshold, model in MODEL_TIERS:
        if total_gb >= threshold:
            return model
    return MODEL_TIERS[0][1]

def wait_for_ollama(max_attempts=30, delay=2):
    for _ in range(max_attempts):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except:
            time.sleep(delay)
    return False

# ======================== MAIN ========================
def main():
    try:
        # 1. تثبيت psutil إذا لم يكن موجوداً (بدون رسائل)
        try:
            import psutil
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil"], capture_output=True)

        # 2. تنظيف آمن – لا نستخدم pkill مباشرة
        safe_kill("ollama")
        safe_kill("cloudflared")
        # إغلاق المنفذ إذا كان مشغولاً
        run_cmd(f"fuser -k {OLLAMA_PORT}/tcp 2>/dev/null || true", check=False)
        # حذف الملفات القديمة
        for f in [OLLAMA_BIN, CLOUDFLARED_BIN, OLLAMA_LOG, CLOUDFLARED_LOG]:
            if os.path.exists(f):
                os.remove(f)

        # 3. تحميل Ollama
        download_file("https://ollama.com/download/ollama-linux-amd64", OLLAMA_BIN)
        os.chmod(OLLAMA_BIN, 0o755)

        # 4. تشغيل الخادم في الخلفية
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"0.0.0.0:{OLLAMA_PORT}"
        env["OLLAMA_ORIGINS"] = "*"

        with open(OLLAMA_LOG, "w") as log:
            proc = subprocess.Popen([OLLAMA_BIN, "serve"], stdout=log, stderr=log, env=env)

        if not wait_for_ollama():
            raise RuntimeError("Ollama server did not start")

        # 5. تحديد النموذج وسحبه
        total_mem = get_total_memory_gb()
        model = recommend_model(total_mem)

        pull_cmd = f"{OLLAMA_BIN} pull {model}"
        result = run_cmd(pull_cmd, check=False, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull {model}")

        # 6. تحميل cloudflared
        download_file("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", CLOUDFLARED_BIN)
        os.chmod(CLOUDFLARED_BIN, 0o755)

        # 7. تشغيل النفق
        with open(CLOUDFLARED_LOG, "w") as log:
            subprocess.Popen([CLOUDFLARED_BIN, "tunnel", "--url", f"http://127.0.0.1:{OLLAMA_PORT}"],
                            stdout=log, stderr=log)

        # 8. استخراج الرابط
        url = None
        for _ in range(15):
            time.sleep(2)
            if os.path.exists(CLOUDFLARED_LOG):
                with open(CLOUDFLARED_LOG, "r") as f:
                    content = f.read()
                    matches = re.findall(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', content)
                    if matches:
                        url = matches[-1]
                        break

        if url:
            print(url)
        else:
            raise RuntimeError("No tunnel URL found")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
