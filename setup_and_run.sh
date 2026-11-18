#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"

say() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

install_system_packages() {
  local packages=("$@")
  if command -v apt-get >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
      apt-get update -y && apt-get install -y "${packages[@]}"
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update -y && sudo apt-get install -y "${packages[@]}"
    else
      say "Cannot install ${packages[*]} automatically: apt-get needs root or sudo."
      return 1
    fi
  elif command -v dnf >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
      dnf install -y "${packages[@]}"
    elif command -v sudo >/dev/null 2>&1; then
      sudo dnf install -y "${packages[@]}"
    else
      say "Cannot install ${packages[*]} automatically: dnf needs root or sudo."
      return 1
    fi
  elif command -v yum >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
      yum install -y "${packages[@]}"
    elif command -v sudo >/dev/null 2>&1; then
      sudo yum install -y "${packages[@]}"
    else
      say "Cannot install ${packages[*]} automatically: yum needs root or sudo."
      return 1
    fi
  elif command -v apk >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
      apk add --no-cache "${packages[@]}"
    elif command -v sudo >/dev/null 2>&1; then
      sudo apk add --no-cache "${packages[@]}"
    else
      say "Cannot install ${packages[*]} automatically: apk needs root or sudo."
      return 1
    fi
  else
    say "No supported package manager found. Install ${packages[*]} and rerun."
    return 1
  fi
}

install_python_runtime() {
  if command -v apt-get >/dev/null 2>&1; then
    install_system_packages python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
    install_system_packages python3 python3-pip
  elif command -v apk >/dev/null 2>&1; then
    install_system_packages python3 py3-pip
  else
    say "Install Python 3.10–3.12 and pip, then rerun."
    return 1
  fi
}

# The PyPI install does not require Git. Only install it if the user opts into
# the upstream Git install, avoiding unnecessary package work on paid GPU time.
if [[ "${QWEN_TTS_SOURCE:-pypi}" == "git" ]] && ! command -v git >/dev/null 2>&1; then
  say "Git is missing; installing it because QWEN_TTS_SOURCE=git."
  install_system_packages git
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  say "Python 3.10–3.12 is missing; attempting to install Python and venv support."
  install_python_runtime
  PYTHON_BIN="$(command -v python3)"
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
  say "The selected interpreter is not Python 3.10–3.12. Set PYTHON_BIN to a supported interpreter and rerun."
  exit 2
fi

if ! "$PYTHON_BIN" -m venv "$VENV" 2>/dev/null; then
  say "Python venv support is missing; installing it and retrying."
  if command -v apt-get >/dev/null 2>&1; then
    install_system_packages python3-venv
  else
    say "This Python build lacks venv support. Install a Python build with venv enabled, then rerun."
    exit 2
  fi
  "$PYTHON_BIN" -m venv "$VENV"
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
ENV_READY=0
if [[ "${QWEN_TTS_SOURCE:-pypi}" == "pypi" ]] && \
  "$PY" -c 'import qwen_tts, soundfile, torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  ENV_READY=1
fi

if [[ "$ENV_READY" -eq 1 ]]; then
  say "Existing Qwen/CUDA environment found; reusing installed packages."
else
  "$PIP" install --upgrade pip setuptools wheel

# Install a CUDA-enabled PyTorch wheel matching the NVIDIA driver reported by
# nvidia-smi. PyTorch provides these official per-CUDA package indexes.
if command -v nvidia-smi >/dev/null 2>&1; then
  # Driver-reported CUDA is a maximum runtime compatibility level, not a
  # required wheel version. Try current official indexes from newest to older;
  # after each install, verify that PyTorch can actually initialize the GPU.
  CUDA_READY=0
  for CUDA_INDEX in cu130 cu128 cu126 cu124 cu121 cu118; do
    say "Trying official PyTorch $CUDA_INDEX wheels."
    if "$PIP" install --upgrade torch torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" && \
      "$PY" -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(), "PyTorch cannot access the GPU"; print(torch.cuda.get_device_name(0))'; then
      CUDA_READY=1
      break
    fi
  done
  if [[ "$CUDA_READY" -ne 1 ]]; then
    say "Could not install a working CUDA PyTorch build. Check the GPU driver/runtime or use a rental image with PyTorch and CUDA preinstalled."
    exit 2
  fi
elif command -v rocm-smi >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1; then
  say "AMD ROCm GPU detected; installing PyTorch ROCm 7.1 wheels."
  "$PIP" install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
else
  say "No NVIDIA/AMD GPU management tool detected. Installing Qwen anyway; the runner will stop if no GPU is available."
fi

if [[ "${QWEN_TTS_SOURCE:-pypi}" == "git" ]]; then
  command -v git >/dev/null 2>&1 || { say "Git is required for QWEN_TTS_SOURCE=git."; exit 2; }
  "$PIP" install --upgrade "git+https://github.com/QwenLM/Qwen3-TTS.git"
else
  "$PIP" install --upgrade -r requirements.txt
fi
fi

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME"

say "Starting Qwen narration generation. Model downloads automatically on first run."
RUNNER="$ROOT/bulk_generate.py"
case "${QWEN_TTS_MODE:-voice-design}" in
  voice-design) ;;
  clone) RUNNER="$ROOT/voice_clone_generate.py" ;;
  *) say "QWEN_TTS_MODE must be 'voice-design' or 'clone'."; exit 2 ;;
esac
exec "$PY" "$RUNNER" "$@"
