#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACE_REPO="$ROOT/.vendor/ACE-Step-1.5"
PYTHON_BIN="${PYTHON_BIN:-python3}"

say() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

install_system_package() {
  local package="$1"
  if command -v apt-get >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then apt-get update -y && apt-get install -y "$package"
    elif command -v sudo >/dev/null 2>&1; then sudo apt-get update -y && sudo apt-get install -y "$package"
    else say "Need root or sudo to install $package."; return 1; fi
  elif command -v dnf >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then dnf install -y "$package"
    elif command -v sudo >/dev/null 2>&1; then sudo dnf install -y "$package"
    else say "Need root or sudo to install $package."; return 1; fi
  else
    say "Unsupported package manager. Please install $package and rerun."
    return 1
  fi
}

if ! command -v git >/dev/null 2>&1; then
  say "Git is missing; installing it."
  install_system_package git
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  say "Python is missing; install Python 3.11 or 3.12 and rerun."
  exit 2
fi
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else 1)' || {
  say "ACE-Step 1.5 requires Python 3.11 or 3.12; found: $("$PYTHON_BIN" --version 2>&1)"
  exit 2
}

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv using its official installer."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    say "Install curl or wget, then rerun so uv can be installed."
    exit 2
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [[ ! -d "$ACE_REPO/.git" ]]; then
  mkdir -p "$(dirname "$ACE_REPO")"
  say "Cloning official ACE-Step 1.5 source."
  git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "$ACE_REPO"
fi

say "Installing ACE-Step dependencies in its isolated uv environment."
cd "$ACE_REPO"
uv sync

cd "$ROOT"
say "Starting background music generation."
exec uv run --project "$ACE_REPO" python "$ROOT/generate_music.py" "$@"

