#!/usr/bin/env bash
set -euo pipefail

## Paste method for RunPod terminals without nano/vim:
## 1. In the RunPod shell, start a heredoc:
##    cat > runpod-bootstrap.sh <<'SH'
## 2. Paste this entire file.
## 3. Paste the closing marker on its own line:
##    SH
## 4. Make it executable:
##    chmod +x runpod-bootstrap.sh
## 5. Run it with the required env vars:
##    export OPENSCADEMCP_REPO_URL="YOUR_REPO_URL"
##    export REMOTE_CUDA_MVS_API_KEY="YOUR_SHARED_SECRET"
##    ./runpod-bootstrap.sh

# One-shot bootstrap for a RunPod GPU pod that will host the remote CUDA MVS server.
#
# Required env vars:
OPENSCADEMCP_REPO_URL   Git URL for this repo
REMOTE_CUDA_MVS_API_KEY Shared secret used by the local app
#
# Optional env vars:
WORKSPACE_DIR           Default: /workspace
OPENSCADEMCP_DIRNAME    Default: openscademcp
CUDA_MVS_REPO_URL       Default: https://github.com/fixstars/cuda-multi-view-stereo.git
CUDA_MVS_DIRNAME        Default: cuda-multi-view-stereo
SERVER_PORT             Default: 8765
SERVER_NAME             Default: RunPod CUDA MVS
SERVER_OUTPUT_DIR       Default: <repo>/output/remote
TMUX_SESSION            Default: cudamvs
SKIP_APT                Default: 0

OPENSCADEMCP_REPO_URL="${OPENSCADEMCP_REPO_URL:-}"
REMOTE_CUDA_MVS_API_KEY="${REMOTE_CUDA_MVS_API_KEY:-}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
OPENSCADEMCP_DIRNAME="${OPENSCADEMCP_DIRNAME:-openscademcp}"
CUDA_MVS_REPO_URL="${CUDA_MVS_REPO_URL:-https://github.com/fixstars/cuda-multi-view-stereo.git}"
CUDA_MVS_DIRNAME="${CUDA_MVS_DIRNAME:-cuda-multi-view-stereo}"
SERVER_PORT="${SERVER_PORT:-8765}"
SERVER_NAME="${SERVER_NAME:-RunPod CUDA MVS}"
TMUX_SESSION="${TMUX_SESSION:-cudamvs}"
SKIP_APT="${SKIP_APT:-0}"

if [[ -z "$OPENSCADEMCP_REPO_URL" ]]; then
  printf 'Missing OPENSCADEMCP_REPO_URL\n' >&2
  exit 1
fi

if [[ -z "$REMOTE_CUDA_MVS_API_KEY" ]]; then
  printf 'Missing REMOTE_CUDA_MVS_API_KEY\n' >&2
  exit 1
fi

REPO_DIR="${WORKSPACE_DIR}/${OPENSCADEMCP_DIRNAME}"
CUDA_MVS_DIR="${WORKSPACE_DIR}/${CUDA_MVS_DIRNAME}"
SERVER_OUTPUT_DIR="${SERVER_OUTPUT_DIR:-${REPO_DIR}/output/remote}"
CUDA_MVS_BIN="${CUDA_MVS_DIR}/build/app_patch_match_mvs"

printf '\n==> Checking GPU\n'
nvidia-smi

if [[ "$SKIP_APT" != "1" ]]; then
  printf '\n==> Installing system dependencies\n'
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y git build-essential cmake python3 python3-pip python3-venv tmux curl
fi

printf '\n==> Preparing workspace\n'
mkdir -p "$WORKSPACE_DIR"

printf '\n==> Cloning or updating CUDA MVS repo\n'
if [[ -d "$CUDA_MVS_DIR/.git" ]]; then
  git -C "$CUDA_MVS_DIR" pull --ff-only
else
  git clone "$CUDA_MVS_REPO_URL" "$CUDA_MVS_DIR"
fi

printf '\n==> Building CUDA MVS\n'
mkdir -p "$CUDA_MVS_DIR/build"
cmake -S "$CUDA_MVS_DIR" -B "$CUDA_MVS_DIR/build"
cmake --build "$CUDA_MVS_DIR/build" -j"$(nproc)"

if [[ ! -x "$CUDA_MVS_BIN" ]]; then
  printf 'CUDA MVS executable not found at %s\n' "$CUDA_MVS_BIN" >&2
  exit 1
fi

printf '\n==> Cloning or updating OpenSCAD MCP repo\n'
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$OPENSCADEMCP_REPO_URL" "$REPO_DIR"
fi

printf '\n==> Creating Python virtualenv and installing dependencies\n'
python3 -m venv "$REPO_DIR/.venv"
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"

printf '\n==> Creating output directory\n'
mkdir -p "$SERVER_OUTPUT_DIR"

SERVER_CMD="cd \"$REPO_DIR\" && source .venv/bin/activate && python src/remote/cuda_mvs_server.py --host 0.0.0.0 --port $SERVER_PORT --cuda-mvs-path \"$CUDA_MVS_DIR\" --output-dir \"$SERVER_OUTPUT_DIR\" --api-key \"$REMOTE_CUDA_MVS_API_KEY\" --server-name \"$SERVER_NAME\" --no-advertise"

printf '\n==> Starting remote server in tmux session %s\n' "$TMUX_SESSION"
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  tmux kill-session -t "$TMUX_SESSION"
fi
tmux new-session -d -s "$TMUX_SESSION" "$SERVER_CMD"

printf '\n==> Waiting for server to start\n'
sleep 5
curl -fsS "http://127.0.0.1:${SERVER_PORT}/api/status"

printf '\n\nBootstrap complete.\n'
printf 'Repo dir: %s\n' "$REPO_DIR"
printf 'CUDA MVS dir: %s\n' "$CUDA_MVS_DIR"
printf 'Output dir: %s\n' "$SERVER_OUTPUT_DIR"
printf 'tmux session: %s\n' "$TMUX_SESSION"
printf 'Local health check: http://127.0.0.1:%s/api/status\n' "$SERVER_PORT"
printf '\nNext step on your Mac:\n'
printf 'Set REMOTE_CUDA_MVS_SERVER_URL to your RunPod public URL and restart the local app.\n'
