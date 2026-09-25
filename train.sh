#!/usr/bin/env bash
# Train Xenarch (Mk15) and see every run at http://localhost:8050
#   ./train.sh                     train on data/xenarch-imagery
#   ./train.sh "training data"     train on any image folder
#   ./train.sh --mk18 [folder]     Mk18 PatchCore instead of the Mk15 VAE (no training, ~5 min)
#   ./train.sh data/xenarch-imagery --latent-channels 8   Mk15 with a sharper spatial latent
#   ./train.sh --view              only open the dashboard
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8050}"
SCRIPT=xenarch_mk15_script.py
if [ "${1:-}" = "--mk18" ]; then SCRIPT=xenarch_mk18_script.py; shift; fi
DATA="${1:-data/xenarch-imagery}"
shift || true          # anything after the folder goes to the script, e.g. --latent-channels 8
mkdir -p logs

if [ ! -x .venv/bin/python ]; then
  echo "Setting up .venv (first run only)…"
  python3 -m venv .venv
  .venv/bin/pip install -q torch torchvision rasterio pandas scipy scikit-learn matplotlib seaborn tqdm loguru pillow huggingface_hub
fi

if curl -s "http://localhost:$PORT/" | grep -q "<title>Xenarch Training Runs</title>"; then
  :   # dashboard already running
elif lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is used by something else — try: PORT=8060 ./train.sh"; exit 1
else
  nohup .venv/bin/python -m http.server "$PORT" --bind 127.0.0.1 --directory dashboard >logs/dashboard.log 2>&1 &
  sleep 1
fi
echo "Dashboard: http://localhost:$PORT"
open "http://localhost:$PORT" 2>/dev/null || true
[ "$DATA" = "--view" ] && exit 0

if [ ! -d "$DATA" ]; then
  echo "No image folder '$DATA'. Download the dataset with:"
  echo "  .venv/bin/hf download Giulia928982/xenarch-imagery --repo-type dataset --include '*_png/*' --local-dir data/xenarch-imagery"
  exit 1
fi

LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -u "$SCRIPT" --data "$DATA" "$@" 2>&1 | tee "$LOG"
echo "Done — the new run is on http://localhost:$PORT"
