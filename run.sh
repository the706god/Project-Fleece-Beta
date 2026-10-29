#!/usr/bin/env bash
# Project: Fleece Beta - Multimodal Turnkey Launcher (Mimir OCR Active)

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Detect Python environment with llama_cpp & Mimir OCR support
if [ -f "$DIR/.venv/bin/python" ]; then
    PY="$DIR/.venv/bin/python"
elif [ -f "/Users/ATLAS/Desktop/CODING/MSNFH INC/charon/.venv/bin/python" ]; then
    PY="/Users/ATLAS/Desktop/CODING/MSNFH INC/charon/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo "Error: Python 3 not found."
    exit 1
fi

echo "🚀 Starting Project: Fleece Beta (Multimodal + Mimir OCR) using: $PY"
if [ "$(uname -s)" = "Darwin" ] && /usr/sbin/sysctl -n hw.optional.arm64 >/dev/null 2>&1; then
    exec arch -arm64 "$PY" fleece.py "$@"
else
    exec "$PY" fleece.py "$@"
fi
