#!/usr/bin/env bash
# Project: Fleece Beta - Multimodal Turnkey Launcher

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Detect Python environment with llama_cpp support
if [ -f "$DIR/.venv/bin/python" ]; then
    PY="$DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo "Error: Python 3 not found."
    exit 1
fi

echo "🚀 Starting Project: Fleece Beta (Multimodal) using: $PY"
exec "$PY" fleece.py "$@"
