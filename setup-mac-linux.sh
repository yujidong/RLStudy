#!/usr/bin/env bash
# RLStudy one-click setup (macOS / Linux).
# Thin wrapper: find Python 3, then run the shared driver setup.py
# (all logic and Chinese output live there).
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[X] python3 not found."
    echo "    macOS          : brew install python  (or installer from python.org)"
    echo "    Ubuntu/Debian  : sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

exec python3 setup.py "$@"
