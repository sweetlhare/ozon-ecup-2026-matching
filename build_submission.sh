#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
    echo "usage: $0 mix408_textneg|final_safe_fwdgate [builder options]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$ROOT/scripts/submission.py" build "$@"
