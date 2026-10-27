#!/bin/bash
# Historical filename retained; the default device remains GPU 0.
# The coordinator manages provenance, recovery and subprocesses only.
set -euo pipefail
project_root="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export PROJECT_ROOT="$project_root"
if ! command -v conda >/dev/null && [[ -f "$project_root/activate_setup_only.sh" ]]; then
    # Existing lab setup is configuration only; it never starts reconstruction.
    source "$project_root/activate_setup_only.sh"
fi
exec "${BATCH_PYTHON:-python3}" "$project_root/scripts/reconstruction_batch.py" "$@"
