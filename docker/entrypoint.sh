#!/usr/bin/env bash
# Finn-Predictor container entrypoint.
#
# Behaviour by ARG ($1):
#   serve         (default) → run Streamlit on 0.0.0.0:8501
#   reset-db                → drop + recreate the schema, then exit
#   retrain                 → run one train_weights cycle, then exit
#   shell                   → drop into bash for debugging
#   <anything else>         → forward verbatim to `python -m
#                             finn_predictor.cli ...` (so future CLI
#                             subcommands work without changing this
#                             script)
#
# Behaviour by ENV:
#   RESET_DB=1              → before `serve`, run reset-db --yes. Lets
#                             you wipe data without docker compose run
#                             gymnastics. Ignored unless ARG is serve.
#
# Examples:
#   docker compose up                            # use existing volume
#   RESET_DB=1 docker compose up                 # wipe + start fresh
#   docker compose run --rm app reset-db --yes   # wipe, exit
#   docker compose run --rm app retrain          # retrain, exit
#   docker compose run --rm app shell            # interactive shell

set -euo pipefail

cd /app

cmd="${1:-serve}"
shift || true

if [[ "${cmd}" == "serve" ]]; then
    if [[ "${RESET_DB:-0}" == "1" || "${RESET_DB:-}" == "true" ]]; then
        echo "RESET_DB is set — wiping the database before serving."
        python -m finn_predictor.cli reset-db --yes
    fi
    exec streamlit run finn_predictor/ui/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false
fi

if [[ "${cmd}" == "shell" ]]; then
    exec /bin/bash "$@"
fi

# Forward everything else to the CLI. Covers reset-db, retrain, and
# any future subcommand without an entrypoint edit.
exec python -m finn_predictor.cli "${cmd}" "$@"
