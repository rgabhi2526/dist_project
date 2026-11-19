#!/usr/bin/env bash
# experiments.sh -- the experiment matrix (decisions.md D6, D8, D9), one line per run.
#
# Usage:  bash experiments.sh pilot   # 1 seed, both presets, Libra distances logged (picks L, T)
#         bash experiments.sh e1      # fedavg / seafl / libra x seeds 0 1 2
# Results: results/<exp>/<algo>[_<preset>]/<seed>/{config.json,evals.csv,events.jsonl,summary.json}
# Run on the Jarvis L4 inside tmux; metrics are flushed per eval so a crash keeps partial data.
set -euo pipefail

PY="${PY:-python}"
BUDGET="${BUDGET:-1200}"          # wall-time budget in seconds, same for every algorithm (D8)
COMMON="--budget-s $BUDGET ${EXTRA:-}"   # e.g. EXTRA="--base-delay 2 --time-scale 0.5"

run() {  # run <exp> <name> <seed> <flags...>
    local exp=$1 name=$2 seed=$3; shift 3
    "$PY" -m flsim.run --seed "$seed" --out "results/$exp/$name/$seed" $COMMON "$@"
}

pilot() {
    # --sinkhorn-L is left unset so Libra only LOGS distances; L is picked from them afterwards.
    run pilot fedavg 0 --algo fedavg
    # Libra runs once: --preset only sets K, beta and n_active, which Libra does not use.
    run pilot libra 0 --algo libra
    for preset in paper scaled; do
        run pilot "seafl_$preset" 0 --algo seafl --preset "$preset"
    done
}

e1() {
    # Main comparison. Presets per algorithm come from the pilot (D6); default scaled here.
    for seed in 0 1 2; do
        run e1 fedavg "$seed" --algo fedavg
        run e1 seafl  "$seed" --algo seafl --preset scaled
        run e1 libra  "$seed" --algo libra --preset scaled
    done
}

case "${1:-}" in
    pilot) pilot ;;
    e1) e1 ;;
    *) echo "usage: bash experiments.sh pilot|e1" >&2; exit 1 ;;
esac
