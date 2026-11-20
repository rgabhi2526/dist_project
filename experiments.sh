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

is_complete() {  # is_complete <run_dir>: exit 0 iff summary.json says the run finished its budget
    # Summaries written before the "complete" flag existed count as done if they have a final acc.
    "$PY" -c 'import json,sys
try: s = json.load(open(sys.argv[1] + "/summary.json"))
except (OSError, ValueError): sys.exit(1)
sys.exit(0 if s.get("complete", s.get("final_acc") is not None) else 1)' "$1"
}

run() {  # run <exp> <name> <seed> <flags...>   (resumable: finished runs are skipped)
    local exp=$1 name=$2 seed=$3; shift 3
    local dir="results/$exp/$name/$seed"
    if is_complete "$dir"; then
        echo "SKIP  $dir (already complete)"
        return
    fi
    if [[ -d "$dir" ]]; then
        echo "RERUN $dir (incomplete or interrupted, deleting it)"
        rm -rf "$dir"
    fi
    local t0=$SECONDS
    echo "=== [$exp] $name seed $seed  start $(date +%H:%M:%S)  budget ${BUDGET}s ==="
    "$PY" -m flsim.run --seed "$seed" --out "$dir" $COMMON "$@"
    echo "=== [$exp] $name seed $seed  done in $((SECONDS - t0))s ==="
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
