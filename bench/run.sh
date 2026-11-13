#!/usr/bin/env bash
# Benchmark zilch on fixed scenarios: each in a fresh copy of examples/ledger-demo, N runs.
#
#   bench/run.sh <label> [runs] [model]      e.g.  OPENCODE_API_KEY=… bench/run.sh before 3 beam-free
#
# Builds the release binary, runs every scenario `runs` times (default 2) with `--yes` (plans
# approved; commands run under the default approval), and writes bench/out/<label>/<scenario>-<n>.*
# (event log, output, seconds, test result). Then prints bench/report.sh <label>.
# Use free models for routine runs; results vary a lot between runs of the same model.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/.." && pwd)"
label=${1:?label}; runs=${2:-2}; model=${3:-beam-free}
scenarios=${SCENARIOS:-fix ask}
out="$here/out/$label"; rm -rf "$out"; mkdir -p "$out"
# ZILCH_BIN=<path> benchmarks another build (e.g. an older release) instead of building this tree.
if [ -n "${ZILCH_BIN:-}" ]; then bin=$ZILCH_BIN; else
  (cd "$root" && cargo build --release -q -p zilch)
  bin="$root/target/release/zilch"
fi
declare -A Q=(
  [fix]="Fix every issue in ISSUES.md. Investigate first, plan TODOs that run in parallel where possible, and finish only when ./run_tests.sh passes completely."
  [ask]="Which function converts amounts between currencies? Answer with the file and the function."
  # followup: the ask question, then this in the same session; only the follow-up is measured.
  [followup]="Which file was that in?"
)
for n in $(seq 1 "$runs"); do
  for sc in $scenarios; do
    ws=$(mktemp -d); cp -r "$root/examples/ledger-demo/." "$ws/"; rm -f "$ws/try.sh" "$ws/README.md"
    (cd "$ws" && git init -q && git add -A && git -c user.name=bench -c user.email=bench@localhost commit -qm base)
    t0=$(date +%s.%N)
    if [ "$sc" = followup ]; then
      # The first order sets up the session (not measured); the follow-up continues it with -c.
      (cd "$ws" && XDG_DATA_HOME="$ws/.data" timeout --foreground 900 "$bin" --model "$model" --yes --no-log \
        -p "${Q[ask]}" > /dev/null 2>&1) || true
      (cd "$ws" && XDG_DATA_HOME="$ws/.data" timeout --foreground 900 "$bin" --model "$model" --yes -c \
        --log "$out/$sc-$n.jsonl" -p "${Q[$sc]}" > "$out/$sc-$n.out" 2>&1) || true
    else
      (cd "$ws" && XDG_DATA_HOME="$ws/.data" timeout --foreground 900 "$bin" --model "$model" --yes --no-session \
        --log "$out/$sc-$n.jsonl" -p "${Q[$sc]}" > "$out/$sc-$n.out" 2>&1) || true
    fi
    awk "BEGIN {print $(date +%s.%N) - $t0}" > "$out/$sc-$n.time"
    (cd "$ws" && ./run_tests.sh 2>&1 | tail -1) > "$out/$sc-$n.tests" || true
    rm -rf "$ws"
  done
done
"$here/report.sh" "$label"
