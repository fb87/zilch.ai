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
out="$here/out/$label"; rm -rf "$out"; mkdir -p "$out"
(cd "$root" && cargo build --release -q -p zilch)
bin="$root/target/release/zilch"
declare -A Q=(
  [fix]="Fix every issue in ISSUES.md. Investigate first, plan TODOs that run in parallel where possible, and finish only when ./run_tests.sh passes completely."
  [ask]="Which function converts amounts between currencies, and how is a malformed CSV row reported to the user? Answer briefly."
)
for n in $(seq 1 "$runs"); do
  for sc in fix ask; do
    ws=$(mktemp -d); cp -r "$root/examples/ledger-demo/." "$ws/"; rm -f "$ws/try.sh" "$ws/README.md"
    (cd "$ws" && git init -q && git add -A && git -c user.name=bench -c user.email=bench@localhost commit -qm base)
    t0=$(date +%s.%N)
    (cd "$ws" && XDG_DATA_HOME="$ws/.data" timeout --foreground 900 "$bin" --model "$model" --yes --no-session \
      --log "$out/$sc-$n.jsonl" -p "${Q[$sc]}" > "$out/$sc-$n.out" 2>&1) || true
    awk "BEGIN {print $(date +%s.%N) - $t0}" > "$out/$sc-$n.time"
    (cd "$ws" && ./run_tests.sh 2>&1 | tail -1) > "$out/$sc-$n.tests" || true
    rm -rf "$ws"
  done
done
"$here/report.sh" "$label"
