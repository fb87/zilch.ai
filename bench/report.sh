#!/usr/bin/env bash
# bench/report.sh <label> [other-label] — per run and median per scenario: model requests, input,
# cached and output tokens, seconds, test result (from the event logs bench/run.sh wrote).
here="$(cd "$(dirname "$0")" && pwd)"
# Only the Usage events: the log also carries each model call's usage inside its SyscallDone result.
sum() { grep '"type":"Usage"' "$2" | grep -o "\"$1\":[0-9]*" | cut -d: -f2 | awk '{s+=$1} END {print s+0}'; }
for label in "$@"; do
  out="$here/out/$label"; echo "== $label"
  printf "%-8s %8s %9s %9s %8s %6s  %s\n" run requests input cached output secs tests
  for f in "$out"/*.time; do r=$(basename "$f" .time); l="$out/$r.jsonl"
    printf "%-8s %8s %9s %9s %8s %6.0f  %s\n" "$r" "$(grep -c '"type":"Usage"' "$l")" "$(sum prompt_tokens "$l")" \
      "$(sum cached_tokens "$l")" "$(sum completion_tokens "$l")" "$(cat "$f")" "$(cat "$out/$r.tests")"
  done | tee "$out/runs.txt"
  for sc in fix ask; do
    grep "^$sc-" "$out/runs.txt" | awk -v sc="$sc" '{r[NR]=$2; i[NR]=$3; c[NR]=$4; o[NR]=$5; t[NR]=$6}
      function med(a, n,   s, k, j, x) { for (k=1;k<=n;k++) s[k]=a[k]; for (k=1;k<=n;k++) for (j=k+1;j<=n;j++) if (s[j]<s[k]) {x=s[k]; s[k]=s[j]; s[j]=x}
        return n%2 ? s[(n+1)/2] : (s[n/2]+s[n/2+1])/2 }
      END { if (NR) printf "%-8s %8.1f %9.0f %9.0f %8.0f %6.0f  (median of %d)\n", sc, med(r,NR), med(i,NR), med(c,NR), med(o,NR), med(t,NR), NR }'
  done
done
