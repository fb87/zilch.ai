#!/usr/bin/env bash
# Copy this demo into a fresh workspace and start zilch on it (free OpenCode Go models).
#   OPENCODE_API_KEY=oc_sk_... ./try.sh            # interactive: you approve the plan and shell commands
#   OPENCODE_API_KEY=oc_sk_... ./try.sh --yolo     # unattended
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
ws="${WORKSPACE:-$(mktemp -d /tmp/ledger-ws.XXXX)}"
cp -r "$here/." "$ws/" && rm -f "$ws/try.sh" "$ws/README.md"
(cd "$ws" && git init -q && git add -A && git -c user.name=zilch -c user.email=zilch@localhost commit -qm baseline)
[ -x "$root/target/release/zilch" ] || (cd "$root" && cargo build --release -q)
echo "workspace: $ws   (baseline committed; inspect changes later with: git -C $ws diff)"
cd "$ws"
exec "$root/target/release/zilch" -v "$@" -p "Fix every issue in ISSUES.md. Investigate first, plan TODOs that run in parallel where possible, and finish only when ./run_tests.sh passes completely."
