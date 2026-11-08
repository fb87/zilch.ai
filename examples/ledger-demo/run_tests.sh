#!/usr/bin/env bash
# Runs the unittest suite. Uses python3 from PATH, or fetches it via nix if missing.
cd "$(dirname "$0")"
args=("$@"); [ ${#args[@]} -eq 0 ] && args=(discover -s tests -t .)
if command -v python3 >/dev/null; then
  exec python3 -m unittest "${args[@]}"
else
  exec nix shell nixpkgs#python3 -c python3 -m unittest "${args[@]}"
fi
