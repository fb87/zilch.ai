# ledger-demo — a real task for the whole zilch team

A tiny Python expense tracker with **4 open issues** (`ISSUES.md`) across 4 modules, and a
15-test suite (13 failing) that defines "done". The issues are designed to exercise every role:

| Issue | File | Kind | Parallel? |
|---|---|---|---|
| 1 CSV parsing breaks on quoted fields | `ledger/parse.py` | bug | yes |
| 2 Currency conversion inverted | `ledger/currency.py` | bug | yes |
| 3 `monthly_totals` missing | `ledger/report.py` | feature | yes |
| 4 `--category` / `--monthly` CLI | `ledger/cli.py` | feature | after #3 |

Expected flow: master reads ISSUES.md + code (or delegates to explorers) → TODOs with disjoint
`files` and `depends_on` (#4 after #3) → plan approval → `coder` instances fix 1–3 in parallel,
then 4 → a worker runs `./run_tests.sh` → master verifies and reports.

`.zilch/config.toml` uses only free models (`beam-free`). Run:

```sh
export OPENCODE_API_KEY=oc_sk_...
./try.sh            # interactive approvals
./try.sh --yolo     # unattended
```
Success = `./run_tests.sh` prints `OK` in the workspace (and `git diff` shows `tests/` untouched).
The suite was validated against a reference solution (not included).
