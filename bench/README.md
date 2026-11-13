# bench

Measures what an order costs, so token and speed changes can be judged by numbers, not by eye.

    OPENCODE_API_KEY=… bench/run.sh <label> [runs] [model]
    bench/report.sh <label> [<other label> …]

Scenarios (`SCENARIOS="ask followup"` picks some; default `fix ask`), each in a fresh copy of
`examples/ledger-demo`:

- `fix` — "fix every issue in ISSUES.md until ./run_tests.sh passes" (4 bugs in 4 files, 15 tests);
- `ask` — a code question that needs navigation;
- `followup` — the `ask` question, then "which file was that in?" in the same session
  (only the follow-up is measured: it can be answered from the conversation).

Reported per run and as the median per scenario: model requests, input tokens, cached input tokens,
output tokens, seconds, and the test suite's last line (`OK` for a correct fix).

`ZILCH_BIN=path/to/zilch` benchmarks another build, e.g. an older release, instead of this tree.

Run the same label before and after a change with several runs each: a model's results vary a lot
between runs, so single runs prove little. Use free models (e.g. `beam-free`) for routine checks.
