# zilch

A minimal, extensible coding agent written in Rust from scratch, for any **OpenAI-compatible** API
(OpenAI, OpenRouter, Ollama, vLLM, llama.cpp, …).

- **One master.** It is the only agent you talk to and works like *plan mode* plus a team lead: it reads code, writes a TODO plan, asks you to approve it, then coordinates workers. It can never edit files or run commands.
- **Parallel workers.** Worker profiles such as `coder` run as several instances at once (`coder#1`, `coder#2`, …). A scheduler runs the TODO dependency graph in parallel waves. Each TODO declares the files it touches, so the scheduler keeps overlapping work apart, and stale-edit detection stops one worker from overwriting another's change.
- **Every agent has its own system prompt and model.** The prompt is a built-in default you can extend or replace, and each agent can use a different model and even a different provider.

```
 you ⇄ MASTER (read_file, glob, grep, git · todo_write, propose_plan, run_todos, delegate)
          │  TODO DAG → scheduler (max_workers, file-scope conflicts)
          ├── coder#1   read/write/edit/glob/grep/git/bash
          ├── coder#2   …in parallel
          └── explorer#1 read-only investigation
```

**Tools.** `glob` and `grep` run in-process on ripgrep's own crates (`.gitignore`-aware, no external
binaries). `git` is read-only (status, diff, log, show) so the master can check what workers changed
with a diff instead of re-reading files; repository config cannot make it run programs (external
diff drivers, textconv, fsmonitor and hooks are off). `edit_file` takes an `edits` list to make
several changes to a file in one call, all or none. Agents with `bash` are told which fast CLI
tools are installed (`rg`, `fd`, `jq`, `ast-grep`) and to prefer them over `grep -r` and `find`.

## Build

```sh
nix develop            # or any Rust toolchain
cargo build --release  # binary: target/release/zilch
cargo test
```

## Use

```sh
export OPENAI_API_KEY=...
zilch                                  # REPL
zilch -p "add a --version flag"        # one-shot; final answer on stdout
zilch --model o4-mini --worker-model coder=gpt-4.1-mini --workers 6
zilch --yolo                           # auto-approve plans and actions
OPENAI_BASE_URL=http://localhost:11434/v1 zilch --model qwen2.5-coder   # Ollama
```

How a change request flows:
1. The master explores the code.
2. It writes TODOs with `todo_write`.
3. It calls `propose_plan`, and you see the TODOs grouped into parallel waves and choose approve, give feedback, or reject.
4. It calls `run_todos`, and workers execute in parallel while a live panel shows progress.
5. The master verifies the results, adds fix TODOs if needed, and reports back.

**Approval modes** (`--approval` or `approval = …` in config):
- `plan` (default): approving the plan authorizes each TODO's writes to its declared files. `bash` and writes outside those files still ask. Prompts from parallel workers queue up one at a time while the other workers keep running.
- `ask`: ask before every mutating action.
- `auto`: never ask (same as `--yolo`).

**REPL commands:** `/todos`, `/team`, `/models`, `/prompt <agent>`, `/usage`, `/reset <instance>`, `/about`, `/redraw`, `/clear`, `/exit`.
Ctrl-C cancels the current turn, including the master and every running worker.

## Configure

Config layers, later wins: built-in defaults < central `~/.config/zilch/config.toml` < `./.zilch/config.toml` < env vars (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `ZILCH_MODEL`, `ZILCH_PROVIDER`) < CLI flags.

Put providers and models in the central file once and every workspace uses them; a workspace's
`.zilch/config.toml` only needs what differs, and is merged key by key on top. The central directory
is `$XDG_CONFIG_HOME/zilch` or `~/.config/zilch` (on macOS, `~/Library/Application Support/zilch` is read
too) and may also hold `agents/*.toml` and prompt files. `zilch --doctor` prints which files were loaded.
Sandboxed commands cannot read `~/.config/zilch` or the macOS directory, so with `chmod 600` it is a
reasonable place for `api_key = "…"` (a custom `$XDG_CONFIG_HOME` location is not hidden; add it to `bash_hide`).

```toml
max_workers = 4
approval = "plan"
markdown_indent = 2                # indent of rendered answers in the terminal (0 = off)

[providers.openai]                 # exists by default
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
max_concurrent = 8                 # per-provider request limit

[providers.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[providers.local]
base_url = "http://localhost:11434/v1"

[defaults]                         # fallback binding for every agent
provider = "openai"
model = "gpt-4.1"

[roles.master]
model = "o4-mini"
reasoning_effort = "high"

[roles.spec]                       # spec lane (/spec); without a model it uses the master's
model = "o3"

[roles.coder]
provider = "openrouter"
model = "qwen/qwen3-coder"
temperature = 0.2
system_prompt_file = "prompts/coder.md"   # relative to this .zilch dir
prompt_mode = "append"                    # append (default) | replace

[roles.explorer]
provider = "local"
model = "qwen2.5-coder:14b"
```

**Extra worker profiles** go in `.zilch/agents/<name>.toml`, and the master sees them in its team roster:

```toml
description = "Reviews diffs for bugs; never edits"
tools = ["read_file", "grep", "glob", "bash"]
model = "gpt-4.1"
system_prompt = "You are a meticulous code reviewer…"
```

**System prompts** are put together when an agent is spawned, in this order:
1. The built-in default (`crates/zilch-core/prompts/*.md`) and/or your prompt.
2. A role contract you can't remove: the master never edits and works through TODOs; workers stay in scope and end with a `STATUS:` report.
3. Environment details (cwd, OS, date, agent id, tools, team roster).
4. `AGENTS.md`, if present.

`{{cwd}}`, `{{date}}`, `{{agent_id}}`, `{{tools}}`, `{{workers}}` and `{{os}}` are filled in. A worker instance keeps its prompt and model for its whole lifetime. A TODO can set `model` (and `provider`) to send a hard item to a stronger model.

Agents: `master` (code lane), `spec` (spec lane), `coder` and `explorer` (workers), plus your own. Each
agent's binding comes from `[roles.<name>]`, then `[defaults]`; the `spec` agent first falls back to
`[roles.master]`, so one master setting (or `--model`) drives both lanes.

Startup stops with an error if a profile uses an unknown provider or its API key is missing, and
warns when an agent's model is not in its provider's `/models` list (`zilch --doctor` checks every
agent and sends each model a tiny request).

## Layout

```
crates/zilch-core   library: no terminal I/O; frontends consume AgentEvents
  llm/              Chat Completions types, hand-written SSE parser, delta accumulator, provider registry
  tools/            Tool trait, worker tools (read/write/edit/glob/grep/git/bash), master tools (todo/plan/run/delegate)
  agent/            agent loop (concurrent tool calls), approval gate, events
  team/             Team, WorkerPool, TODO list, DAG scheduler, file locks, plan review
  config.rs profile.rs prompt.rs
crates/zilch-cli    `zilch` binary: REPL, live panel, approval and plan-review prompts
```

**Extension points:**
- `LlmClient`: add a backend.
- `Tool`: add a tool (and its name to a profile's `tools`).
- `Approver` / `PlanReviewer`: a different UI.
- `EventSink`: a different renderer, such as a TUI or a server.

## Testing

```sh
cargo test                                   # offline: unit, integration, fuzz (proptest), corner cases
PROPTEST_CASES=10000 cargo test --test fuzz  # fuzz harder
# Live tests against a real FREE model (read-only tools, temp dirs only):
OPENCODE_API_KEY=... cargo test -p zilch-core --test live_corners -- --ignored --nocapture --test-threads=2
OPENCODE_API_KEY=... cargo test -p zilch-core --test live_compaction -- --ignored --nocapture
```
Fuzzed properties cover the SSE decoder/parser, tool-call reassembly, `edit_file` semantics,
output truncation, path normalization, file scopes, TODO graphs, and history compaction
(never splitting a tool call from its results).

## Safety & operations

| Concern | What zilch does | Config |
|---|---|---|
| File access | File tools are **sandboxed to the workspace**: `..`, absolute paths and symlinks that resolve outside it are refused; writes into `.git/` are refused | `sandbox = false` to disable |
| Secrets | `bash` runs **without secret env vars** (every provider `api_key_env`, plus `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, …) so a prompt-injected `env` can't leak keys | `bash_env_passthrough = ["GITHUB_TOKEN"]` |
| Arbitrary commands | `bash` runs in zilch's **built-in sandbox** — no external tools needed. Linux: native user/mount/PID/IPC/UTS(/net) namespaces implemented in Rust (the `zilch` binary re-executes itself as the sandbox helper); macOS: the OS's Seatbelt. Host filesystem read-only, only the workspace, a private `/tmp` and `/dev/shm` writable, credential stores hidden (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gh`, zilch config/logs, …), tool caches (`~/.cargo`, `~/.cache`, `~/.npm`, …) copy-on-write via overlayfs, `.git/` read-only, own PID namespace (the command is PID 1; everything inside dies with it), optional no-network (loopback only). `auto` falls back to unsandboxed with a visible warning when user namespaces are disabled; `required` refuses | `bash_sandbox = "auto" \| "required" \| "off"`, `bash_network = false`, `bash_hide`, `bash_writable`, `bash_overlay`, `bash_protect_git` |
| Runaway commands | each `bash` call runs in its own process group; on finish/timeout/cancel the **whole group is killed** (no orphaned servers) | `timeout_secs` per call (max 600) |
| Runaway cost | optional **team-wide token cap**; agents stop with a clear error when it's reached (soft cap: checked before each request) | `max_session_tokens = 2_000_000` / `--max-tokens` |
| Stuck agents | a code-only **supervisor** (no extra tokens) watches every agent: the same tool call with the same result 3× in a turn (which also catches read → failed edit cycles), the same reply 3×, 5 steps in a row of failing tool calls, or a reply that keeps repeating itself while it streams (cut on the spot). First time: a one-line nudge to change approach; next time: the turn stops (a worker's TODO fails, so the master re-plans). At team level, a TODO whose unchanged task already failed twice is not run again | `loop_limit = 3` (0 = off) |
| Hung providers | no headers or no stream data for `idle_timeout_secs` → error instead of hanging; 429/408/5xx retried with backoff, `Retry-After` honoured, 4xx not retried | `[providers.x] idle_timeout_secs = 180` |
| Context overflow | history compaction near each agent's `context_tokens` (see above) | `context_tokens = 128000` |
| Debugging | every session writes a **JSONL event log** (mode 0600) to `<data dir>/zilch/logs/`; its path is printed on errors | `--log FILE`, `--no-log` |
| Setup problems | `zilch --doctor` prints the resolved config and pings every distinct model | |
| Interrupts | Ctrl-C cancels the turn (all agents); twice within 2s force-quits; `-p` exits 130 when cancelled, 1 on error | |

## Sessions

Every turn is saved automatically (after success, failure or Ctrl-C) to
`<data dir>/zilch/sessions/<id>.json` (owner-only, atomic writes, outside the repo):
master history, TODO list and approvals, every worker instance with its history and model.

```sh
zilch -c                      # continue the latest session in this directory (REPL); shows a recap
zilch -c -p "now add tests"   # continue it non-interactively
zilch --sessions              # list sessions for this directory
zilch --resume <id>           # resume a specific one
zilch --no-session            # don't save
```
On resume, TODOs that were in progress go back to pending, a crash mid-turn is trimmed back to a
consistent point, and workers must re-read files before editing them (stale-edit protection is
never trusted across processes). `/clear` starts a new session; the old one stays resumable.

Resuming prints a short recap (no model call) so you can decide what to do next: the first and
last request, the last answer (or a warning if the last request was interrupted), TODO progress
with unfinished and failed items first, and the workers that kept their context. `/recap` shows it
again at any time.

## Protocols, costs and releases

**Protocols** — each provider speaks one of:
```toml
[providers.openai]     # default
protocol = "chat"      # POST {base}/chat/completions
[providers.oai-resp]
protocol = "responses" # POST {base}/responses   (OpenAI Responses API)
[providers.anthropic]
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"
protocol = "anthropic" # POST {base}/messages    (x-api-key auth)
```

**Costs** — at startup zilch reads each provider's `{base}/models` (OpenCode and OpenRouter
publish prices and context windows) so `/usage`, the REPL footer and `--doctor` show USD, and
`context_tokens` defaults to the model's real context window. Override or fill in prices per role
with `price_in` / `price_out` (USD per 1M tokens); `discover = false` on a provider skips the lookup.
`max_session_cost = 5.0` (or `--max-cost 5`) stops the team before a request would exceed the cap;
`max_session_tokens` likewise, both checked **before** sending.

**Sandbox on macOS** — `bash` runs under `sandbox-exec` with the same policy (writes limited to the
workspace, temp dirs and tool caches; secrets and the Keychain hidden; optional no-network). No
copy-on-write there, so caches are writable directly. `bash_sandbox = "required"` refuses to run
commands without an OS sandbox on either platform. `.git/` is read-only to commands by default
(`bash_protect_git = false` to let agents commit).

**Releases** — pushing a `v*` tag builds and tests Linux (gnu + static musl, x86_64 + aarch64) and
macOS (Intel + Apple Silicon) binaries and publishes them with `SHA256SUMS`:
```sh
curl -LO https://github.com/<owner>/zilch/releases/latest/download/zilch-vX.Y.Z-x86_64-unknown-linux-musl.tar.gz
```

## Live progress

While agents work, the CLI shows an animated status line per busy agent with what it is doing
and for how long, including the latest output line of long-running commands:
```
⠹ master · thinking · 4s
  ⠼ [t2] Add CLI flags · coder#2 · bash(cargo test) · 1m07s · Compiling serde v1.0.200
```
Without a terminal (e.g. `zilch -p … 2> log`), a plain `… coder#2 still running bash(cargo test) (1m00s)`
heartbeat is printed every 30s instead, so logs never look frozen.

## Spec lane and code lane

Two conversations run **in parallel** in the REPL; `/spec` and `/code` only switch which one you
are typing into — neither stops the other:

- **`/spec` — requirements engineering.** The spec master interviews you (a few focused questions
  per turn, with suggested answers), reads the code so it never asks what the repo already says,
  and maintains three documents in the spec directory (`spec_dir = "…"` or `--spec-dir`,
  default `specs/`):
  - `spec.md` — requirements with stable ids (FR-n, NFR-n) and EARS acceptance criteria;
  - `design.md` and `plan.md` — derived from a **released** spec version.
- **Versioning is enforced by zilch, not just the prompt:** `release_spec` freezes the draft as an
  immutable `versions/spec-vN.md` with a changelog entry; every write of design/plan is stamped
  `based-on: spec vN` and requires a released version.
- **`/code` — implementation**, as before. The code master works from `plan.md`/`design.md`
  (never `spec.md`); its context shows which spec version the plan is based on, warns when it is
  **stale**, and notes when the spec lane updated the plan mid-run.

While a lane works in the background its output is summarized (`✓ code lane finished — /code to
see its answer`); switching shows what it said. Approvals and plan reviews from either lane appear
above the prompt and your next line answers them. `/status` shows both lanes and the document
versions; `zilch -p "…" --lane spec` talks to the spec lane non-interactively. Both conversations
are saved in the session and resume with `-c`.

## Editor integration (Neovim)

`editor/nvim` is a small Neovim plugin: select code, `:ZilchSend why is this rounding here?`, and
the running zilch receives the file, line range, selected code and your comment in the active lane
(or `:ZilchCode` / `:ZilchSpec` to pick one). It talks to the REPL over a local socket (abstract on Linux, owner-only, peer-uid checked) using
newline-delimited JSON — see [`editor/nvim/README.md`](editor/nvim/README.md).
