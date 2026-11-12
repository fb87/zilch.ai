# zilch

A coding agent written in Rust from scratch, for any **OpenAI-compatible** API (OpenAI,
OpenRouter, OpenCode, Ollama, vLLM, llama.cpp, …), built like a small operating system: the CLI is
the waiter, a kernel is the kitchen, agents are cooks. No agent framework and no async runtime —
plain `std` threads.

- **One main agent, many subagents.** You talk to one main agent that keeps the whole session's
  context and only dispatches: a question goes to an **answerer**, a change to a **planner**. You
  approve the planner's plan; the kernel starts one fresh **developer** per task and runs
  independent tasks in parallel. Each developer may write only the files its task owns.
- **A real kernel.** Agents are processes with a pid, run one step at a time on a pool of step
  workers, and reach the outside world only through syscalls (model calls and tool calls alike) run
  on driver pools. A timer tick drives a supervisor that nudges, restarts or stops looping and hung
  processes.
- **Two lanes.** `/spec` switches to a requirements engineer that maintains a versioned
  specification; `/code` back to the main agent, which builds from it. Both run in parallel.

```
 you ⇄ CLI (waiter) ──orders/answers──▶ KERNEL (kitchen) ──events──▶ CLI, JSONL log
                                          │ process table · scheduler + timer tick · supervisor
            ┌─────────────────────────────┼───────────────────────────────┐
     main agent (code lane)        spec agent (spec lane)        answerer · planner · developers
     dispatch only                 spec.md / design / plan       read · read+plan · owned paths
            └──── syscalls ──▶ driver pools: LLM (HTTP) · files/search/git · bash (sandbox)
```

## Build

```sh
nix develop            # or any Rust toolchain
cargo build --release  # binary: target/release/zilch
cargo test
```

## Use

```sh
export OPENAI_API_KEY=...
zilch                                  # REPL: type an order; y approves the plan
zilch -p "add a --version flag" --yes  # one order, then exit (0 = served)
zilch -c                               # continue the last session in this directory
zilch --model gpt-4.1 --developer-model gpt-4.1-mini --workers 8
zilch --doctor                         # the resolved setup, and a ping of every model
OPENAI_BASE_URL=http://localhost:11434/v1 zilch --model qwen2.5-coder   # Ollama
```

How an order flows:
1. An order the conversation already answers ("which file was that?", "thanks") gets a direct
   reply. Anything else the main agent dispatches with a self-contained brief: a question,
   explanation or check to the read-only **answerer** (its report comes back and the main agent
   replies; no plan, nothing to approve), a change to the **planner**.
2. The planner reads what the change touches (read-only **explorers** can investigate a large
   codebase in parallel) and calls `submit_plan`. Nobody writes planning documents into your
   project (approved plans are kept with the session, outside it).
3. You see the tasks grouped into parallel waves: `y` approves, `n` rejects, anything else is
   feedback. A stray Enter never approves.
4. The kernel runs the plan as a dependency graph: independent tasks at once on their own
   subagents, dependents after them, overlapping file scopes never at the same time. A failed task
   blocks its dependents.
5. The planner checks the result and reports; the main agent replies, which serves the order.
   Orders typed while it is busy are queued; a follow-up can refer to earlier orders.

**REPL commands**

| | |
|---|---|
| `/ps` | process table with a `top`-style header: calls in flight, busy/waiting processes, tokens per group |
| `/kill <pid>` · `/cancel` · Ctrl-C | stop one process · cancel the newest order (the main agent keeps its context) |
| `@3 text` · `@developer text` · `@main text` · `@spec text` | more input for a process or group, read before its next model call; nothing is cancelled |
| `/spec` · `/code` · `/status` | switch lanes · which lane you type into, what runs, the spec documents' state |
| `/usage` · `/models` · `/model main\|developer <model> [provider]` · `/ack` | tokens and USD · models in use · switch for this session (`default` switches back) · silence the high-cost warning |
| `/sessions` · `/recap` · `/clear` | saved sessions · this session's orders and last answer · a fresh session |
| `/prompt [main\|answerer\|planner\|developer\|spec]` · `/about` · `/redraw` · `/help` · `/exit` | |

Tab completes commands, their arguments (groups, model names from the providers' lists, live pids)
and `@` mentions. A status row above the prompt shows what each busy process is doing, a waiting
plan or command question, and the session's spend.

## Configure

Layers, later wins: `~/.config/zilch/config.toml` (or `$XDG_CONFIG_HOME/zilch`; on macOS also
`~/Library/Application Support/zilch`) < `./.zilch/config.toml` < env (`OPENAI_BASE_URL`,
`OPENAI_API_KEY`, `ZILCH_MODEL`, `ZILCH_PROVIDER`) < flags. A workspace file only needs what
differs. Sandboxed commands cannot read the zilch config directory, so with `chmod 600` it is a
reasonable place for `api_key = "…"`. `zilch --doctor` shows which files were loaded.

```toml
approval = "plan"                  # plan (default) | ask | auto
spec_dir = "specs"                 # the spec lane's documents
loop_limit = 3                     # supervisor: repeats before a nudge (0 = off)
max_session_cost = 5.0             # USD cap, checked before each model call
max_session_tokens = 2_000_000
repo_map = true                    # repository overview in the prompts of the agents that read code

[providers.openai]                 # exists by default
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
max_concurrent = 8                 # per-provider request limit

[providers.opencode]
base_url = "https://opencode.ai/inference/go/openai/v1"
api_key_env = "OPENCODE_API_KEY"
session_header = "x-opencode-session"

[defaults]                         # every agent unless its role says otherwise
provider = "openai"
model = "gpt-4.1"

[roles.master]                     # the main agent (also [roles.architect]); the spec agent follows it
model = "o4-mini"
reasoning_effort = "high"
context_tokens = 64000             # compaction starts at 75%

[roles.coder]                      # the subagents (also [roles.developer])
provider = "opencode"
model = "qwen3.8-flash"
reasoning_effort = "low"           # thinking models: fewer output tokens on small tasks
price_cached = 0.003               # USD / 1M cached input tokens, when the provider lists none
system_prompt = "Prefer small diffs."
prompt_mode = "append"             # append (default) | replace; or system_prompt_file = "…"

[roles.tester]                     # any other role: a custom subagent profile
description = "runs and writes tests"
tools = ["read_file", "glob", "grep", "bash"]

[quotas]                           # token caps per group, like cgroups
developer = 2_000_000
```

Custom profiles can also be files: `.zilch/agents/<name>.toml`. The planner sees the profiles
and gives a task to one with the task's `worker` field; the task then runs on that profile's model,
tools and prompt. Every agent's prompt ends with the environment (workspace, OS, date) and the
nearest `AGENTS.md`. Startup stops if a group's provider is unknown or its key is missing.

**Approval.** `plan`: you approve each plan; subagents' commands then run in the sandbox without
asking. `ask` (`--approval ask`): every command also waits for `y` / `n` / `a` (allow all for the
rest of the session). `auto` (`--yolo`, `--yes`): nothing asks. A lone `y`, `n` or `a` answers a
question even if it arrived while you were typing.

## The kernel

```
crates/zilch-ipc      pids, groups, messages, mailboxes (send never blocks), syscalls, kernel events,
                      the event bus (slow subscribers get Lagged) and the JSONL log
crates/zilch-kernel   process table, scheduler (priority queues, aging) driven by a timer tick, syscall
                      dispatch to driver pools, capabilities, plan DAG firing, lanes, supervisor, budgets
crates/zilch-drivers  LLM over blocking HTTP (and a scripted fake), file/search/git tools, sandboxed bash,
                      spec documents, sessions, config, price catalog
crates/zilch-agents   one LLM-driven program; the main agent, spec agent, subagents and explorers are recipes
crates/zilch          the `zilch` binary: REPL, status row, plan/command questions, editor link
```

- **Processes and syscalls.** One kernel thread owns every process. A step never does I/O: model
  calls and tool calls are syscalls that run on driver pools (commands on their own pool), several
  per step in parallel, so step workers never wait on the network. Long calls send heartbeats, so
  slow is not mistaken for stuck.
- **Capabilities.** The main agent only dispatches; the answerer, planner and explorers only read;
  the spec agent writes only its spec documents, a developer only its task's `owned_paths`; only
  developers run commands.
- **Supervisor** (code only, no extra model calls). Loops are compared after removing what changes
  from run to run (durations, timestamps, temp paths, addresses): the same call with the same result
  3×, the same reply 3×, 5 failing steps in a row, or a reply that keeps repeating itself while it
  streams (cut on the spot). First a one-line nudge; then a restart with a summary of what was tried
  (at most twice in 10 minutes); then a stop. A process with no progress is interrupted, then
  restarted or killed. Processes waiting on each other's mailboxes are woken instead of hanging.
- **Scheduling.** Tasks can carry a `priority` (0 urgent … 3 background); waiting work ages upward;
  per-process step and token budgets; `[quotas]` per group.
- **Context.** Subagents and explorers start fresh per task. A resident main agent compacts while
  idle: right after an order is served (past 40% of its budget) and again once the provider has
  likely dropped its cache (5 min, past 20%), into a digest of the finished orders; the digest
  request reuses its cached prefix. At 75% during an order, old tool output is shortened first,
  then older turns are summarized.
- **Events.** Everything the kernel does is an event; the JSONL log (`<data dir>/zilch/logs/`, mode
  0600) records every syscall result, which is what a replay needs.

## Finding code, spending fewer tokens

- **Repository map.** A module overview (every source directory with its purpose, each documented
  file) goes into the prompts of the agents that read code. `repo_map` lists a directory's or
  file's definitions with line numbers; `find_definition` finds where a function or type is
  defined. Extracted from the code itself (doc comments, READMEs, manifests, per-language
  patterns), no model calls; cached in the data directory for large trees and refreshed where it
  changed.
- **Search and reads.** `grep` searches in parallel like ripgrep. Re-reading unchanged lines, or an
  identical `grep` / `glob` with the same result, returns a short note instead of the text again;
  a file over 1,500 lines read without a range returns its outline first.
- **Less resent.** Command output loses build progress and passing-test lines; reasoning from
  earlier orders is not sent again; every developer instance shares one prompt, so the provider's
  cache serves them all; Anthropic requests carry cache markers (it caches nothing without them).

## Outside sources (off by default)

With `--web` or `[web] enabled = true`, every agent can look outside the workspace:

- `fetch_source`: a read-only copy of a git repository (https, ssh or `git@host:owner/repo`; a
  branch, tag or full commit) or a source archive (`.tar.gz`, `.tgz`, `.zip`, `.crate`, GitHub
  `/archive/` links), kept in `<data dir>/zilch/sources`, never in your project, and reused.
  `read_file`, `grep`, `glob`, `repo_map` and `find_definition` read it; nothing writes to it.
  Archives are unpacked by zilch itself (no links, no `..`, size caps); clones are shallow and keep
  no `.git`.
- `web_fetch`: one page as plain text (links kept as `[text](url)`), 20k characters per call.
- `web_search`: only with a search service.

```toml
[web]
enabled = true
search = "searxng"                  # or "brave" (key in BRAVE_API_KEY, or search_api_key_env)
search_url = "http://localhost:8888" # a SearXNG instance with the JSON format enabled
```

URLs the models supply must resolve to public addresses (no localhost, private networks or cloud
metadata), also after redirects. Pages, search results and fetched code are marked as untrusted:
they can carry instructions aimed at the model, so keep plan approval on when you use this.

## Lanes

`/spec` switches to the spec lane's resident agent, a requirements engineer that interviews you (a
few focused questions per turn, with suggested answers) and maintains `spec.md` (requirements with
stable ids and EARS acceptance criteria), `design.md` and `plan.md` in `spec_dir`. Versioning is
enforced by zilch: `release_spec` freezes the draft as an immutable `versions/spec-vN.md` with a
changelog entry, and design and plan are stamped with the spec version they derive from. `/code`
switches back; the planner works from the spec lane's design and plan. Both lanes run in
parallel; spec orders are marked in the output.

## Sessions

After every served order the resident agents' conversations are saved to
`<data dir>/zilch/os-sessions/<workspace>/<id>.json` (owner-only, atomic writes, outside the repo).

```sh
zilch -c                      # continue the latest session in this directory
zilch -c -p "now add tests"   # … non-interactively
zilch --sessions              # list this directory's sessions
zilch --resume <id>           # continue a specific one
zilch --no-session            # don't save
```

Order numbers carry on in a continued session. Work on an order that was not served yet is not
saved. `/clear` starts a fresh session; the old one stays resumable.

## Daemon (optional)

Plain `zilch` runs in its own process, as before. For sessions that outlive the terminal, one
daemon per user hosts any number of them, each with its own workspace, config, kernel and costs:

```sh
zilch daemon start                 # in the background (log: <data dir>/zilch/logs/daemon.log)
zilch attach                       # REPL on this directory's session (created if missing)
zilch attach ledger-1              # ... or a named one; /detach leaves it running, /stop ends it
zilch new --workspace ~/src/app --name app --web   # a session, from anywhere
zilch -p "run the tests" --session app             # one order; prints its progress, exits when served
zilch ls                           # id, state (idle, working, waiting for you), spend, clients
zilch stop app                     # end a session (its conversation stays resumable with -c)
zilch daemon stop                  # end them all
```

Attaching replays the session's recent output, then follows it live; any number of clients can
attach at once. Orders keep running with nobody attached; a plan approval or command question
then waits for the next client (`zilch ls` shows "waiting for you"). Nothing is approved for you
unless the session was created with `--approval auto`; `on_question = "notify-send zilch
\"$ZILCH_QUESTION\""` in the config runs a command when a question waits unattended
(`ZILCH_WORKSPACE`, `ZILCH_TICKET`, `ZILCH_QUESTION` are set). Each session also serves the editor
link of its workspace.

The daemon listens on a local socket for this user only (Linux: an abstract socket; elsewhere a
socket file in a private directory; peer uid checked; no TCP: use `ssh -L` to reach it remotely).
`ZILCH_DAEMON_SOCKET` picks another endpoint (`@name` or a path). Provider keys come from the
daemon's environment: start it from a shell that has them, or for a systemd user unit put them in
an `EnvironmentFile` and run `zilch daemon run` (foreground).

## Costs and models

At startup zilch reads each provider's `{base}/models` (OpenCode and OpenRouter publish prices and
context windows, and OpenCode its cache price and peak-hour multipliers); `price_in` / `price_out` /
`price_cached` on a role override them, `discover = false` on a provider skips the lookup. Cached
input is billed at the cache price. `/usage` shows tokens (with the share served from the
provider's cache) and USD per agent and model, a session line and where the input went (system,
tools, history, each tool's output); the session line is also printed on exit and after `-p`. The
status row shows the session's spend; the caps stop a call before it would exceed them. `/model` switches a group for
the session. A model whose output price is at least `expensive_model_price` (default 10 USD/1M) and
that has been in use for `expensive_model_minutes` (30) or cost `expensive_model_spend` ($2)
pulses in the status row until `/ack` (next step) or until no group uses it.

**Protocols** — each provider speaks one of:
```toml
[providers.openai]
protocol = "chat"      # POST {base}/chat/completions (default)
[providers.oai-resp]
protocol = "responses" # POST {base}/responses
[providers.anthropic]
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"
protocol = "anthropic" # POST {base}/messages (x-api-key auth)
```

## Safety & operations

| Concern | What zilch does | Config |
|---|---|---|
| File access | File tools are confined to the workspace (`..`, absolute paths and symlinks resolving outside are refused; no writes into `.git/`), on top of the kernel's capabilities | `sandbox = false` to disable |
| Parallel edits | per-file locks; stale-edit detection refuses to overwrite a file changed since it was read | |
| Secrets | `bash` runs without secret env vars (every provider `api_key_env`, `*_API_KEY`, `*_TOKEN`, `*_SECRET`, …) | `bash_env_passthrough = ["GITHUB_TOKEN"]` |
| Commands | `bash` runs in zilch's built-in sandbox. Linux: native user/mount/PID/IPC/UTS(/net) namespaces in Rust (the `zilch` binary re-executes itself as the helper); macOS: Seatbelt. Read-only host, writable workspace and private `/tmp`, credential stores hidden, tool caches copy-on-write, `.git/` read-only, own PID namespace, optional no-network | `bash_sandbox = "auto" \| "required" \| "off"`, `bash_network`, `bash_hide`, `bash_writable`, `bash_protect_git` |
| Runaway commands | each call in its own process group, killed on finish, timeout or cancel | `timeout_secs` per call (max 600) |
| Runaway cost | session token and USD caps, group quotas, per-process budgets, the supervisor | see Configure |
| Hung providers | no data for `idle_timeout_secs` → error; 429/408/5xx retried with backoff, `Retry-After` honoured | `[providers.x] idle_timeout_secs = 180` |
| Debugging | JSONL event log per session; `--verbose` shows every model call and result | `--log FILE`, `--no-log` |
| Setup problems | `zilch --doctor` | |

## Editor integration (Neovim)

`editor/nvim` is a small Neovim plugin: select code, `:ZilchSend why is this rounding here?`, and
the running zilch receives the file, line range, selected code and your comment as an order in the
active lane (`:ZilchCode` / `:ZilchSpec` pick one; `:ZilchStatus` shows both lanes). It talks to the
REPL over a local socket — abstract on Linux, a private socket file elsewhere, peer-uid checked —
whose address zilch writes to `.zilch/ipc`. See [`editor/nvim/README.md`](editor/nvim/README.md).

## Testing and releases

```sh
cargo test     # offline: kernel behaviour with scripted agents, tools, protocols, end-to-end orders
               # with a scripted model on real files, sessions, the editor link
```

`bench/run.sh <label> [runs] [model]` measures what orders cost (requests, input, cached and
output tokens, time, tests passing) on fixed scenarios in a copy of `examples/ledger-demo`;
`bench/report.sh` compares labels. Compare before and after a change with several runs each.

Pushing a `v*` tag builds and tests Linux (gnu + static musl, x86_64 + aarch64) and macOS (Intel +
Apple Silicon) binaries and publishes them with `SHA256SUMS`.
