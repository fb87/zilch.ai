# zilch

A minimal, extensible coding agent written in Rust from scratch, for any **OpenAI-compatible** API
(OpenAI, OpenRouter, Ollama, vLLM, llama.cpp, …).

- **One master.** It is the only agent you talk to and works like *plan mode* plus a team lead: it reads code, writes a TODO plan, asks you to approve it, then coordinates workers. It can never edit files or run commands.
- **Parallel workers.** Worker profiles such as `coder` run as several instances at once (`coder#1`, `coder#2`, …). A scheduler runs the TODO dependency graph in parallel waves. Each TODO declares the files it touches, so the scheduler keeps overlapping work apart, and stale-edit detection stops one worker from overwriting another's change.
- **Every agent has its own system prompt and model.** The prompt is a built-in default you can extend or replace, and each agent can use a different model and even a different provider.

```
 you ⇄ MASTER (read_file, glob, grep · todo_write, propose_plan, run_todos, delegate)
          │  TODO DAG → scheduler (max_workers, file-scope conflicts)
          ├── coder#1   read/write/edit/glob/grep/bash
          ├── coder#2   …in parallel
          └── explorer#1 read-only investigation
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

**REPL commands:** `/todos`, `/team`, `/models`, `/prompt <agent>`, `/usage`, `/reset <instance>`, `/clear`, `/exit`.
Ctrl-C cancels the current turn, including the master and every running worker.

## Configure

Config layers, later wins: built-in defaults < `~/.config/zilch/config.toml` < `./.zilch/config.toml` < env vars (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `ZILCH_MODEL`, `ZILCH_PROVIDER`) < CLI flags.

```toml
max_workers = 4
approval = "plan"

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

Startup stops with an error if a profile uses an unknown provider or its API key is missing.

## Layout

```
crates/zilch-core   library: no terminal I/O; frontends consume AgentEvents
  llm/              Chat Completions types, hand-written SSE parser, delta accumulator, provider registry
  tools/            Tool trait, worker tools (read/write/edit/glob/grep/bash), master tools (todo/plan/run/delegate)
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
