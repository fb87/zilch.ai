# zilch.nvim

Send code to a running **zilch** session from Neovim: select lines, add a comment, and the agent
receives the file path, line range, the selected code and your comment — exactly as if you had
typed it at the zilch prompt.

## Install

lazy.nvim (from a local checkout of zilch):
```lua
{ dir = "/path/to/zilch-agent/editor/nvim", config = function()
    require("zilch").setup({ keymaps = { send = "<leader>zs", code = "<leader>zc", spec = "<leader>zp" } })
end }
```
Manual: `set runtimepath+=/path/to/zilch-agent/editor/nvim` (the commands load automatically;
`setup()` is only needed for keymaps).

## Use

Start `zilch` (the REPL) in your project, then in Neovim:

| Command | Does |
|---|---|
| `:ZilchSend [comment]` | send the current line, or a visual range, to the **active** lane |
| `:ZilchCode [comment]` | send to the **code** lane (plan + implement) |
| `:ZilchSpec [comment]` | send to the **spec** lane (requirements; e.g. "this behaviour should be a requirement") |
| `:ZilchStatus` | what both lanes are doing + spec document versions |

Without a comment argument you are prompted for one. Visual mode: select, then `:ZilchSend`
(`:'<,'>ZilchSend`). If the target lane is busy you get a notice instead of a queued message.

## How it connects

While its REPL runs, zilch listens on a Unix socket (owner-only) and writes its location to
`<workspace>/.zilch/ipc`; the plugin finds that file by walking up from the current file.
Messages are newline-delimited JSON:
```json
{"type":"send","text":"why?","lane":"code","file":"src/x.rs","start":10,"end":20,"selection":"..."}
{"type":"status"}
```
Any editor or script can use the same protocol (`socat - UNIX-CONNECT:$(cat .zilch/ipc)`).
