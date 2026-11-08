-- zilch.nvim — talk to a running zilch REPL from Neovim.
--
-- While its REPL runs, zilch writes its socket endpoint to <workspace>/.zilch/ipc (`@name` = Linux
-- abstract socket, else a socket file). This plugin finds it by walking up from the current file
-- and sends newline-delimited JSON:
--   {"type":"send","text":…,"lane":…,"file":…,"start":…,"end":…,"selection":…}
--   {"type":"status"}
local M = {}

M.config = {
  -- Optional keymaps, e.g. { send = "<leader>zs", code = "<leader>zc", spec = "<leader>zp" }.
  keymaps = {},
}

local uv = vim.uv or vim.loop

local function notify(msg, level)
  vim.schedule(function()
    vim.notify("zilch: " .. msg, level or vim.log.levels.INFO)
  end)
end

local function is_socket(path)
  local st = path and uv.fs_stat(path)
  return st ~= nil and st.type == "socket"
end

--- Turn the pointer's endpoint into something libuv can connect to: `@name` is a Linux abstract
--- socket (no file; libuv wants a leading NUL byte), anything else is a socket file path.
local function resolve(endpoint)
  if endpoint:sub(1, 1) == "@" then
    return "\0" .. endpoint:sub(2)
  end
  return is_socket(endpoint) and endpoint or nil
end

--- The running zilch for the workspace at or above `start_dir`; returns the connectable endpoint
--- and the workspace root. zilch writes `.zilch/ipc` with its endpoint while the REPL runs.
function M.find_socket(start_dir)
  local dir = start_dir
  while dir and dir ~= "" do
    local f = io.open(dir .. "/.zilch/ipc", "r")
    if f then
      local endpoint = (f:read("*l") or ""):gsub("%s+$", "")
      f:close()
      local sock = endpoint ~= "" and resolve(endpoint)
      if sock then
        return sock, dir
      end
    end
    local parent = vim.fs.dirname(dir)
    if not parent or parent == dir then
      break
    end
    dir = parent
  end
  return nil, nil
end

--- Send one request; `on_reply(tbl)` gets the decoded JSON reply.
function M.request(sock, payload, on_reply)
  local pipe = uv.new_pipe(false)
  local abstract = sock:sub(1, 1) == "\0"
  if abstract and not pipe.connect2 then
    pipe:close()
    notify("this Neovim's libuv cannot reach abstract sockets; Neovim 0.10+ is required", vim.log.levels.ERROR)
    return
  end
  local function on_connect(err)
    if err then
      notify("cannot connect to " .. sock:gsub("%z", "@") .. " (" .. err .. ") — is the zilch REPL running?", vim.log.levels.ERROR)
      pipe:close()
      return
    end
    local buf = ""
    pipe:read_start(function(rerr, chunk)
      if rerr or not chunk then
        if not pipe:is_closing() then
          pipe:close()
        end
        return
      end
      buf = buf .. chunk
      local nl = buf:find("\n", 1, true)
      if nl then
        local line = buf:sub(1, nl - 1)
        pipe:read_stop()
        pipe:close()
        local ok, reply = pcall(vim.json.decode, line)
        if ok then
          on_reply(reply)
        else
          notify("bad reply: " .. line, vim.log.levels.ERROR)
        end
      end
    end)
    pipe:write(vim.json.encode(payload) .. "\n")
  end
  if abstract then
    -- connect2 passes the name with its length, so the leading NUL byte survives.
    pipe:connect2(sock, {}, on_connect)
  else
    pipe:connect(sock, on_connect)
  end
end

local function relative(path, root)
  if root and path:sub(1, #root + 1) == root .. "/" then
    return path:sub(#root + 2)
  end
  return path
end

--- Send lines line1..line2 of the current buffer plus a comment.
--- opts: { line1, line2, comment?, lane? ("code" | "spec" | nil = active lane) }
function M.send(opts)
  opts = opts or {}
  local file = vim.api.nvim_buf_get_name(0)
  local start_dir = file ~= "" and vim.fs.dirname(file) or uv.cwd()
  local sock, root = M.find_socket(start_dir)
  if not sock then
    sock, root = M.find_socket(uv.cwd())
  end
  if not sock then
    notify("no running zilch found (looked for .zilch/ipc above " .. start_dir .. ")", vim.log.levels.ERROR)
    return
  end
  local l1 = opts.line1 or vim.fn.line(".")
  local l2 = opts.line2 or l1
  local selection = table.concat(vim.api.nvim_buf_get_lines(0, l1 - 1, l2, false), "\n")

  local function go(comment)
    if not comment or comment == "" then
      notify("cancelled (no comment)", vim.log.levels.WARN)
      return
    end
    local payload = {
      type = "send",
      text = comment,
      lane = opts.lane,
      file = file ~= "" and relative(file, root) or nil,
      start = l1,
      ["end"] = l2,
      selection = selection,
    }
    M.request(sock, payload, function(reply)
      if reply.ok then
        notify("sent to the " .. reply.lane .. " lane")
      else
        notify(reply.error or "request failed", vim.log.levels.WARN)
      end
    end)
  end

  if opts.comment and opts.comment ~= "" then
    go(opts.comment)
  else
    local where = opts.lane and (opts.lane .. " lane") or "zilch"
    vim.ui.input({ prompt = where .. " ← comment: " }, go)
  end
end

--- Show what both lanes are doing and the spec documents' versions.
function M.status()
  local file = vim.api.nvim_buf_get_name(0)
  local sock = M.find_socket(file ~= "" and vim.fs.dirname(file) or uv.cwd()) or M.find_socket(uv.cwd())
  if not sock then
    notify("no running zilch found", vim.log.levels.ERROR)
    return
  end
  M.request(sock, { type = "status" }, function(r)
    notify(string.format("code: %s · spec: %s (active: %s)\n%s", r.code, r.spec, r.active, r.docs or ""))
  end)
end

function M.setup(opts)
  M.config = vim.tbl_deep_extend("force", M.config, opts or {})
  local km = M.config.keymaps
  local function map(lhs, command)
    if lhs then
      vim.keymap.set({ "n" }, lhs, ":" .. command .. "<CR>", { silent = true, desc = "zilch " .. command })
      vim.keymap.set({ "x" }, lhs, ":" .. command .. "<CR>", { silent = true, desc = "zilch " .. command })
    end
  end
  map(km.send, "ZilchSend")
  map(km.code, "ZilchCode")
  map(km.spec, "ZilchSpec")
end

return M
