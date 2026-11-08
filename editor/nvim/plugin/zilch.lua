-- zilch.nvim: send code selections with comments to a running zilch session.
if vim.g.loaded_zilch then
  return
end
vim.g.loaded_zilch = true

local function cmd(name, lane, desc)
  vim.api.nvim_create_user_command(name, function(o)
    require("zilch").send({ line1 = o.line1, line2 = o.line2, comment = o.args, lane = lane })
  end, { range = true, nargs = "*", desc = desc })
end

cmd("ZilchSend", nil, "zilch: send the line/selection + a comment to the active lane")
cmd("ZilchCode", "code", "zilch: send the line/selection + a comment to the code lane")
cmd("ZilchSpec", "spec", "zilch: send the line/selection + a comment to the spec lane")
vim.api.nvim_create_user_command("ZilchStatus", function()
  require("zilch").status()
end, { desc = "zilch: show what both lanes are doing" })
