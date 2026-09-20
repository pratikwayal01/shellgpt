#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  Shell integration — adds Ctrl+G as autocomplete keybinding
#  Source this in your ~/.bashrc or ~/.zshrc:
#    source /path/to/shell_integration.sh
#
#  Requires: Python 3, shell_gpt.py, trained shell_gpt.pt
# ─────────────────────────────────────────────────────────────────

SHELLGPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHELLGPT_SCRIPT="$SHELLGPT_DIR/shell_gpt.py"
export PYTHONPATH="$SHELLGPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Quick one-shot completion helper (called by the keybinding)
_shellgpt_complete() {
    local partial="${READLINE_LINE}"
    if [ -z "$partial" ]; then
        return
    fi

    # Get completions from the model (top 5, one per line)
    local completions
    completions=$(python3 - "$partial" <<'PYEOF'
import sys
from shell_gpt import load_model, get_completions, cfg
model = load_model()
for c in get_completions(model, sys.argv[1], n=5):
    print(c)
PYEOF
    )

    if [ -z "$completions" ]; then
        echo "(no completions)"
        return
    fi

    echo ""
    echo "Completions for: $partial"
    echo "$completions" | nl -w2 -s". "
    echo ""
    read -p "Pick (1-5) or Enter to cancel: " choice

    if [[ "$choice" =~ ^[1-5]$ ]]; then
        local selected
        selected=$(echo "$completions" | sed -n "${choice}p")
        READLINE_LINE="$selected"
        READLINE_POINT=${#READLINE_LINE}
    fi
}

# Bind Ctrl+G to the autocomplete function (bash only)
if [ -n "$BASH_VERSION" ]; then
    bind -x '"\C-g": _shellgpt_complete'
    echo "ShellGPT: Ctrl+G bound to AI autocomplete"
fi

# Zsh version (add to ~/.zshrc manually):
# zle -N _shellgpt_complete_zsh
# _shellgpt_complete_zsh() { ... }
# bindkey '^G' _shellgpt_complete_zsh
