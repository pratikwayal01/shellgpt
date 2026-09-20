# Shell Autocomplete Tiny LLM

A ~2M parameter GPT trained from scratch to autocomplete Linux shell commands.
Runs on free Colab T4, Kaggle GPU, or even CPU.

## Files
- `shell_gpt.py` — everything: model, training, safety guard, REPL
- `colab_quickstart.py` — paste-ready Colab cells
- `shell_integration.sh` — Ctrl+G keybinding for bash/zsh

## Quick start (Colab)

1. Runtime → T4 GPU
2. Upload `shell_gpt.py`
3. Run cells from `colab_quickstart.py` in order

## Quick start (local)

```bash
pip install torch
python shell_gpt.py --mode train      # ~5 min on GPU, ~30 min CPU
python shell_gpt.py --mode complete   # interactive REPL
```

## Workflow

```
train → complete → add_cmd → finetune → complete (improved)
```

## Model size vs GPU

| n_layer | n_head | n_embd | Params  | Trains in     |
|---------|--------|--------|---------|---------------|
| 4       | 4      | 64     | ~0.5M   | 3 min T4      |
| 4       | 4      | 128    | ~2M     | 8 min T4  ← default |
| 6       | 6      | 192    | ~6M     | 20 min T4     |
| 8       | 8      | 256    | ~15M    | 45 min T4     |

## Safety layers

Every command passes through two checks:

1. **Show check** (`safety_check`) — should we even suggest this?
   - Hard-blocks patterns like `rm -rf /`, fork bombs, pipe-to-bash
   
2. **Execute check** (`execution_check`) — can we auto-run it?
   - `run`: safe read-only commands (ls, grep, cat…) — auto-runs with 10s timeout
   - `confirm`: destructive or state-changing — asks before running
   - `block`: always refused, never runs

## Adding your own commands

```bash
python shell_gpt.py --mode add_cmd
# paste your custom commands one per line

python shell_gpt.py --mode finetune
# model learns them in ~500 steps
```

Or directly edit / append to `custom_commands.txt`.

## How it learns new commands

Fine-tuning runs 500 gradient steps on your custom data with a 3× lower
learning rate than initial training. This updates weights without catastrophically
forgetting the built-in commands — standard "continued pre-training" approach.

## Shell integration (optional)

```bash
# Add to ~/.bashrc
source /path/to/shell_integration.sh
# Then press Ctrl+G on any partial command
```
