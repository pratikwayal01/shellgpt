# shellGPT

A tiny GPT trained **from scratch** to autocomplete Linux shell commands —
**837,888 parameters**, trained in ~2 minutes on a free Colab T4, serialized
to a **3.1 MB numpy file**. No tokenizer, no GPU needed at inference, no
server, no llama.

> The whole GPT fits in a file smaller than most screenshots.

## Highlights

- **Byte-level** — no tokenizer (vocab = 256 bytes), works on any command
- **837,888 params** — trains in ~110 s on a free T4 (loss 1.40), CPU-only fine
- **numpy inference** — `export_npz` produces a 3.1 MB `.npz`; the exported
  logits match PyTorch to within 4e-6
- **Local autocomplete tier** — the exported model powers the `[gpt]` tier in
  [shellmate](https://github.com/pratikwayal01/shellmate), giving you
  command completion for partial input with zero model server running
- **Safety layers** — every suggestion must pass a show check and an execute
  check before it can run
- **Learns your commands** — `add_cmd` + `finetune` teaches it your own commands
  in ~500 steps

## Files

| File                   | Purpose                                        |
|------------------------|------------------------------------------------|
| `shell_gpt.py`         | everything: model, training, REPL, safety, export |
| `export_npz.py`        | torch checkpoint → numpy weights (`shellmate_model.npz`) |
| `fetch_tldr.py`        | downloads the tldr-pages corpus used for training |
| `colab_quickstart.py`  | paste-ready Colab cells (train on free T4)      |
| `shell_integration.sh` | Ctrl+G keybinding for bash/zsh                  |
| `shellmate_model.npz`  | shipped exported weights (3.1 MB)               |

## Quick start (Colab)

1. Runtime → T4 GPU
2. Upload `shell_gpt.py`
3. Run the cells from `colab_quickstart.py` in order
4. Download `shell_gpt.pt` and export numpy weights locally:
   `python export_npz.py shell_gpt.pt`

## Quick start (local)

```bash
pip install torch
python shell_gpt.py --mode train      # ~2 min on T4, ~30 min CPU
python shell_gpt.py --mode complete   # interactive REPL
python shell_gpt.py --mode export_npz # → shellmate_model.npz (3.1 MB)
```

## Workflow

```
train → complete → add_cmd → finetune → complete (improved)
```

## Model size vs training time

| n_layer | n_head | n_embd | Params   | Trains in  |
|---------|--------|--------|----------|------------|
| 4       | 4      | 64     | ~0.22M   | ~1 min T4  |
| 4       | 4      | 128    | 837,888  | ~2 min T4 ← default |
| 6       | 6      | 192    | ~2.7M    | ~8 min T4  |
| 8       | 8      | 256    | ~6.4M    | ~20 min T4 |

## Safety layers

Every command passes through two checks:

1. **Show check** (`safety_check`) — should we even suggest this?
   Hard-blocks `rm -rf /`, fork bombs, pipe-to-bash, etc.

2. **Execute check** (`execution_check`) — can we auto-run it?
   - `run` — safe read-only commands (ls, grep, cat…) auto-run with a 10 s timeout
   - `confirm` — destructive or state-changing commands ask before running
   - `block` — always refused, never runs

## Adding your own commands

```bash
python shell_gpt.py --mode add_cmd    # paste your custom commands, one per line
python shell_gpt.py --mode finetune   # model learns them in ~500 steps
```

Fine-tuning runs 500 gradient steps at a 3× lower learning rate, so the model
updates on your commands without catastrophically forgetting the built-ins
(standard continued pre-training).

## Shell integration (optional)

```bash
# Add to ~/.bashrc
source /path/to/shell_integration.sh
# Press Ctrl+G on any partial command
```

## Use it in shellmate

[shellmate](https://github.com/pratikwayal01/shellmate) ships
`shellmate_model.npz` and falls back to this model (tier `[gpt]`) for partial
commands, before the local llama-server tier:

```
$ git bran
[gpt] git branch
```

Numpy-only inference — `pip install numpy` is the only dependency.