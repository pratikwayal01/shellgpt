"""
Shell Autocomplete Tiny LLM
============================
Train a small GPT on shell command data, autocomplete partial commands,
optionally execute them safely, and fine-tune on new commands you add.

Usage:
  python shell_gpt.py --mode train      # build dataset and train
  python shell_gpt.py --mode complete   # interactive autocomplete REPL
  python shell_gpt.py --mode add_cmd    # add new commands to learn from
  python shell_gpt.py --mode finetune   # fine-tune on newly added commands

Colab: set COLAB=True below, or just run each section cell by cell.
"""

import os, json, math, time, re, subprocess, shlex, argparse, readline
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# ─────────────────────────────────────────────
#  CONFIG  — tune these to fit your GPU/CPU
# ─────────────────────────────────────────────
@dataclass
class Config:
    # Model size  (~2M params with defaults — trains in <10 min on Colab T4)
    vocab_size:   int   = 256       # byte-level: no tokenizer needed
    block_size:   int   = 128       # max context length (characters)
    n_layer:      int   = 4
    n_head:       int   = 4
    n_embd:       int   = 128
    dropout:      float = 0.1

    # Training
    batch_size:   int   = 64
    max_iters:    int   = 3000
    eval_interval:int   = 300
    lr:           float = 3e-4
    device:       str   = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    data_file:    str   = "shell_corpus.txt"
    model_file:   str   = "shell_gpt.pt"
    custom_file:  str   = "custom_commands.txt"

# Checkpoints carry a Config; allowlist it so torch.load (default
# weights_only=True since 2.6) still rejects arbitrary globals.
torch.serialization.add_safe_globals([Config])

cfg = Config()

# ─────────────────────────────────────────────
#  DATASET — shell command corpus
# ─────────────────────────────────────────────
# A solid built-in corpus of common Linux commands.
# Each line is one "command" the model learns from.
BUILTIN_CORPUS = """
git status
git log --oneline -20
git diff HEAD
git add -A && git commit -m "update"
git push origin main
git pull origin main
git checkout -b feature/new-branch
git stash && git stash pop
git rebase -i HEAD~3
find . -name "*.py" -type f
find . -mtime -7 -type f
find /var/log -name "*.log" -size +10M
grep -rn "TODO" . --include="*.py"
grep -r "error" /var/log/syslog | tail -50
grep -E "^[0-9]+" file.txt
ls -lah --sort=size
ls -ltr /tmp/
du -sh * | sort -h
df -h
ps aux | grep python
ps aux | sort -k3 -n | tail -20
kill -9 $(pgrep python)
top -bn1 | head -20
htop -d 5
cat /proc/cpuinfo | grep "model name"
cat /etc/os-release
uname -a
hostname -I
ip addr show
curl -s https://ifconfig.me
wget -q -O - https://example.com/file.tar.gz | tar xz
tar -czf archive.tar.gz ./directory
tar -xzf archive.tar.gz -C /tmp/
unzip file.zip -d ./output/
zip -r archive.zip ./directory/
ssh user@192.168.1.1 -p 22
scp -r ./local/ user@host:/remote/path/
rsync -avz --progress ./src/ user@host:/dst/
chmod +x script.sh
chmod 644 file.txt
chown -R user:group /path/
sudo apt update && sudo apt upgrade -y
sudo apt install -y vim curl wget git
sudo systemctl status nginx
sudo systemctl restart nginx
sudo journalctl -u nginx -f
docker ps -a
docker images
docker run -it --rm ubuntu:22.04 bash
docker build -t myapp:latest .
docker-compose up -d
docker logs -f container_name
docker exec -it container_name bash
kubectl get pods -n default
kubectl describe pod my-pod
kubectl logs -f deployment/my-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.txt
pip list --outdated
python3 -c "import sys; print(sys.version)"
pytest tests/ -v --tb=short
black . && isort .
flake8 . --max-line-length=88
mypy src/ --ignore-missing-imports
cat file.txt | wc -l
cat file.txt | sort | uniq -c | sort -rn
head -100 large_file.txt
tail -f /var/log/syslog
sed -i 's/old/new/g' file.txt
awk '{print $1, $3}' file.txt
cut -d',' -f1,3 data.csv
sort -t',' -k2 -n data.csv
echo "hello world" | tr 'a-z' 'A-Z'
env | grep PATH
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
history | grep docker
!234
watch -n 2 df -h
crontab -e
nohup python3 server.py &
jobs -l
fg %1
bg %1
lsof -i :8080
netstat -tulpn | grep 8080
ss -tulpn | grep LISTEN
iptables -L -n -v
ping -c 4 google.com
traceroute google.com
nmap -p 22,80,443 192.168.1.0/24
openssl req -newkey rsa:2048 -x509 -days 365 -out cert.pem -keyout key.pem
md5sum file.txt
sha256sum file.txt
diff file1.txt file2.txt
patch < patchfile.diff
strace -p 1234
ltrace ./binary
gdb ./binary core
valgrind --leak-check=full ./binary
make && make install
cmake .. -DCMAKE_BUILD_TYPE=Release
gcc -O2 -o out main.c
g++ -std=c++17 -o out main.cpp
nvcc -o out main.cu
nvidia-smi
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
free -h
vmstat 1 5
iostat -xz 1 5
sar -u 1 5
dmesg | tail -50
lsblk
fdisk -l
mount | column -t
""".strip()

def build_corpus():
    """Merge built-in commands with any custom commands the user has added."""
    corpus = BUILTIN_CORPUS
    custom = Path(cfg.custom_file)
    if custom.exists():
        extra = custom.read_text().strip()
        if extra:
            corpus = corpus + "\n" + extra
    Path(cfg.data_file).write_text(corpus)
    print(f"Corpus: {len(corpus)} chars, {corpus.count(chr(10))+1} lines")
    return corpus

# ─────────────────────────────────────────────
#  TOKENIZER — byte-level (char codes 0-255)
# ─────────────────────────────────────────────
def encode(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))

def decode(tokens: list[int]) -> str:
    return bytes([max(0, min(255, t)) for t in tokens]).decode("utf-8", errors="replace")

# ─────────────────────────────────────────────
#  MODEL — tiny GPT (same design as nanoGPT)
# ─────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.c_attn  = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.c_proj  = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.drop    = nn.Dropout(c.dropout)
        self.n_head  = c.n_head
        self.n_embd  = c.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(c.block_size, c.block_size))
                                          .view(1, 1, c.block_size, c.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc   = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
        self.drop = nn.Dropout(c.dropout)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))

class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.attn = CausalSelfAttention(c)
        self.mlp  = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ShellGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop    = nn.Dropout(c.dropout)
        self.blocks  = nn.Sequential(*[Block(c) for _ in range(c.n_layer)])
        self.ln_f    = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        # weight tying
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def get_batch(data: torch.Tensor, c: Config):
    ix = torch.randint(len(data) - c.block_size, (c.batch_size,))
    x  = torch.stack([data[i     : i + c.block_size] for i in ix])
    y  = torch.stack([data[i + 1 : i + c.block_size + 1] for i in ix])
    return x.to(c.device), y.to(c.device)

@torch.no_grad()
def estimate_loss(model, data, c: Config, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, c)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def train(extra_data: Optional[str] = None):
    corpus = build_corpus()
    if extra_data:
        corpus = corpus + "\n" + extra_data

    data = torch.tensor(encode(corpus), dtype=torch.long)
    n    = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = ShellGPT(cfg).to(cfg.device)
    print(f"Model params: {model.num_params():,}  |  device: {cfg.device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    t0 = time.time()
    for it in range(cfg.max_iters):
        if it % cfg.eval_interval == 0:
            vl = estimate_loss(model, val_data, cfg)
            elapsed = time.time() - t0
            print(f"iter {it:4d} | val loss {vl:.4f} | {elapsed:.0f}s elapsed")

        x, y = get_batch(train_data, cfg)
        with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.save({"model": model.state_dict(), "cfg": cfg}, cfg.model_file)
    print(f"\nSaved to {cfg.model_file}")
    return model

# ─────────────────────────────────────────────
#  GENERATION — top-k sampling with temperature
# ─────────────────────────────────────────────
@torch.no_grad()
def generate(model: ShellGPT, prefix: str, max_new: int = 80,
             temperature: float = 0.7, top_k: int = 10) -> str:
    model.eval()
    tokens = encode(prefix)[-cfg.block_size:]
    idx    = torch.tensor([tokens], dtype=torch.long, device=cfg.device)
    result = []
    for _ in range(max_new):
        idx_cond = idx[:, -cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        ch = decode([next_tok.item()])
        result.append(ch)
        if ch == "\n":
            break
        idx = torch.cat([idx, next_tok], dim=1)
    return "".join(result).rstrip("\n")

def get_completions(model: ShellGPT, prefix: str, n: int = 5) -> list[str]:
    """Return n diverse completions for a partial command."""
    seen, results = set(), []
    for _ in range(n * 3):          # oversample then deduplicate
        c = generate(model, prefix)
        full = prefix + c
        key  = full.strip()
        if key not in seen:
            seen.add(key)
            results.append(full.strip())
        if len(results) >= n:
            break
    return results

# ─────────────────────────────────────────────
#  SAFETY GUARD  — the most important part
# ─────────────────────────────────────────────
# Commands that should NEVER auto-execute (always require manual confirmation)
ALWAYS_CONFIRM = {
    "rm", "rmdir", "shred", "dd", "mkfs", "fdisk", "parted",
    "chmod", "chown", "chgrp",
    "mv", "cp",
    "kill", "killall", "pkill",
    "shutdown", "reboot", "poweroff", "halt",
    "sudo", "su", "doas",
    "passwd", "usermod", "useradd", "userdel",
    "crontab",
    "iptables", "ufw", "firewall-cmd",
    "curl", "wget",         # could download and pipe to shell
    "pip", "pip3",          # package installs
    "npm", "yarn", "cargo",
    "apt", "apt-get", "yum", "dnf", "pacman",
    "git",                  # could push/pull/reset
    "docker", "kubectl",
    "ssh", "scp", "rsync",
    "nc", "netcat",
    "python", "python3", "bash", "sh", "zsh", "perl", "ruby",  # arbitrary exec
}

# Patterns that are always hard-blocked, never executed
HARD_BLOCK_PATTERNS = [
    r"rm\s+-rf?\s+/",           # rm -rf / style
    r":\(\)\{.*\}",             # fork bomb
    r">\s*/dev/sd[a-z]",       # overwrite disk
    r"mkfs\.",                  # format disk
    r"dd\s+.*of=/dev/",        # write to raw device
    r"\|\s*bash",               # pipe to bash
    r"\|\s*sh\b",               # pipe to sh
    r"eval\s+\$\(",             # eval subshell
    r"base64.*\|.*bash",        # encoded payload
    r"wget.*\|\s*(bash|sh)",    # download and exec
    r"curl.*\|\s*(bash|sh)",    # download and exec
]

def safety_check(cmd: str) -> tuple[bool, str]:
    """
    Returns (is_safe_to_show, reason).
    Note: 'safe to show' just means we can suggest it as autocomplete.
    Execution is a separate stricter check.
    """
    stripped = cmd.strip()
    for pat in HARD_BLOCK_PATTERNS:
        if re.search(pat, stripped, re.IGNORECASE):
            return False, f"Hard-blocked pattern detected: {pat}"
    return True, "ok"

def execution_check(cmd: str) -> tuple[str, str]:
    """
    Returns (verdict, reason) where verdict is:
      'run'     — low-risk read-only command, auto-run with timeout
      'confirm' — potentially destructive, ask user first
      'block'   — hard-blocked, never run
    """
    stripped = cmd.strip()

    for pat in HARD_BLOCK_PATTERNS:
        if re.search(pat, stripped, re.IGNORECASE):
            return "block", f"Matches dangerous pattern: {pat}"

    try:
        parts = shlex.split(stripped)
    except ValueError:
        return "block", "Could not parse command safely"

    if not parts:
        return "block", "Empty command"

    base = os.path.basename(parts[0])

    if base in ALWAYS_CONFIRM:
        return "confirm", f"`{base}` can modify system state — confirm before running"

    # Read-only safe-ish commands can auto-run
    SAFE_READONLY = {"ls", "cat", "head", "tail", "grep", "find", "echo",
                     "pwd", "whoami", "date", "which", "type", "man",
                     "ps", "top", "df", "du", "free", "uname", "hostname",
                     "env", "printenv", "history", "wc", "sort", "uniq",
                     "diff", "file", "stat", "lsof", "ss", "netstat", "ip",
                     "ping", "dig", "nslookup", "host"}
    if base in SAFE_READONLY:
        return "run", "read-only command"

    # Default: ask
    return "confirm", f"Unknown command `{base}` — confirm before running"

def safe_execute(cmd: str, timeout: int = 10) -> str:
    """
    Execute a shell command with hard guardrails:
    - timeout enforced
    - no TTY / interactive
    - stdout/stderr captured
    - never runs as root escalation
    """
    verdict, reason = execution_check(cmd)

    if verdict == "block":
        return f"[BLOCKED] {reason}"

    if verdict == "confirm":
        print(f"\n  [SAFETY] {reason}")
        try:
            ans = input(f"  Run `{cmd[:60]}`? [y/N] ").strip().lower()
        except EOFError:  # non-interactive stdin → fail closed
            return "[Cancelled] (non-interactive)"
        if ans != "y":
            return "[Cancelled]"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,  # no interactive prompt hijack
            env={**os.environ, "SUDO_ASKPASS": "/bin/false"},  # block sudo -A
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0 and err:
            return f"[exit {result.returncode}] {err[:500]}"
        return out[:2000] if out else f"[done, exit {result.returncode}]"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s]"
    except Exception as e:
        return f"[ERROR] {e}"

# ─────────────────────────────────────────────
#  INTERACTIVE REPL
# ─────────────────────────────────────────────
def load_model() -> ShellGPT:
    if not Path(cfg.model_file).exists():
        print("No model found — training first...")
        return train()
    ckpt  = torch.load(cfg.model_file, map_location=cfg.device)
    model = ShellGPT(ckpt["cfg"]).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {cfg.model_file} ({model.num_params():,} params)")
    return model

def repl(model: ShellGPT):
    print("\nShell Autocomplete REPL")
    print("  Type a partial command → see completions")
    print("  Pick a number → optionally execute it")
    print("  Type 'quit' to exit\n")

    while True:
        try:
            prefix = input("partial> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prefix.lower() in ("quit", "exit", "q"):
            break
        if not prefix:
            continue

        completions = get_completions(model, prefix, n=5)

        # Hide completions that fail the show-check (PRD 3.2)
        safe = [c for c in completions if safety_check(c)[0]]
        if not safe:
            print("  (no safe completions)")
            continue

        print()
        for i, c in enumerate(safe, 1):
            print(f"  {i}. {c}")

        print()
        choice = input("Run which? (1-5 / Enter to skip) ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(safe):
                cmd = safe[idx]
                print(f"\n  $ {cmd}")
                output = safe_execute(cmd)
                if output:
                    print(output)
        print()

# ─────────────────────────────────────────────
#  ADD NEW COMMANDS & FINE-TUNE
# ─────────────────────────────────────────────
def add_commands():
    """Interactive wizard to add new commands to the custom corpus."""
    print("\nAdd custom commands (one per line, empty line to finish):")
    print("These will be learned during fine-tuning.\n")
    lines = []
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        lines.append(line)

    if not lines:
        print("Nothing added.")
        return

    with open(cfg.custom_file, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Added {len(lines)} commands to {cfg.custom_file}")
    print("Run with --mode finetune to update the model.")

def finetune():
    """Fine-tune the existing model on newly added custom commands."""
    custom = Path(cfg.custom_file)
    if not custom.exists() or not custom.read_text().strip():
        print("No custom commands found. Add some first with --mode add_cmd")
        return

    new_data = custom.read_text().strip()
    print(f"Fine-tuning on {len(new_data.splitlines())} custom commands...")

    # Load existing model and fine-tune with lower LR, fewer iters
    ckpt  = torch.load(cfg.model_file, map_location=cfg.device)
    model = ShellGPT(ckpt["cfg"]).to(cfg.device)
    model.load_state_dict(ckpt["model"])

    # Fine-tune on custom data only (fast)
    data = torch.tensor(encode(new_data), dtype=torch.long)
    if len(data) <= 1:
        print("Custom data too small to fine-tune (need 2+ chars)")
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.3)

    orig_iters = cfg.max_iters
    cfg.max_iters = 500

    model.train()
    for it in range(cfg.max_iters):
        if len(data) <= cfg.block_size:
            # Data too small for batch — just do a single gradient step
            x = data[:-1].unsqueeze(0).to(cfg.device)
            y = data[1:].unsqueeze(0).to(cfg.device)
            x = x[:, :cfg.block_size]
            y = y[:, :cfg.block_size]
        else:
            x, y = get_batch(data, cfg)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if it % 100 == 0:
            print(f"  iter {it} | loss {loss.item():.4f}")

    cfg.max_iters = orig_iters
    torch.save({"model": model.state_dict(), "cfg": cfg}, cfg.model_file)
    print(f"Fine-tuned and saved to {cfg.model_file}")

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shell autocomplete tiny LLM")
    parser.add_argument("--mode", choices=["train", "complete", "add_cmd", "finetune"],
                        default="complete")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "complete":
        model = load_model()
        repl(model)
    elif args.mode == "add_cmd":
        add_commands()
    elif args.mode == "finetune":
        finetune()
