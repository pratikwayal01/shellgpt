#!/usr/bin/env python
"""One-off: shell_gpt.pt -> shellmate_model.npz + expect_logits.npy."""
import numpy as np
import torch
from shell_gpt import ShellGPT

ckpt = torch.load("shell_gpt.pt", map_location="cpu")
cfg = ckpt["cfg"]
sd = {k: v.numpy().astype(np.float32) for k, v in ckpt["model"].items()}
cfgm = {f"cfg_{k}": np.array(getattr(cfg, k)) for k in
        ("n_embd", "n_layer", "n_head", "block_size", "vocab_size")}
np.savez_compressed("shellmate_model.npz", **sd, **cfgm)
print("params:", sum(v.size for v in sd.values()))

model = ShellGPT(cfg).eval()
model.load_state_dict(ckpt["model"])
prefix = "docker p"
idx = torch.tensor([list(prefix.encode())][-cfg.block_size:])
logits, _ = model(idx)
np.save("expect_logits.npy", logits[-1].detach().numpy().astype(np.float32))
print("expect_logits.npy:", logits.shape)