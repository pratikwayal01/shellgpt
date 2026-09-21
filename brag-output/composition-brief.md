# Hyperframes Composition Brief: shellGPT

## Objective
Create a short launch-style brag video for shellGPT — a 837,888-parameter GPT trained from scratch on tldr pages that autocompletes shell commands.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: ~20.2 seconds (5 scenes: 3.2 + 4.5 + 4.5 + 4.0 + 4.0)

## Source Material
- Project root: /home/pratik/code/shellGPT
- Primary files read: README.md, shell_gpt.py (CLI REPL), shellmate integration (shellgpt.py numpy inference)
- Product name: shellGPT
- Tagline / strongest claim: "The tiniest GPT for the terminal" — 837,888 params, trained in 110s on a free T4, ships as a 3.1MB numpy file
- Key UI or visual moment to recreate: terminal window with blinking cursor, `$ git bran` typed, `[gpt]` completion pop with numbered picks, `run? [y/N]` offer
- Copy that must appear verbatim:
  - "837,888" (parameter count)
  - "110s" / "free Colab T4" (training)
  - "3.1 MB" (numpy export)
  - "learned in 500 steps"
  - `github.com/pratikwayal01/shellgpt`

## Creative Direction
- Tone preset: default
- Creative direction: tiny model, no apologies — the spec sheet is the joke, delivered with a straight face
- Interpretation: playful, clean, postable; monospace terminal type; numbers land like facts; fast-in + hold, never flash-off
- Angle: the world races to a trillion parameters; this one has 837,888 and finishes the command anyway
- Hook: `$ git bran` typing into a terminal while the caption deadpans "Every LLM wants a trillion parameters."
- Outro / punchline: shellGPT — "The tiniest GPT for the terminal." + the GitHub URL
- Avoid:
  - Generic SaaS language ("streamline", "supercharge")
  - Abstract filler (generic gradient pulses, blob motion)
  - Non-terminal visual redesign — stay in the dark monospace world
  - Fake "AI magic" sparkles

## Visual Identity
- Background: #0d1117 (GitHub-dark)
- Text: #e6edf3 (terminal white)
- Accent: #3fb950 (terminal green — prompt `$`, `[gpt]` tag, success lines)
- Dim: #8b949e (captions, log noise)
- Display font: JetBrains Mono / ui-monospace
- Body font: JetBrains Mono / ui-monospace — whole video is monospace
- Visual references from the project: the shellGPT/shellmate CLI terminal look, `[gpt]` tier tag, `$` prompt, numbered pick list

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. The problem — 3.2s — terminal + `$ git bran` typing + "Every LLM wants a trillion parameters."
2. The tiny answer — 4.5s — `[gpt] git branch` beats in at 4.23s; chips `837,888 params / 110s training / 3.1 MB`
3. The proof — 4.5s — log lines "5760 iters | loss 1.40 | free Colab T4", "shellmate_model.npz 3.1M — no GPU, no server, no llama"; caption "The whole GPT fits in a file smaller than this video."
4. It learns yours — 4.0s — `--mode add_cmd` → `--mode finetune` → "learned in 500 steps" at 12.65s; Ctrl+G keycap
5. Outro — 4.0s — shellGPT logo, tagline, URL, fade

## Audio
- Audio role: warm upbeat bed + sparse professional terminal accents
- Audio arc: music enters at 0, brighter at the reveal, fades over final 1.2s; typing ticks → pop → chip ticks → one chime → soft logo hit
- Music: happy-beats-business-moves-vol-9-by-ende-dot-app.mp3 (already copied to composition/assets/music/)
- Music treatment: moderate volume, present under terminal SFX, fade out over last 1.2s, logo hit rings over the fade
- Music cue guidance: bundled preset (assets/music/cues/...vol-9...music-cues.json in the brag skill); ~114.84 BPM; strong cues 4.23 / 8.44 / 10.54 / 12.65 / 23.17; beat-lock the S2 completion reveal to 4.23 (within 0.15s); snap the 3 stat chips to consecutive beats ~4.75/5.28/5.80 (reveal all quickly, hold the set); treat other cues as optional hints
- Audio-reactive treatment: subtle — the `$` prompt / green accent may gain a faint glow on music energy; no waveforms, no strobing
- Audio-coupled moments:
  - S1 typed text — keyboard ticks per character
  - S2 completion reveal — one pop SFX, beat-locked chips tick
  - S4 green result — one subscription chime
  - S5 logo — one soft hit over the fade
- SFX selection guidance: use ui/keyboard/interface families from the brag skill's sfx library; sparse, dry, no whooshes/lasers; match motion exactly (sound + frame land together)
- SFX analysis guidance: skills/brag/assets/sfx/sfx-analysis.json — prefer low high-frequency-risk files; keep polish on repeated moments
- Exact SFX choice: choose filenames/timestamps/density/volume after the visuals exist
- Audio files: music already at composition/assets/music/

## Hyperframes Instructions
Load the Hyperframes domain skills (hyperframes-core, hyperframes-animation, hyperframes-creative, hyperframes-keyframes, hyperframes-cli) and build/update `brag-output/composition/` from this brief + the storyboard in brag-plan.md. /brag workflow — do not enter the hyperframes intent interview; do not use the generic promo workflow.

Requirements:
- Show at least one real UI element: the terminal window + `$ git bran` → `[gpt]` completion flow (scene 2 is the product in use)
- Keep all text readable in the final render (fast-in + hold; stair text to the 0.3s/word floor)
- Video within 15-25s (plan targets 20.2s)
- Include the music layer (file staged) + terminal SFX
- Beat-lock the scene-2 reveal to ~4.23s strong cue; snap chips to beats; mark `// beat-locked` / `// beat-grid`
- Audio-reactive: one subtle element (green prompt glow) or document extraction failure and skip — do not block render
- Run `hyperframes check` (the single gate) and fix all errors including WCAG contrast before render