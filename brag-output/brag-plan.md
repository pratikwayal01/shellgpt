# Brag Plan: shellGPT

## What is this app?
shellGPT is a GPT trained entirely from scratch on tldr pages + shell builtins — only **837,888 parameters** — that autocompletes partial shell commands in the terminal, offers numbered picks, and asks before running.

## The angle
The world is racing to trillion-parameter models. This one has 837,888 — trained in 110 seconds on someone's free T4, now exported to a 3.1MB numpy file so it completes commands with zero GPU, zero server, zero llama. The absurdity is the spec sheet. Play the numbers dead straight; they're funny on their own.

## Hook (first 2-3 seconds)
Terminal window materializes, cursor blinks, `$ git bran` types in char-by-char with key ticks. Caption slides under, dryly: "Every LLM wants a trillion parameters." The terminal just sits there. Setup of the absurdity.

## Key moments (the middle)
- **The reveal, ON the beat (4.23s):** `[gpt] git branch` completes instantly — `1. git branch` pick line. Whole GPT answered in ~2ms and it shows.
- **The spec chips:** `837,888 params` · `110s training` · `3.1 MB` pop one by one, ticking with the music.
- **The proof line:** training log `5760 iters | loss 1.40 | free T4` then the flex: `shellmate_model.npz 3.1M` → "no GPU, no server, no llama."
- **The learning loop:** `--mode add_cmd` → `--mode finetune` → "learns your commands in 500 steps."

## Outro / punchline
Logo: **shellGPT**. Tagline: "The tiniest GPT for the terminal." URL: `github.com/pratikwayal01/shellgpt`. One soft logo hit, music fades.

## User flow worth showing
Entry → key action → result (the 3 beats, all real):
1. Type a partial command: `$ git bran` (real shellmate smoke test — routes to the gpt tier)
2. The completion appears instantly: `[gpt] ...` pick `1.` (real output shape)
3. The run? offer (`run? [y/N]`) — the safe-execution promise (real maybe_run flow)

## Tone
- Preset: default
- Creative direction: tiny model, no apologies — the spec sheet is the joke, delivered with a straight face
- Interpretation: playful, clean, postable; monospace terminal type; numbers land like facts, not boasts; fast-in + hold, never flash-off

## Format: landscape — 1920x1080
## Duration: ~20s

## Visual identity (from the project)
- Background: #0d1117 (GitHub-dark terminal)
- Accent: #3fb950 (terminal green — prompt, tags, successful text)
- Text: #e6edf3 (terminal white)
- Dim/muted: #8b949e (captions, log noise)
- Display font: JetBrains Mono / ui-monospace (project is a terminal)
- Body font: same — everything in monospace, that's the identity
- Strongest visual element: the terminal window with the blinking cursor and the `[gpt]` completion pop

## Share copy (draft)
Made shellGPT — a GPT with 837,888 parameters that autocompletes shell commands. Trained in 110s on a free Colab T4; ships as one 3.1MB file. It learns your commands in 500 steps.

## Audio direction
- Role: warm upbeat bed, professional restraint, key ticks on typing
- Music: happy-beats-business-moves-vol-9-by-ende-dot-app.mp3 (bundled); enter at 0, fade out over final 1.2s
- Music treatment: volume moderate, present under terminal SFX; final logo hit rings over a quick fade
- Music cue guidance: bundled preset for vol-9, ~114.84 BPM; strong cues: 4.23 (S2 reveal — beat-lock), 8.44 / 10.54 (S3 stat chips), 12.65 (S4 add_cmd) — lock 1-3, treat rest as optional hints
- Audio-reactive treatment: subtle; slight glow/brightness of the terminal prompt responding to music energy
- SFX posture: sparse professional — keyboard ticks for typing, one pop for the completion reveal, tick sounds on stat chips, one soft logo hit; no cutesy sounds
- Audio-coupled moments: typed text (key ticks), beat-revealed stat chips, completion pop
- Restraint rule: no robot voices, no whooshes, no laser sounds; the terminal is a quiet place

## Storyboard

### Scene 1 — The problem — 3.2s
Terminal window (traffic dots, "shellmate — git bran") center-dark. `$ git bran` types in char-by-char (~1.0s, key ticks). Caption below slams in and holds: "Every LLM wants a trillion parameters."
Sequential/interaction: yes — typed text, simulated keystroke by keystroke
Audio intent: dry, building; music enters at 0
Audio-coupled idea: type ticks per character
Music: upbeat, low-mid volume
Transition mood: hard → Scene 2 (land the completion on the beat)

### Scene 2 — The tiny answer — 4.5s
Typing stops. ON strong cue 4.23s, the completion slams in: `[gpt] git branch` + pick lines `1. git branch  2. ...` — hold 2.5s (readable). Under it, three spec chips arrive one-by-one on beats: `837,888 params` · `110s training` · `3.1 MB`.
Sequential/interaction: yes — completion reveal beat-locked; 3 chips snap to consecutive beats (reveal quickly, hold full set to respect reading floor)
Audio intent: the payoff — pop on reveal, soft ticks on chips
Audio-coupled idea: chip-by-chip beat reveal; completion pop
Music: continue; brighter
Transition mood: clean → Scene 3

### Scene 3 — The proof — 4.5s
Two terminal log lines, sequential: line 1 "5760 iters | loss 1.40 | free Colab T4" (hold), line 2 "shellmate_model.npz 3.1M — no GPU, no server, no llama" (hold). Caption on beat 10.54: "The whole GPT fits in a file smaller than this video."
Sequential/interaction: yes — two log lines then one caption, each with hold
Audio intent: calm confidence, music carries
Audio-coupled idea: minor bass accent on the caption
Music: steady
Transition mood: slide → Scene 4

### Scene 4 — It learns yours — 4.0s
Terminal again: `$ --mode add_cmd` → `$ --mode finetune` → on strong cue 12.65: "learned in 500 steps" in green. Keycap below: Ctrl+G — "finish your thought mid-command."
Sequential/interaction: yes — two commands then the green result
Audio intent: satisfying, a small win
Audio-coupled idea: key tap, one subscription chime on the green result
Music: continue
Transition mood: wipe → Scene 5

### Scene 5 — Outro — 4.0s
Logo: **shellGPT** big, centered. Subline: "The tiniest GPT for the terminal." URL: `github.com/pratikwayal01/shellgpt`. Music fades over final 1.2s; one soft logo hit.
Sequential/interaction: none
Audio intent: landing, warm close
Audio-coupled idea: logo hit on arrival
Music: fade out
Transition mood: end on black

**Music mood for this video:** upbeat, business-happy, keeps the absurd specs from feeling thin
**Audio summary:** upbeat bed throughout, key ticks for typing, pop + chip ticks on the reveal, one chime on the 500-step win, soft logo hit, fade out.