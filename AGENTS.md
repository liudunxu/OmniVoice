# AGENTS.md

Behavioral guidelines for AI coding agents working on OmniVoice project.

## Project Context

OmniVoice is a multilingual zero-shot TTS model supporting 600+ languages. Built on diffusion language model architecture.

## Core Principles

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Project-Specific Guidelines

### TTS Model Changes
- Understand the model architecture before modifying inference code
- Test with multiple languages when changing generation logic
- Preserve backward compatibility for existing API endpoints

### Audio Processing
- Always validate audio format (sample rate, channels, bit depth)
- Use soundfile for I/O operations, torchaudio for transformations
- Clean up temporary audio files after processing

### API Development
- Maintain consistent response format: `{"ok": bool, ...}`
- Return proper HTTP status codes (400 for client errors, 502 for server errors)
- Log request IDs for traceability

### Dependencies
- Check existing dependencies before adding new ones
- Prefer well-maintained, widely-used libraries
- Document any version constraints

### Vast.ai Instance Operations

- The `VAST_API_KEY` is stored in `~/.zshrc` (line 162). Load it from there or export it explicitly; do not hard-code tokens in this repository.
- Speaker diarization (`diarize=true` on `/api/whisper/transcribe`) uses the gated `pyannote/speaker-diarization-community-1` model, so every new instance needs an HF token: read `HF_TOKEN` from `~/.zshrc` or the environment (never hard-code it, never put it in the Dockerfile/image) and pass it at creation time: `VAST_API_KEY=$KEY python scripts/vast.py create <offer_id> --env HF_TOKEN="$HF_TOKEN"`. The token's HF account must have accepted the model's terms on its HuggingFace page, otherwise diarization degrades to `diarization.error` (transcription itself is unaffected).
- Use the reusable helper script `scripts/vast.py` for all Vast.ai operations. It reads the API key from the `VAST_API_KEY` environment variable; do not hard-code tokens.
  - Check balance/credit: `VAST_API_KEY=$KEY python scripts/vast.py balance`
  - List instances: `VAST_API_KEY=$KEY python scripts/vast.py list`
  - Search RTX 4090 offers (Asia by default): `VAST_API_KEY=$KEY python scripts/vast.py search`
  - Search US/CA offers: `VAST_API_KEY=$KEY python scripts/vast.py search --region us`
  - Create instance: `VAST_API_KEY=$KEY python scripts/vast.py create <offer_id> [--image ... --label ...]`
  - Wait for ready URL: `VAST_API_KEY=$KEY python scripts/vast.py wait <instance_id>`
  - Health check: `python scripts/vast.py health <url>`
  - Smoke test: `python scripts/vast.py smoke <url>`
  - Fetch logs: `VAST_API_KEY=$KEY python scripts/vast.py logs <instance_id>`
  - Destroy one instance: `VAST_API_KEY=$KEY python scripts/vast.py destroy <instance_id>`
  - Stop one instance (pause, keep it): `VAST_API_KEY=$KEY python scripts/vast.py stop <instance_id>`
  - Stop all active OmniVoice instances: `VAST_API_KEY=$KEY python scripts/vast.py stop-all --yes`
  - Destroy all active OmniVoice instances: `VAST_API_KEY=$KEY python scripts/vast.py destroy-all --yes`
- Use label prefix `omnivoice-api` for OmniVoice Vast.ai service instances.
- When the user says "停止vast.ai实例" or asks to stop/pause Vast.ai instances, **stop** all Vast.ai instances whose `label` starts with `omnivoice-api` and whose state is running/loading/active. Confirm the instance list afterward.
- When the user says "销毁vast.ai实例"/"关闭vast.ai实例" or asks to destroy/close Vast.ai instances, **destroy** all Vast.ai instances whose `label` starts with `omnivoice-api` and whose state is running/loading/active. Confirm the remaining instance list afterward.
- When the user says "开启vast.ai实例" or asks to start/open a Vast.ai instance, **launch two RTX 5090-or-better instances** (fall back to RTX 4090 / RTX 3090 only when no ≥30GB offers are available) with:
  - image: `liudunxu/omnivoice-api:vast-gpu` unless the user specifies a fixed tag
  - label: `omnivoice-api-mvp-<short-tag-or-date>-<region>`
  - disk: `80`
  - runtype: `args`
  - env: `{"-p 8000:8000": "1", "PORT": "8000", "HOST": "0.0.0.0", "MODEL_DIR": "/workspace/models"}`
- Prefer verified, rentable, on-demand, single-GPU offers with **at least 30GB GPU RAM**, at least one direct port, and at least 80GB disk space. **GPU preference order (always try the next tier down only if the current tier has no available offers):**
  1. **≥30GB VRAM tier** — preferred default. Includes RTX 5090 (32GB), RTX 6000 Ada (48GB), RTX PRO 5000, A100, H100, etc. Pick the cheapest available ≥30GB offer first.
  2. RTX 4090 / RTX 3090 / RTX 3090 Ti (24GB) — fallback only when no ≥30GB offers are available; RTX 4090 does **not** have 30GB VRAM.
  Use `scripts/vast.py search --gpu-name "RTX 5090"` first. If no RTX 5090 offer is available, a verified, rentable, on-demand RTX 4090 in Southeast Asia is an acceptable fallback. Outside Southeast Asia, only fall back to `RTX 4090` (or `RTX 3090`) when the ≥30GB search returns an empty list. VoxCPM occupies ~22 GiB on a single 24GB card, so run the OmniVoice and VoxCPM engines exclusively (call `POST /api/unload` before switching engines) and keep `VAST_GPU` host-side flows on the same engine the smoke test exercises. The `scripts/vast.py smoke` helper intentionally only probes `/health` and `/api/health` — it does **not** call `/api/synthesize` or `/api/voxcpm/synthesize`, so it will not lazy-load either engine onto the GPU. **Prioritize Southeast Asia for both instances.** If only one Southeast Asia offer is available, place the second instance in the next preferred region (East Asia, then Central/West Asia, then Americas US/CA). Verify the host can reach Docker Hub before committing; stalled `Pulling fs layer` or registry timeouts mean the offer should be abandoned.
- After creating instances, wait for the public ports and verify `GET /` returns `ok` and `GET /health` returns `{"ok": true, ...}`. **Report the first instance that passes health checks immediately; do not wait for both instances to be ready.** Once both are verified, report the second URL as well.
- Do not keep old and new Vast.ai instances running after a redeploy unless the user explicitly asks for overlap. Once the new instance is verified, destroy old `omnivoice-api*` instances.

### Known Working Vast.ai Instances

Track instances that have successfully launched and run the OmniVoice API. When launching a new instance, prefer offers whose `machine_id` matches one of these proven hosts, as the Docker image is more likely to be cached locally and startup will be faster.

| Machine ID | GPU        | Region        | Country | Image used                        | Commits / Notes                              |
|------------|------------|---------------|---------|-----------------------------------|----------------------------------------------|
| 140986     | RTX 3090   | East Asia     | TW      | `liudunxu/omnivoice-api:vast-gpu` | `5648d6c` (Return audio QC for synthesis)    |
| 141336     | RTX 3090 Ti| Southeast Asia| TH      | `liudunxu/omnivoice-api:latest`   | `2026-07-10` (Latest image smoke test pass)  |
| 136815     | RTX 5090   | Southeast Asia| TH      | `liudunxu/omnivoice-api:vast-gpu` | `6c748fe` (cu128 + ort 1.23; /api/separate verified on sm_120) |

Hosts observed stalling on Docker Hub image pulls (avoid when possible): machine 57952 / 91308 (both on public IP 209.146.116.50, US). A pull whose `status_msg` shows no newly completed layer for 10+ minutes should be abandoned.

Use this list to choose a preferred offer during `scripts/vast.py search`, but fall back to the next best offer if none of these machines are available.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
