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
- Use the reusable helper script `scripts/vast.py` for all Vast.ai operations. It reads the API key from the `VAST_API_KEY` environment variable; do not hard-code tokens.
  - Check balance/credit: `VAST_API_KEY=$KEY python scripts/vast.py balance`
  - List instances: `VAST_API_KEY=$KEY python scripts/vast.py list`
  - Search RTX 3090 offers (Asia by default): `VAST_API_KEY=$KEY python scripts/vast.py search`
  - Search US/CA offers: `VAST_API_KEY=$KEY python scripts/vast.py search --region us`
  - Create instance: `VAST_API_KEY=$KEY python scripts/vast.py create <offer_id> [--image ... --label ...]`
  - Wait for ready URL: `VAST_API_KEY=$KEY python scripts/vast.py wait <instance_id>`
  - Health check: `python scripts/vast.py health <url>`
  - Smoke test: `python scripts/vast.py smoke <url>`
  - Fetch logs: `VAST_API_KEY=$KEY python scripts/vast.py logs <instance_id>`
  - Destroy one instance: `VAST_API_KEY=$KEY python scripts/vast.py destroy <instance_id>`
  - Destroy all active OmniVoice instances: `VAST_API_KEY=$KEY python scripts/vast.py destroy-all --yes`
- Use label prefix `omnivoice-api` for OmniVoice Vast.ai service instances.
- When the user says "关闭vast.ai实例" or asks to close/stop Vast.ai instances, destroy all Vast.ai instances whose `label` starts with `omnivoice-api` and whose state is running/loading/active. Confirm the remaining instance list afterward.
- When the user says "开启vast.ai实例" or asks to start/open a Vast.ai instance, create a new RTX 3090 instance with:
  - image: `liudunxu/omnivoice-api:vast-gpu` unless the user specifies a fixed tag
  - label: `omnivoice-api-mvp-<short-tag-or-date>`
  - disk: `80`
  - runtype: `args`
  - env: `{"-p 8000:8000": "1", "PORT": "8000", "HOST": "0.0.0.0", "MODEL_DIR": "/workspace/models"}`
- Prefer verified, rentable, on-demand, single RTX 3090 offers with at least 24GB GPU RAM, at least one direct port, and at least 80GB disk space. 选择实例时优先选亚洲地区且能访问 Docker Hub 的实例；若不可用，再选美洲地区（US/CA）。
- After creating an instance, wait for the public port, then verify `GET /` returns `ok` and `GET /health` returns `{"ok": true, ...}`. Report the URL, instance id, image tag, label, GPU, and `dph_total`.
- Do not keep old and new Vast.ai instances running after a redeploy unless the user explicitly asks for overlap. Once the new instance is verified, destroy old `omnivoice-api*` instances.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
