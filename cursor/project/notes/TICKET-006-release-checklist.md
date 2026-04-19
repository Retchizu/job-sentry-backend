# TICKET-006: Release checklist (deployers and API consumers)

Use this before promoting a build that serves the phase6 fused model.

## Environment

- [ ] `JOBSENTRY_PHASE6_FUSED_DIR` points at a directory containing `fused_meta.json`, tokenizer files, and weights (`model.safetensors` or `checkpoints/epoch_*.pt`).
- [ ] Optional: `JOBSENTRY_PHASE6_FUSED_CHECKPOINT` — explicit checkpoint path when not using `model.safetensors`.
- [ ] Optional: `JOBSENTRY_DEVICE` — `cpu` or `cuda` (default: CUDA if available, else CPU).
- [ ] Optional: `JOBSENTRY_MAX_BATCH_SIZE` — cap on posts per `POST /predict` request (default `50`).

## Artifact promotion

- [ ] Open `fused_meta.json` in the bundle and confirm **`artifact_version`** matches the release you intend to ship (e.g. `2026-04-20` or a semver you have documented).
- [ ] Prefer deploying a **versioned** directory or object-storage prefix, e.g. `phase6_fused-<artifact_version>/`, and point `JOBSENTRY_PHASE6_FUSED_DIR` at that path.

## Health

- [ ] `GET /health` returns `model_loaded: true`, `mode: phase6_fused`, and expected `device` when the model is configured.
- [ ] With `JOBSENTRY_PHASE6_FUSED_DIR` unset, the app still starts; `model_loaded` is `false` and `POST /predict` returns **503** (degraded mode).

## Smoke

- [ ] `POST /predict` with at least one post containing text (see README example body) returns **200** and softmax fields (`legit_probability`, `warning_probability`, `fraud_probability`) that sum to ~1 per post.
- [ ] Batch over `JOBSENTRY_MAX_BATCH_SIZE` returns **422**.

## Rollback

- [ ] To roll back weights without changing code: prefer swapping `model.safetensors`, or set `JOBSENTRY_PHASE6_FUSED_CHECKPOINT` to a specific `checkpoints/epoch_XX.pt`, or point `JOBSENTRY_PHASE6_FUSED_DIR` at an older artifact bundle. See `README.md` and `app/fused_loader.py` (`resolve_weight_source`).

## Tests

- [ ] From the repo root: `pytest -q` passes.

## Documentation

- [ ] Evaluation narrative and benchmark protocol: [TICKET-006-evaluation-summary.md](./TICKET-006-evaluation-summary.md).
- [ ] API fields and migration from legacy binary responses: [README.md](../../../README.md).
