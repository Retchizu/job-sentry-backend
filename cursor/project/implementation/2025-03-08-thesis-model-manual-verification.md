# Thesis Model Manual Verification

**Plan**: `cursor/project/plan/2025-03-08-thesis-trained-model-remaining-work.md` (Phase 2)

## Checklist

- [ ] App running with real checkpoint (Phase 1 steps)
- [ ] **Health**: `GET /health` → `status: "ok"`, `model_loaded: true`, `model_name` present
- [ ] **Single predict**: `POST /predict` with sample job post → response has `prediction`, `confidence`, `scam_probability`, `warning_signals`; sanity-check legitimate vs scam
- [ ] **Batch predict**: `POST /batch-predict` with 5–10 job posts → confirm `count` and per-item shape
- [ ] **Latency**: 10–20 requests to `POST /predict`, compute p95 (target: p95 < 2 s)

## Recorded Results

**Date**: _fill in_

**Environment**: _e.g. macOS, Python 3.x, CPU_

**Outcome**:
- Health: _ok / notes_
- Predict / batch-predict: _behaved as expected / notes_
- p95 latency: _e.g. "p95 < 2 s" or actual value_
- How measured: _e.g. "20 sequential requests" or one-liner script_
