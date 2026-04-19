# TICKET-008: `/predict` contract (3-class labels, observability, docs) — Implementation Plan

## Overview

Align the public **`POST /predict`** JSON contract with [`cursor/project/tickets/TICKET-008-backend-predict-deployment.md`](../tickets/TICKET-008-backend-predict-deployment.md): response fields `predicted_class`, `predicted_label`, per-class probabilities, `confidence`, plus request validation, error behavior, observability (latency and success/error logging), tests, and a documented example. The deployed **phase6 fused** artifact remains a **binary** classifier (`num_labels=2`, `FusedScamPredictor.predict_proba` → P(scam)); [`cursor/project/tickets/TICKET-001-label-schema-and-mapping.md`](../tickets/TICKET-001-label-schema-and-mapping.md) defines **numeric** classes `0=legit`, `1=warning`, `2=fraud`. This plan implements a **deterministic mapping** from `(P(scam), heuristic warning codes)` to three-way labels and one-hot probabilities, without retraining or replacing the model.

## Current State Analysis

- **Route and startup loading**: [`app/main.py`](../../app/main.py) registers `POST /predict`, loads `FusedScamPredictor` once in `lifespan`, stores `app.state.predictor` (lines 39–49, 131–147).
- **Request**: [`app/schemas.py`](../../app/schemas.py) — `PredictRequest` / `JobPostInput` / `combined_text()` already match “raw text or combined fields” (lines 31–59).
- **Inference**: [`app/fused_predictor.py`](../../app/fused_predictor.py) — `predict_proba` returns scalar P(scam) from 2-class softmax (lines 62–92). [`app/fused_loader.py`](../../app/fused_loader.py) builds `HybridFusedClassifier` with `num_labels` from `fused_meta` (default 2) (lines 127–136).
- **Response gap**: `PredictResponse` today exposes `scam_probabilities`, `predicted_scam`, `threshold`, `warnings` ([`app/schemas.py`](../../app/schemas.py) lines 62–70) — not the TICKET-008 field set.
- **Data-pipeline alignment**: Three-way `risk_class` exists in [`datasets_row_merge.py`](../../datasets_row_merge.py) (TICKET-001 precedence in code); the **serving** path does not yet expose the same string/int labels.
- **Observability gap**: No per-request timing or success/error logs in `predict()` ([`app/main.py`](../../app/main.py) lines 96–128).
- **Tests**: [`tests/test_api.py`](../../tests/test_api.py) assert the current binary response shape (e.g. lines 67–70, 105–108).

**Key discoveries**

- [`cursor/project/research/2026-04-18-TICKET-008-backend-predict-deployment.md`](../research/2026-04-18-TICKET-008-backend-predict-deployment.md) documents the binary-vs-ticket mismatch; this plan resolves it by policy mapping, not by changing `HybridFusedClassifier` output size in this task.
- No `Makefile`; automated checks use `pytest` per [`README.md`](../../README.md).

## Desired End State

- **`POST /predict`** returns JSON matching TICKET-008 fields (below), with **`warnings`** retained as today (heuristic codes per post).
- **Label mapping** (deterministic, documented, unit-tested):
  - Inputs per post: `p` = P(scam) from `predict_proba`, `heuristic_codes` = `compute_warnings(text)`.
  - Config: `JOBSENTRY_WARN_THRESHOLD` (`T_warn`) and `JOBSENTRY_FRAUD_THRESHOLD` (`T_fraud`) with **`T_warn < T_fraud`** (defaults **0.35** and **0.65**), exposed in [`app/config.py`](../../app/config.py) and [`.env.example`](../../.env.example).
  - **Precedence** (matches product intent: fraud overrides ambiguity):
    1. If `p >= T_fraud` → `predicted_class = 2`, `predicted_label = "fraud"`.
    2. Else if `p >= T_warn` **or** `len(heuristic_codes) > 0` → `predicted_class = 1`, `predicted_label = "warning"`.
    3. Else → `predicted_class = 0`, `predicted_label = "legit"`.
  - **Per-class probabilities**: **one-hot** for the chosen class: the winning class gets `1.0`, the others `0.0` (sum = 1; stable and testable).
  - **`confidence`**: per post, `max(p, 1.0 - p)` — strength of the underlying **binary** model (document in README; not the same as one-hot mass).
- **Observability**: Structured log line after successful inference with duration (ms) and batch size; log lines on 503/422 paths with reason (no PII / full raw text in logs).
- **Documentation**: README section with **example request/response JSON** and error cases; `.env.example` updated for new thresholds.
- **Verification**: `pytest` green; manual `uvicorn` + `/docs` or `curl` smoke test optional.

### Key Discoveries

- Binary head output is fixed for current artifacts; three **soft** calibrated class probabilities would require a **3-class trained model** — explicitly out of scope here (see “What We’re NOT Doing”).

## What We’re NOT Doing

- Retraining or swapping in a **3-class** fused checkpoint (would change `num_labels`, loss, and artifacts).
- Changing `JobPostInput` / `combined_text()` behavior or using `rate` in the model input (unless a separate ticket).
- Adding `/batch-predict` or non-HTTP transports.
- Prometheus/OpenTelemetry metrics (ticket asks for **basic** logging only).
- Keeping the old response fields (`scam_probabilities`, `predicted_scam`, `threshold`) as aliases — **breaking change** acceptable; call out in README.

## Implementation Approach

1. Add a small **pure function** module (e.g. `app/risk_labels.py`) implementing `map_binary_to_risk_class(p, heuristic_codes, t_warn, t_fraud) -> ...` returning predicted indices, string labels, one-hot triples, and `confidence` scalar. Keeps logic testable without HTTP.
2. Extend **Settings** with `warn_threshold` and `fraud_threshold` with validation `warn < fraud`.
3. Replace **`PredictResponse`** fields in [`app/schemas.py`](../../app/schemas.py); update **`predict()`** in [`app/main.py`](../../app/main.py) to build the new response, wrap `predict_proba` in timing, add logging.
4. Update **`tests/test_api.py`**, add **`tests/test_risk_labels.py`** (or similar) for mapping edge cases; adjust **`tests/test_schemas.py`** if response models are asserted.
5. Update **README** and **`.env.example`**.

## Phase 1: Mapping module and configuration

### Overview

Centralize TICKET-001–aligned mapping and threshold settings.

### Changes Required

#### 1. New module: `app/risk_labels.py`

**Changes**: Implement:

- Constants or `Literal` for `predicted_label` strings: `"legit"`, `"warning"`, `"fraud"`.
- `def map_binary_to_risk(...) -> tuple[int, str, tuple[float, float, float], float]`:
  - Returns `predicted_class`, `predicted_label`, `(legit_probability, warning_probability, fraud_probability)` one-hot, and `confidence = max(p, 1-p)`.

```python
# Illustrative — actual names/types should match project style
def map_binary_to_risk(
    p_scam: float,
    heuristic_codes: list[str],
    *,
    warn_threshold: float,
    fraud_threshold: float,
) -> tuple[int, str, tuple[float, float, float], float]:
    ...
```

#### 2. `app/config.py`

**Changes**: Add `warn_threshold: float`, `fraud_threshold: float` with defaults **0.35** / **0.65**; add a validator ensuring `warn_threshold < fraud_threshold`.

#### 3. `.env.example`

**Changes**: Document `JOBSENTRY_WARN_THRESHOLD` and `JOBSENTRY_FRAUD_THRESHOLD` (names must match `Settings` field → env mapping).

### Success Criteria

#### Automated Verification

- [x] `pytest -q` passes for new unit tests on `map_binary_to_risk` (cases: `p` above/between/below thresholds; heuristic-only warning; fraud wins when `p` high).
- [x] Config validation fails fast if `warn_threshold >= fraud_threshold` (unit test with `pytest.raises` or Settings construction).

#### Manual Verification

- [ ] Skim new env vars in `.env.example` for typos.

**Implementation Note**: After automated checks pass, pause for quick confirmation before Phase 2 if you rely on cross-team API review.

---

## Phase 2: API schema and handler

### Overview

Swap `PredictResponse` to TICKET-008 shape; wire mapping in `predict()`; add logging and latency measurement.

### Changes Required

#### 1. `app/schemas.py`

**Changes**:

- Replace `PredictResponse` fields with:
  - `predicted_class: list[int]` — values in `{0, 1, 2}` per post
  - `predicted_label: list[str]` — `legit` / `warning` / `fraud`
  - `legit_probability`, `warning_probability`, `fraud_probability`: each `list[float]`, length = number of posts
  - `confidence: list[float]`
  - Keep `warnings: list[list[str]]` as today
- Ensure OpenAPI descriptions reference TICKET-001 numeric mapping for `predicted_class`.

#### 2. `app/main.py`

**Changes**:

- After `probs = predictor.predict_proba(texts)` and `warnings = [...]`, for each index `i` call `map_binary_to_risk(probs[i], warnings[i], ...)` — **Note**: for mapping, use **heuristic codes before** one-hot overwrite: mapping input should use the same `warnings[i]` list returned to the client. Class precedence uses `heuristic_codes` as non-empty set, not the post-hoc one-hot.
- Wrap `predict_proba` (and optionally full handler body) with `time.perf_counter()`; compute `latency_ms`.
- `logger.info` on success: e.g. `predict_ok posts=N latency_ms=...` (no raw post text).
- `logger.warning` when returning 422/503 with short reason code.

#### 3. `README.md`

**Changes**:

- Replace binary response description with TICKET-008 fields; add **example JSON** request/response.
- Document **deterministic error** cases: 503 when model missing; 422 for empty text, batch too large, invalid `rate`.
- Note **breaking change** from previous `scam_probabilities` / `predicted_scam` / `threshold` fields.

### Success Criteria

#### Automated Verification

- [x] `pytest -q` full suite passes.
- [x] All `/predict` tests in [`tests/test_api.py`](../../tests/test_api.py) updated to new JSON keys; fake predictor tests still inject `_FakePred` with `predict_proba` returning controlled `p` values to hit each class band.

#### Manual Verification

- [ ] `uvicorn app.main:app --reload` → `POST /predict` via `/docs` returns new fields for a sample body.
- [ ] Logs show a line with latency on successful predict when log level allows.

---

## Phase 3: Tests and documentation hardening

### Overview

Fill gaps for “invalid payload” and mapping edge cases; ensure example request/response is copy-pasteable.

### Changes Required

#### 1. `tests/test_risk_labels.py` (new)

**Changes**: Parametrized tests for boundary values `p` near `T_warn` and `T_fraud`; test heuristic-only warning with low `p`.

#### 2. `tests/test_api.py`

**Changes**: Add or adjust tests so at least one case per `predicted_label` appears using `_FakePred` and fixed `p` values (with empty or controlled warnings where needed).

#### 3. `tests/test_schemas.py`

**Changes**: Serialize/deserialize a full `PredictResponse` with lists of equal length.

#### 4. README

**Changes**: Short “Migration” bullet for clients that parsed old fields.

### Success Criteria

#### Automated Verification

- [x] `pytest -q` passes.

#### Manual Verification

- [ ] Example JSON in README matches actual OpenAPI schema (field names and nesting).

---

## Testing Strategy

### Unit Tests

- `map_binary_to_risk`: all branches, threshold boundaries, heuristic-only path.
- Settings: invalid threshold ordering rejected.

### Integration Tests (`TestClient`)

- 503 / 422 unchanged behavior with updated assertions where applicable.
- 200 responses assert new keys and expected labels for injected `p`.

### Manual Testing Steps

1. Export `JOBSENTRY_PHASE6_FUSED_DIR` to real artifacts; start `uvicorn`; call `/predict` with structured fields and verify labels behave sensibly.
2. Toggle `JOBSENTRY_WARN_THRESHOLD` / `JOBSENTRY_FRAUD_THRESHOLD` and confirm boundary behavior (optional dev check).

## Performance Considerations

- Mapping is O(n) over posts with negligible overhead vs existing `predict_proba`.
- Logging should remain O(1) per request (no huge payloads).

## Migration Notes

- **Breaking API change**: clients must switch to the new response fields; document in README and release notes if applicable.

## References

- Ticket: [`cursor/project/tickets/TICKET-008-backend-predict-deployment.md`](../tickets/TICKET-008-backend-predict-deployment.md)
- Label schema: [`cursor/project/tickets/TICKET-001-label-schema-and-mapping.md`](../tickets/TICKET-001-label-schema-and-mapping.md)
- Research: [`cursor/project/research/2026-04-18-TICKET-008-backend-predict-deployment.md`](../research/2026-04-18-TICKET-008-backend-predict-deployment.md)
- Current app: [`app/main.py`](../../app/main.py), [`app/schemas.py`](../../app/schemas.py), [`app/fused_predictor.py`](../../app/fused_predictor.py)
