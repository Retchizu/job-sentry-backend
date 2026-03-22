# Add `rate` to `/predict` request body — Implementation Plan

## Overview

Add an optional nested object `rate` on each `JobPostInput` with `amount_min`, `amount_max`, `currency`, and `type`, matching the JSON shape clients will send for future pipelines. The current phase6 fused model remains **text-only**; this change **does not** alter `combined_text()` or `FusedScamPredictor` unless a later phase explicitly feeds rate into text or the network.

## Current State Analysis

- **`JobPostInput`** (`app/schemas.py`) exposes optional `text` plus structured text fields `job_title`, `job_desc`, `skills_desc`, `company_profile`. There is no `rate` (or salary) field.
- **`POST /predict`** (`app/main.py:94–123`) builds one string per post via `combined_text()` and calls `predictor.predict_proba(texts)` only.
- **`HybridFusedClassifier`** (`app/hybrid_fused_model.py:104–118`) accepts DistilBERT and LSTM tensors derived from that string only; artifacts (`fused_meta.json`) have no slots for structured rate features.

**Key discovery:** Optional API fields that are **not** concatenated into `combined_text()` do not affect scores for the deployed fused checkpoint ([`cursor/project/research/2026-03-22-salary-hours-structured-fields-phase6-fused.md`](../../project/research/2026-03-22-salary-hours-structured-fields-phase6-fused.md)).

## Desired End State

1. Clients may include `rate` on each post in `POST /predict` bodies; FastAPI/Pydantic validate and expose it in OpenAPI (`/docs`).
2. When `rate` is omitted, behavior matches today (same strings → same probabilities).
3. When `rate` is present, values are validated (types, sensible ranges, consistency between min/max).
4. Automated tests cover accepted payloads and validation errors.

### Verification

- **Automated:** `pytest` passes for existing and new tests.
- **Manual:** Swagger UI shows `rate` on `JobPostInput`; sample request with `rate` returns 200 when the model is loaded (or use tests with injected predictor).

## What We're NOT Doing

- **Not** changing `HybridFusedClassifier`, `fused_loader`, or checkpoints.
- **Not** appending `rate` to `combined_text()` or changing prediction logic in this task (reserved for a follow-up if product wants rates to influence the **current** text model).
- **Not** persisting requests to a database or adding new response fields echoing `rate` (unless explicitly requested later).
- **Not** editing `README.md` unless a separate docs task is requested (OpenAPI remains the source of truth for the new shape).

## Implementation Approach

Introduce a small nested Pydantic model `RateInput` (name can match domain language; e.g. `Rate` if you prefer a shorter OpenAPI name) and add `rate: Optional[RateInput] = None` to `JobPostInput`. Use field validators so partial/invalid compensation objects are rejected clearly. Keep `main.py` unchanged for inference so behavior stays identical.

## Phase 1: Schema and validation

### Overview

Define the nested model and attach it to `JobPostInput`.

### Changes Required

#### 1. Nested rate model + optional field

**File:** `app/schemas.py`

**Changes:**

- Add `RateInput` (or `Rate`) with:
  - `amount_min: float` — use `Field(..., ge=0)` (or `gt=0` if business rules forbid zero; default `ge=0` matches “amount” semantics).
  - `amount_max: float` — `Field(..., ge=0)`.
  - `currency: str` — recommend ISO 4217 alpha codes: `Field(..., min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")` so `"PHP"` validates; document in Field `description` that clients send uppercase ISO codes.
  - `type: str` — encode pay period. Options:
    - **A (strict):** `Literal["hourly", "daily", "weekly", "monthly", "yearly"]`.
    - **B (flexible):** `str` with `Field(..., min_length=1, max_length=32)` for unknown future values.
  - Add a **`model_validator`** (Pydantic v2) after model definition: if both amounts are set, enforce `amount_min <= amount_max`.

- On `JobPostInput`, add: `rate: Optional[RateInput] = None`.

- **Do not** change `combined_text()` in this phase (it should continue to ignore `rate`).

**Example (illustrative — adjust imports and validator API to match project Pydantic version):**

```python
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class RateInput(BaseModel):
    amount_min: float = Field(..., ge=0)
    amount_max: float = Field(..., ge=0)
    currency: str = Field(..., min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")
    type: Literal["hourly", "daily", "weekly", "monthly", "yearly"]  # or str with max_length

    @model_validator(mode="after")
    def min_le_max(self) -> "RateInput":
        if self.amount_min > self.amount_max:
            raise ValueError("amount_min must be <= amount_max")
        return self


class JobPostInput(BaseModel):
    text: Optional[str] = None
    job_title: Optional[str] = None
    # ... existing fields ...
    rate: Optional[RateInput] = None
```

**Decision to lock before coding:** Choose strict `Literal` vs free `str` for `type`. If upstream systems may send values outside the list, use `str` + document common values.

### Success Criteria

#### Automated Verification

- [x] `pytest tests/` passes from repo root: `pytest tests/`
- [x] No new linter issues in `app/schemas.py` (run your usual Ruff/mypy if configured)

#### Manual Verification

- [ ] `GET /docs` → `PredictRequest` → `JobPostInput` shows optional `rate` with nested properties
- [ ] Invalid body (e.g. `amount_min` > `amount_max`) returns **422** with a clear validation error

**Implementation note:** After Phase 1 automated checks pass, confirm manual OpenAPI behavior before treating the task as done.

---

## Phase 2: Tests

### Overview

Prevent regressions and document the contract.

### Changes Required

#### 1. Unit-style schema tests (preferred location)

**File:** `tests/test_schemas.py` (new) **or** extend `tests/test_api.py` if you prefer fewer files.

**Cases:**

- Deserialize `JobPostInput` with full `rate` matching the user’s example (amounts as numbers, `currency` `"PHP"`, `type` `"daily"`).
- Reject `amount_min` > `amount_max`.
- Reject bad `currency` length or lowercase if pattern enforces uppercase (or add a `BeforeValidator` to normalize `currency.upper()` — only if product wants lenient input; otherwise keep strict).

#### 2. API integration test with injected predictor

**File:** `tests/test_api.py`

- Extend `test_predict_with_injected_predictor` (or add a sibling test) to `POST` a body that includes `rate` alongside structured text fields; assert **200** and unchanged probability behavior (fake predictor still returns `[0.9]`).

### Success Criteria

#### Automated Verification

- [x] `pytest tests/test_api.py tests/test_schemas.py` (or equivalent) passes

#### Manual Verification

- [ ] None required beyond Phase 1 if CI runs `pytest`

---

## Testing Strategy

### Unit tests

- `RateInput` validation boundaries (min/max ordering, currency pattern, allowed `type` values if using `Literal`).

### Integration tests

- `/predict` accepts posts with `rate` and returns the same shape as today.

### Manual testing steps

1. Start the app with `JOBSENTRY_PHASE6_FUSED_DIR` set and POST a valid payload including `rate`.
2. POST an invalid `rate` and confirm 422.

## Performance Considerations

Negligible: one extra nested object per post parsed by Pydantic.

## Migration Notes

- **Clients:** Backward compatible — existing clients without `rate` unchanged.
- **No server-side migration** — no database.

## References

- Research: [`cursor/project/research/2026-03-22-salary-hours-structured-fields-phase6-fused.md`](../research/2026-03-22-salary-hours-structured-fields-phase6-fused.md)
- Schema: [`app/schemas.py`](../../app/schemas.py)
- Predict handler: [`app/main.py`](../../app/main.py) lines 94–123
- Fused predictor (text-only): [`app/fused_predictor.py`](../../app/fused_predictor.py) lines 62–92
- Prior plan mentioning richer schema: [`cursor/project/plan/2025-03-08-jobsentry-backend-implementation.md`](2025-03-08-jobsentry-backend-implementation.md)
