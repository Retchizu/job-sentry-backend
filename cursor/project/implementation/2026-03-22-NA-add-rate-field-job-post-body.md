# Implementation summary: Add `rate` to `/predict` request body

**Date:** 2026-03-22  
**Plan:** `cursor/project/plan/2026-03-22-add-rate-field-job-post-body.md`

## What was implemented

1. **`app/schemas.py`**
   - Added **`RateInput`**: `amount_min` / `amount_max` (`ge=0`), `currency` (3-letter uppercase ISO pattern + description), `type` as `Literal["hourly", "daily", "weekly", "monthly", "yearly"]`, and a Pydantic v2 **`model_validator`** enforcing `amount_min <= amount_max`.
   - Added **`rate: Optional[RateInput] = None`** on **`JobPostInput`**. **`combined_text()`** is unchanged and does not use `rate`.

2. **`tests/test_schemas.py`** (new)
   - Validates deserialization of `JobPostInput` / `PredictRequest` with a full `rate` (e.g. PHP / daily).
   - Covers rejection of min > max, bad currency length, lowercase currency, and unknown `type`.

3. **`tests/test_api.py`**
   - **`test_predict_with_injected_predictor_and_rate`**: `POST /predict` with structured fields + `rate` → **200**, same fake probabilities as without `rate`.
   - **`test_predict_422_when_rate_min_exceeds_max`**: invalid `rate` → **422**.

4. **`app/config.py`** (test / env ergonomics)
   - **`field_validator`** on `phase6_fused_dir` and `phase6_fused_checkpoint`: empty or whitespace-only strings → **`None`**, so tests can **`setenv(..., "")`** to override a `.env` value and simulate “no fused dir” (previously `delenv` did not override file-based settings).

## Verification

- **`pytest tests/`** — 24 passed (local run).
- **Lint:** no issues reported on edited files.

## Not done (per plan)

- No changes to **`HybridFusedClassifier`**, **`fused_loader`**, **`fused_predictor`**, or **`combined_text()`** for model input.
- **README** not updated (OpenAPI is the contract).

## Manual verification (from plan)

- **`GET /docs`** → `PredictRequest` → `JobPostInput` shows optional **`rate`** and nested fields.
- Invalid body (e.g. `amount_min` > `amount_max`) returns **422** with a clear error (also covered by **`test_predict_422_when_rate_min_exceeds_max`**).
