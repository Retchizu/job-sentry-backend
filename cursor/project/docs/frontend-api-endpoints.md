# Frontend API Endpoints

Base URL (local): `http://127.0.0.1:8000`

## 1) Service Info

- **Method:** `GET`
- **Path:** `/`
- **Request body:** none

### Response (`200`)

```json
{
  "service": "job-sentry-backend",
  "version": "0.2.0",
  "docs": "/docs"
}
```

## 2) Health Check

- **Method:** `GET`
- **Path:** `/health`
- **Request body:** none

### Response (`200`)

```json
{
  "status": "ok",
  "model_loaded": true,
  "mode": "phase6_fused",
  "artifact_path": "artifacts/models/phase6_fused",
  "device": "cpu",
  "message": null
}
```

Notes:
- `status` can be `ok` or `degraded`.
- If `model_loaded` is `false`, prediction requests will return `503`.

## 3) Scam Prediction

- **Method:** `POST`
- **Path:** `/predict`
- **Request body:** required JSON object with `posts` array (minimum 1 item)

### Request body schema

```json
{
  "posts": [
    {
      "text": "optional free-form text",
      "job_title": "optional",
      "job_desc": "optional",
      "skills_desc": "optional",
      "company_profile": "optional",
      "rate": {
        "amount_min": 0,
        "amount_max": 0,
        "currency": "USD",
        "type": "hourly"
      }
    }
  ]
}
```

### Request body rules

- Provide either:
  - non-empty `text`, or
  - at least one of: `job_title`, `job_desc`, `skills_desc`, `company_profile`.
- `posts` must contain at least 1 item.
- If `rate` is provided:
  - `amount_min` and `amount_max` must be `>= 0`
  - `amount_min <= amount_max`
  - `currency` must be a 3-letter uppercase ISO code (example: `USD`, `PHP`)
  - `type` must be one of: `hourly`, `daily`, `weekly`, `monthly`, `yearly`

### Example request

```json
{
  "posts": [
    {
      "job_title": "Remote Data Entry Specialist",
      "job_desc": "Work from home, immediate start, no experience needed",
      "skills_desc": "Typing, spreadsheets",
      "company_profile": "Global talent solutions provider"
    }
  ]
}
```

### Response (`200`)

```json
{
  "scam_probabilities": [0.87],
  "predicted_scam": [true],
  "threshold": 0.5
}
```

### Error responses

- `422 Unprocessable Entity`
  - invalid payload shape/validation
  - empty text and no structured fields for a post
  - request batch larger than server `max_batch_size`
- `503 Service Unavailable`
  - model not loaded (`JOBSENTRY_PHASE6_FUSED_DIR` not configured)
