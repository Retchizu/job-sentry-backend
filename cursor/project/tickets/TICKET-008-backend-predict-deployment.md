# TICKET-008: Backend deployment for `/predict` endpoint

## Objective

Deploy model inference behind a backend API endpoint: `POST /predict`.

## Scope

- Add or update backend route for `POST /predict`.
- Load tokenizer/model artifacts at service startup (not on every request).
- Accept request payload with raw text fields (or `combined_text`) needed for inference.
- Run preprocessing + model inference and map class index to:
  - `legit`
  - `warning`
  - `fraud`
- Return stable JSON response including:
  - `predicted_class`
  - `predicted_label`
  - `legit_probability`
  - `warning_probability`
  - `fraud_probability`
  - `confidence`
- Add request/response validation and clear error handling for invalid payloads.
- Add basic observability (latency + success/error logging).

## Acceptance Criteria

- `POST /predict` returns valid prediction responses for known test inputs.
- Model is loaded once and reused across requests.
- Error responses are deterministic and documented.
- Endpoint is callable from local/dev environment with an example request.

## Deliverables

- Backend handler/controller implementation for `/predict`.
- Minimal integration test(s) for happy path and invalid payload.
- API contract documentation snippet with example request/response.
