# Job Sentry Backend

FastAPI backend for Job Sentry.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs

## Running with the real model

The app loads the thesis-trained DistilBERT when a checkpoint directory containing `model.safetensors` is available. Without it, the app falls back to the base (untrained) model and logs a warning; predictions are not meaningful.

**Prerequisites:** The checkpoint directory must contain `model.safetensors` (e.g. from the thesis project: `models/distilbert_run/checkpoint-5412/`).

**Option A — Relative path:** Clone or copy the thesis project so the checkpoint is at `thesis-scam-job-post/models/distilbert_run/checkpoint-5412` relative to the process working directory (e.g. repo root). No env change needed.

**Option B — Override path:** Set `JOBSENTRY_MODEL_ARTIFACT_PATH` to the absolute or relative path of the checkpoint directory that contains `model.safetensors`:

```bash
export JOBSENTRY_MODEL_ARTIFACT_PATH=/path/to/checkpoint-5412
uvicorn app.main:app --reload
```

**Verify:** Start the app with `uvicorn app.main:app --reload`. Check logs for `Loading fine-tuned model from ...` (not "No model.safetensors found"). Call `GET /health` and confirm `model_loaded: true` and `status: "ok"`.

## Deployment and model artifact

**Production:** The app expects the checkpoint directory (containing `model.safetensors`) at the path given by `JOBSENTRY_MODEL_ARTIFACT_PATH`. Options: (1) Mount a volume or bind-mount that directory into the container or filesystem at a known path and set `JOBSENTRY_MODEL_ARTIFACT_PATH` to that path; (2) Copy the checkpoint into the image at a fixed path and set `JOBSENTRY_MODEL_ARTIFACT_PATH` accordingly; (3) Use a shared filesystem or artifact store and set the env var to that path.

**Degraded mode:** If the path is missing or does not contain `model.safetensors`, the app starts with the base (untrained) DistilBERT and logs a warning; `GET /health` still returns `model_loaded: true` but predictions are not meaningful. Ensure the artifact is available in production for real use.

**Security:** Do not commit `model.safetensors` to the app repo; obtain it from the thesis project or a secure artifact store.

**Checkpoint:** Default is `thesis-scam-job-post/models/distilbert_run/checkpoint-5412` (final training step). Available checkpoints: 783, 1566, 2349, 3608, 5412. See the thesis-scam-job-post project for training details and metrics.

## Hybrid model (optional)

By default the app uses only the DistilBERT checkpoint. To enable ensemble predictions, set one or both of these environment variables:

| Variable | Description |
|---|---|
| `JOBSENTRY_PHASE6_MERGED_PATH` | Path to a joblib pipeline artifact (e.g. TF-IDF + Logistic Regression from the thesis `phase6_merged`). |
| `JOBSENTRY_BILSTM_ARTIFACT_PATH` | Path to a Keras Bi-LSTM model (`.h5` or `.keras`). A `word_index.json` should be in the same directory. |
| `JOBSENTRY_HYBRID_COMBINATION` | Combination method (default: `soft_voting`). |

When either path is set and the artifact loads successfully, the app runs all loaded models on each prediction and averages their scam probabilities (soft voting). `GET /health` reports which models are loaded and whether hybrid mode is active.

When neither path is set, behavior is unchanged — DistilBERT only.

Artifact format must match expectations (joblib pipeline for phase6, Keras for Bi-LSTM). See the thesis-scam-job-post project for exact artifact paths and formats.

## FastAPI setup checklist

- [x] `requirements.txt` with `fastapi`, `uvicorn[standard]`
- [x] App package `app/` with `main.py`
- [x] FastAPI app instance with title/version
- [x] Root route `GET /`
- [x] Health check `GET /health`
- [ ] Virtual environment created and deps installed (run locally)
- [ ] Optional: routers, Pydantic models, env config, tests
