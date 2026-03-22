# Job Sentry Backend

FastAPI backend for Job Sentry — **phase6 fused** (DistilBERT + word BiLSTM + fusion head) inference when local artifacts are configured.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** `tensorflow` is listed for optional legacy Keras BiLSTM workflows; it is **not** required for the fused model. If `pip install` fails on your platform for TensorFlow, install the other packages individually or use a Python version TensorFlow supports.

Copy `.env.example` to `.env` and set variables as needed.

## Run

```bash
uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000  
- Docs: http://127.0.0.1:8000/docs  

## Phase 6 fused model (recommended)

The app loads **`HybridFusedClassifier`** weights and tokenizers from a single artifact directory (for example `artifacts/models/phase6_fused/`). Layout:

- `fused_meta.json` — hyperparameters and default threshold  
- `word_index.json` — LSTM vocabulary (from training)  
- `tokenizer.json`, `tokenizer_config.json` — DistilBERT tokenizer  
- **Weights:** `model.safetensors` **or** `checkpoints/epoch_NN.pt` (if safetensors is missing, the **highest** `NN` is used unless you override)

**Configure:**

```bash
export JOBSENTRY_PHASE6_FUSED_DIR=/absolute/or/relative/path/to/phase6_fused
uvicorn app.main:app --reload
```

Optional:

- `JOBSENTRY_PHASE6_FUSED_CHECKPOINT` — path to a specific `epoch_XX.pt`  
- `JOBSENTRY_DEVICE` — `cpu` or `cuda` (default: CUDA if available, else CPU)  
- `JOBSENTRY_MAX_BATCH_SIZE` — cap for number of `posts` in one `/predict` call  

**Failure behavior:** If `JOBSENTRY_PHASE6_FUSED_DIR` is set but artifacts are missing or weights cannot be loaded, the process **exits at startup** with an error. If the variable is **unset**, the app starts in a **degraded** mode: `GET /health` reports `model_loaded: false` and `POST /predict` returns **503**.

**Health check:** `GET /health` returns `model_loaded`, `mode` (`phase6_fused` or `none`), `device`, and optional `message` when no model is configured.

**Predict:** `POST /predict` with JSON body `{"posts": [ ... ]}`. Each post may use either:

- `"text": "..."` — single combined string, or  
- Structured fields merged like training `combined_text`: `job_title`, `job_desc`, `skills_desc`, `company_profile` (non-empty parts joined with spaces).

Response: `scam_probabilities`, `predicted_scam` (using threshold from `fused_meta.json` when present), and `threshold`.

**Performance:** Each request runs DistilBERT and the LSTM branch; use a GPU in production for throughput. CPU is supported but slower.

## Tests

```bash
pytest -q
```

## Legacy DistilBERT-only and hybrid ensemble

Earlier versions of this repository documented DistilBERT-only checkpoints (`JOBSENTRY_MODEL_ARTIFACT_PATH`) and optional TF-IDF / Keras ensembles. The **current** `app` implements **only** the phase6 fused path above. Restoring DistilBERT-only serving in the same process is a separate change; the old env vars are kept in `.env.example` as comments for reference.

## FastAPI checklist

- [x] `requirements.txt` with FastAPI stack and torch/transformers for fused inference  
- [x] `app/` package with `main.py`, fused model loader, and tests  
- [x] `GET /`, `GET /health`, `POST /predict`  
- [ ] Virtual environment and deps installed locally  
- [ ] `JOBSENTRY_PHASE6_FUSED_DIR` pointed at real artifacts in your environment  
