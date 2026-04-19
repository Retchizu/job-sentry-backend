# Job Sentry Backend

FastAPI backend for Job Sentry — **phase6 fused** (DistilBERT + word BiLSTM + fusion head) inference when local artifacts are configured.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** TensorFlow is **not** in the default `requirements.txt` (it is optional for legacy Keras/BiLSTM workflows and often has no wheel for the newest Python versions). The fused API does **not** need it. If you need TensorFlow, use a [supported Python version](https://www.tensorflow.org/install/pip) and run `pip install -r requirements-optional-tensorflow.txt`.

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
- `JOBSENTRY_WARN_THRESHOLD` / `JOBSENTRY_FRAUD_THRESHOLD` / `JOBSENTRY_CONFIDENCE_THRESHOLD` — used by optional **binary→3-way** helpers (see below), **not** by default `POST /predict` softmax (defaults `0.35` / `0.65` / `0.5`; require `WARN < FRAUD`)  

**Failure behavior:** If `JOBSENTRY_PHASE6_FUSED_DIR` is set but artifacts are missing or weights cannot be loaded, the process **exits at startup** with an error. If the variable is **unset**, the app starts in a **degraded** mode: `GET /health` reports `model_loaded: false` and `POST /predict` returns **503**.

**Health check:** `GET /health` returns `model_loaded`, `mode` (`phase6_fused` or `none`), `device`, and optional `message` when no model is configured.

### `POST /predict`

Request body: `{"posts": [ ... ]}` (at least one post). Each post may use either:

- `"text": "..."` — single combined string, or  
- Structured fields merged like training `combined_text`: `job_title`, `job_desc`, `skills_desc`, `company_profile` (non-empty parts joined with spaces).

Optional `rate` on each post is validated but **not** fed into the text model.

**Response** (one entry per post, parallel arrays):

| Field | Meaning |
| --- | --- |
| `predicted_class` | `0` = legit, `1` = warning, `2` = fraud (TICKET-001 schema); **argmax** over the three softmax probabilities |
| `predicted_label` | `"legit"`, `"warning"`, or `"fraud"` (same index as `predicted_class`) |
| `legit_probability`, `warning_probability`, `fraud_probability` | **Softmax** probabilities from the fused 3-logit head; they **sum to ~1.0** per post (not one-hot) |
| `confidence` | `max(legit_probability, warning_probability, fraud_probability)` — probability of the predicted class |
| `warnings` | Heuristic regex codes (e.g. `upfront_payment`), same order as `posts` |

**Optional threshold policies:** Default inference does **not** use `JOBSENTRY_*_THRESHOLD` to choose the class. The model emits a full **3-class softmax**; `predicted_class` / `predicted_label` follow **argmax** on that triple. For workflows that only have a scalar **P(scam)** (binary head), `app.risk_labels.map_binary_to_risk` maps to legit/warning/fraud using `JOBSENTRY_WARN_THRESHOLD` and `JOBSENTRY_FRAUD_THRESHOLD`. `JOBSENTRY_CONFIDENCE_THRESHOLD` is reserved for future product rules (e.g. “review if confidence below τ”) and is **not** applied by `POST /predict` today. As a **documentation-only** example of a borderline policy: you might escalate when the top two softmax values are within a small ε (implement in the client or a later middleware if needed).

**Errors (deterministic):**

| HTTP | When |
| --- | --- |
| **503** | No fused model loaded (`JOBSENTRY_PHASE6_FUSED_DIR` unset or startup failed) |
| **422** | Empty combined text, invalid `rate` (e.g. min > max), or batch larger than `JOBSENTRY_MAX_BATCH_SIZE` |

**Example**

Request body:

```json
{
  "posts": [
    { "job_title": "Engineer", "job_desc": "Build reliable systems." }
  ]
}
```

Example response shape (values depend on model weights):

```json
{
  "predicted_class": [0],
  "predicted_label": ["legit"],
  "legit_probability": [0.91],
  "warning_probability": [0.06],
  "fraud_probability": [0.03],
  "confidence": [0.91],
  "warnings": [[]]
}
```

**Migration:** Earlier responses used `scam_probabilities`, `predicted_scam`, and `threshold` instead of the fields above — update clients accordingly. If you assumed **one-hot** `legit_probability` / `warning_probability` / `fraud_probability`, treat them as **softmax** masses that sum to ~1 per post.

**Performance:** Each request runs DistilBERT and the LSTM branch; use a GPU in production for throughput. CPU is supported but slower.

## Tests

```bash
pytest -q
```

## Validation and rollout (TICKET-006)

Evidence-backed evaluation, benchmark protocol (multiclass vs legacy binary baselines), and consumer-facing rollout notes:

- [TICKET-006 evaluation summary](cursor/project/notes/TICKET-006-evaluation-summary.md) — strengths, risks, downstream impact, artifact versioning.
- [TICKET-006 release checklist](cursor/project/notes/TICKET-006-release-checklist.md) — env vars, health, smoke, rollback, tests.

## Data: row-level splits (TICKET-002)

After merging fake + job row-level CSVs (TICKET-007), build stratified **train / validation / test** files with `combined_text` and `risk_class`:

```bash
python scripts/combine_job_postings_rows.py
python scripts/build_row_level_merged_splits.py
```

Defaults write **`artifacts/data/processed/merged_train.csv`**, **`merged_val.csv`**, **`merged_test.csv`**, and **`merged_splits.summary.json`** (70% / 15% / 15%, stratified on `risk_class`, `random_state=42`). Use **`--job-only`** to split only `dataset_source == "job_rows"` rows.

**Warning:** Running this **replaces** any existing **`artifacts/data/processed/merged_*.csv`** at those paths. Older Phase 6 notebooks also used `merged_*.csv` names for **different** (D1+D2, binary `fraudulent`) tables — back up those files first if you still need them.

## Legacy DistilBERT-only and hybrid ensemble

Earlier versions of this repository documented DistilBERT-only checkpoints (`JOBSENTRY_MODEL_ARTIFACT_PATH`) and optional TF-IDF / Keras ensembles. The **current** `app` implements **only** the phase6 fused path above. Restoring DistilBERT-only serving in the same process is a separate change; the old env vars are kept in `.env.example` as comments for reference.

## FastAPI checklist

- [x] `requirements.txt` with FastAPI stack and torch/transformers for fused inference  
- [x] `app/` package with `main.py`, fused model loader, and tests  
- [x] `GET /`, `GET /health`, `POST /predict`  
- [ ] Virtual environment and deps installed locally  
- [ ] `JOBSENTRY_PHASE6_FUSED_DIR` pointed at real artifacts in your environment  
