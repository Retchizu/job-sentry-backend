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

## FastAPI setup checklist

- [x] `requirements.txt` with `fastapi`, `uvicorn[standard]`
- [x] App package `app/` with `main.py`
- [x] FastAPI app instance with title/version
- [x] Root route `GET /`
- [x] Health check `GET /health`
- [ ] Virtual environment created and deps installed (run locally)
- [ ] Optional: routers, Pydantic models, env config, tests
