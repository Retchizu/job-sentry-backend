from tests.conftest import SAMPLE_JOB_POST, SCAM_JOB_POST


class TestPredictEndpoint:
    def test_valid_post_returns_200(self, client):
        resp = client.post("/predict", json=SAMPLE_JOB_POST)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("scam", "legitimate")
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["scam_probability"] <= 1
        assert isinstance(data["warning_signals"], list)

    def test_scam_post_returns_200(self, client):
        resp = client.post("/predict", json=SCAM_JOB_POST)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("scam", "legitimate")
        assert isinstance(data["warning_signals"], list)

    def test_missing_required_field_returns_422(self, client):
        resp = client.post("/predict", json={"job_title": "Engineer"})
        assert resp.status_code == 422

    def test_empty_title_returns_422(self, client):
        payload = {**SAMPLE_JOB_POST, "job_title": ""}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_model_not_loaded_returns_503(self, client_no_model):
        resp = client_no_model.post("/predict", json=SAMPLE_JOB_POST)
        assert resp.status_code == 503


class TestBatchPredictEndpoint:
    def test_valid_batch_returns_200(self, client):
        body = {"job_posts": [SAMPLE_JOB_POST, SCAM_JOB_POST]}
        resp = client.post("/batch-predict", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2
        for result in data["results"]:
            assert result["prediction"] in ("scam", "legitimate")
            assert 0 <= result["confidence"] <= 1

    def test_empty_list_returns_422(self, client):
        resp = client.post("/batch-predict", json={"job_posts": []})
        assert resp.status_code == 422

    def test_model_not_loaded_returns_503(self, client_no_model):
        body = {"job_posts": [SAMPLE_JOB_POST]}
        resp = client_no_model.post("/batch-predict", json=body)
        assert resp.status_code == 503


class TestHealthEndpoint:
    def test_health_with_model(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_without_model(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Job Sentry API" in resp.json()["message"]
