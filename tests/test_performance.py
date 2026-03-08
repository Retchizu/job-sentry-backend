"""Performance tests — run with a real model for meaningful latency numbers.

With the dummy model (used in CI), these tests verify the pipeline runs
end-to-end but latency values are not representative of production.
"""

import time

from tests.conftest import SAMPLE_JOB_POST


TARGET_LATENCY_SECONDS = 2.0


class TestPerformance:
    def test_single_predict_latency(self, client):
        start = time.perf_counter()
        resp = client.post("/predict", json=SAMPLE_JOB_POST)
        elapsed = time.perf_counter() - start

        assert resp.status_code == 200
        assert elapsed < TARGET_LATENCY_SECONDS, (
            f"Single prediction took {elapsed:.2f}s, target is <{TARGET_LATENCY_SECONDS}s"
        )

    def test_batch_predict_latency(self, client):
        batch = {"job_posts": [SAMPLE_JOB_POST] * 5}
        start = time.perf_counter()
        resp = client.post("/batch-predict", json=batch)
        elapsed = time.perf_counter() - start

        assert resp.status_code == 200
        per_item = elapsed / 5
        assert per_item < TARGET_LATENCY_SECONDS, (
            f"Per-item latency {per_item:.2f}s, target is <{TARGET_LATENCY_SECONDS}s"
        )
