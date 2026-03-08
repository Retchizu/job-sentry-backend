import pytest
import torch
from fastapi.testclient import TestClient
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from app.main import app
from app.model import ScamDetectionModel


@pytest.fixture(scope="session")
def dummy_model():
    """A randomly-initialised DistilBERT model for fast testing."""
    config = DistilBertConfig(
        vocab_size=30522,
        n_layers=2,
        n_heads=2,
        dim=128,
        hidden_dim=256,
        num_labels=2,
    )
    model = DistilBertForSequenceClassification(config)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    device = torch.device("cpu")
    return ScamDetectionModel(model=model, tokenizer=tokenizer, device=device)


@pytest.fixture()
def client(dummy_model):
    with TestClient(app) as c:
        app.state.model = dummy_model
        yield c


@pytest.fixture()
def client_no_model():
    with TestClient(app) as c:
        app.state.model = None
        yield c


SAMPLE_JOB_POST = {
    "job_title": "Senior Software Engineer",
    "job_desc": (
        "We are looking for an experienced software engineer to join our team. "
        "You will work on building scalable microservices using Python and AWS. "
        "Minimum 5 years of experience required."
    ),
    "company_profile": "TechCorp Inc is a leading software company founded in 2010.",
    "skills_desc": "Python, AWS, Docker, Kubernetes, PostgreSQL",
    "salary_range": "120000-160000",
    "employment_type": "Full-time",
}

SCAM_JOB_POST = {
    "job_title": "Easy Money Work From Home",
    "job_desc": "Earn 5000week immediate hiring contact now! No experience needed!!!",
    "company_profile": None,
    "skills_desc": None,
    "salary_range": None,
    "employment_type": None,
}
