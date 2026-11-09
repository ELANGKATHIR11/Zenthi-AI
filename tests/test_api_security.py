"""
Tests for API Security and Endpoint Validation.
"""

import os
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_api_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "engine" in data

def test_api_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "hardware" in data
    assert "ram_total_gb" in data["hardware"]

def test_api_upload_disallowed_extension(client):
    # Try uploading a .sh script
    files = {"file": ("malicious.sh", b"echo 'bad'", "application/x-sh")}
    res = client.post("/upload", files=files)
    assert res.status_code == 400
    assert "extension" in res.json()["detail"].lower()

def test_api_upload_path_traversal(client):
    # Try uploading with ../ traversal in filename
    files = {"file": ("../../etc_passwd.txt", b"sample content", "text/plain")}
    res = client.post("/upload", files=files)
    # Either succeeds saving to sanitized basename or rejects traversal cleanly
    assert res.status_code in (200, 400)
    if res.status_code == 200:
        assert res.json()["filename"] == "etc_passwd.txt"
