from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Welcome to SatQuery AI" in data["message"]
    assert data["health"] == "/api/health"

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "SatQuery AI"
    assert data["sih_problem_id"] == "SIH26167"
    assert data["organization"] == "ISRO"
    assert data["theme"] == "Space Technology"
    assert len(data["modalities_supported"]) > 0
    assert len(data["specialist_capabilities"]) > 0
