import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def local_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def live_client():
    base_url = os.environ.get("DETECTOR_BASE_URL")
    if not base_url:
        pytest.skip("DETECTOR_BASE_URL not set")
    with httpx.Client(base_url=base_url) as c:
        yield c


@pytest.fixture(scope="session", params=["local", "live"])
def any_client(request, local_client):
    if request.param == "local":
        return local_client
    base_url = os.environ.get("DETECTOR_BASE_URL")
    if not base_url:
        pytest.skip("DETECTOR_BASE_URL not set")
    client = httpx.Client(base_url=base_url)
    request.addfinalizer(client.close)
    return client
