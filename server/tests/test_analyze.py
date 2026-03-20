import pytest

from app.services.text_analysis import count_words


# --- Service-layer unit tests ---

def test_normal_text():
    assert count_words("hello world") == 2


def test_empty_string():
    assert count_words("") == 0


def test_whitespace_only():
    assert count_words("   ") == 0


def test_single_word():
    assert count_words("hello") == 1


def test_multiple_spaces():
    assert count_words("hello   world") == 2


def test_newlines():
    assert count_words("hello\nworld\nfoo") == 3


# --- HTTP integration tests (run against both local and live client) ---

@pytest.mark.http
def test_health(any_client):
    resp = any_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.http
def test_analyze_200(any_client):
    resp = any_client.post("/analyze", json={"text": "hello world from the chrome extension"})
    assert resp.status_code == 200
    assert resp.json() == {"word_count": 6}


@pytest.mark.http
def test_analyze_empty_text(any_client):
    resp = any_client.post("/analyze", json={"text": ""})
    assert resp.status_code == 200
    assert resp.json() == {"word_count": 0}


@pytest.mark.http
def test_analyze_missing_field(any_client):
    resp = any_client.post("/analyze", json={})
    assert resp.status_code == 422


@pytest.mark.http
def test_analyze_wrong_type(any_client):
    resp = any_client.post("/analyze", json={"text": 123})
    assert resp.status_code == 422
