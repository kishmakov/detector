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


# --- HTTP integration tests ---

def test_analyze_200(client):
    resp = client.post("/analyze", json={"text": "hello world from the chrome extension"})
    assert resp.status_code == 200
    assert resp.json() == {"word_count": 6}


def test_analyze_empty_text(client):
    resp = client.post("/analyze", json={"text": ""})
    assert resp.status_code == 200
    assert resp.json() == {"word_count": 0}


def test_analyze_missing_field(client):
    resp = client.post("/analyze", json={})
    assert resp.status_code == 422


def test_analyze_wrong_type(client):
    resp = client.post("/analyze", json={"text": 123})
    assert resp.status_code == 422


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
