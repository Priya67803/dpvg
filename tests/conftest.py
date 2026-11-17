import pytest
from app.llm_backends import OllamaBackend
from tests.fake_ollama import FakeOllama


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    """Route every OllamaBackend the app creates to an in-process fake server (no network, no GPU)."""
    fake = FakeOllama()
    real_init = OllamaBackend.__init__

    def init(self, *a, **kw):
        kw.setdefault("transport", fake.transport())
        real_init(self, *a, **kw)
    monkeypatch.setattr(OllamaBackend, "__init__", init)
    return fake
