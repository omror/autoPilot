"""Testler icin ortak ayarlar."""
import pytest


@pytest.fixture(autouse=True)
def llm_kapali(monkeypatch):
    """Testler LLM'e cikmasin: deterministik ve ucretsiz kalsinlar.

    Kural-tabanli yol her zaman calisir, testler onu dogrular.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
