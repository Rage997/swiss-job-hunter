"""Tests for environment-driven settings."""

import importlib

import pytest
from pydantic_settings import SettingsError

@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch, tmp_path):
    """A developer's .env must not merge extra keys into the test configuration."""
    monkeypatch.chdir(tmp_path)



def load_settings_class(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    settings_module = importlib.import_module("config.settings")
    return importlib.reload(settings_module).Settings


def test_keyword_presets_can_be_loaded_from_env(monkeypatch):
    monkeypatch.setenv(
        "KEYWORD_PRESETS",
        '{"devops": ["DevOps engineer", "SRE"], "backend": ["Python engineer"]}',
    )
    Settings = load_settings_class(monkeypatch)

    settings = Settings()

    assert settings.keyword_presets == {
        "devops": ["DevOps engineer", "SRE"],
        "backend": ["Python engineer"],
    }


def test_keyword_presets_invalid_json_is_rejected(monkeypatch):
    monkeypatch.setenv("KEYWORD_PRESETS", "not-json")

    with pytest.raises(SettingsError, match='error parsing value for field "keyword_presets"'):
        load_settings_class(monkeypatch)
