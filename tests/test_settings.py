import importlib

import app.config.settings as settings_module


def test_settings_load_project_dotenv(monkeypatch):
    monkeypatch.delenv("ENABLE_MILVUS", raising=False)

    reloaded = importlib.reload(settings_module)

    assert reloaded.settings.enable_milvus is True


def test_system_environment_overrides_project_dotenv(monkeypatch):
    monkeypatch.setenv("ENABLE_MILVUS", "false")

    reloaded = importlib.reload(settings_module)

    assert reloaded.settings.enable_milvus is False
