# -*- coding: utf-8 -*-

import pytest
from unittest.mock import patch, MagicMock

from ipychat.models import (
    get_current_model,
    get_model_by_name,
    get_models_by_provider,
    get_ollama_models,
    ModelConfig,
)


def test_get_model_by_name():
    model = get_model_by_name("gpt-4o")
    assert model.name == "gpt-4o"
    assert model.provider == "openai"

    with pytest.raises(ValueError):
        get_model_by_name("nonexistent-model")


def test_get_models_by_provider():
    openai_models = get_models_by_provider("openai")
    assert all(m.provider == "openai" for m in openai_models)
    assert len(openai_models) > 0

    anthropic_models = get_models_by_provider("anthropic")
    assert all(m.provider == "anthropic" for m in anthropic_models)
    assert len(anthropic_models) > 0


def test_get_current_model(mock_config, monkeypatch):
    monkeypatch.setattr("ipychat.models.load_config", lambda: mock_config)
    model = get_current_model()
    assert model.name == "gpt-4o"
    assert model.provider == "openai"


def test_get_ollama_models_success():
    mock_model = MagicMock()
    mock_model.model = "llama3"

    with patch("ipychat.models.ollama.list") as mock_list:
        mock_list.return_value.models = [mock_model]

        result = get_ollama_models()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == ModelConfig(name="llama3", provider="local")


def test_get_ollama_models_exception():
    with patch(
        "ipychat.models.ollama.list", side_effect=Exception("Ollama not available")
    ):
        result = get_ollama_models()
        assert result == []
