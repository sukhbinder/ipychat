# -*- coding: utf-8 -*-

from unittest.mock import Mock, patch, MagicMock

import pytest

from ipychat.providers import get_provider
from ipychat.providers.anthropic import AnthropicProvider
from ipychat.providers.google import GoogleProvider
from ipychat.providers.openai import OpenAIProvider
from ipychat.providers.local import OllamaProvider


def test_get_provider(mock_config):
    provider = get_provider(mock_config)
    assert isinstance(provider, OpenAIProvider)

    mock_config["current"]["provider"] = "anthropic"
    provider = get_provider(mock_config)
    assert isinstance(provider, AnthropicProvider)

    mock_config["current"]["provider"] = "invalid"
    with pytest.raises(ValueError):
        get_provider(mock_config)


def test_openai_provider_stream_chat(mock_config):
    provider = OpenAIProvider(mock_config)
    provider.initialize_client()

    mock_response = Mock()
    mock_response.choices = [Mock(delta=Mock(content="test response"))]

    with patch.object(provider.client.chat.completions, "create") as mock_create:
        mock_create.return_value = [mock_response]

        responses = list(provider.stream_chat("system prompt", "user content"))
        assert responses == ["test response"]


def test_anthropic_provider_stream_chat(mock_config):
    provider = AnthropicProvider(mock_config)
    provider.initialize_client()

    mock_chunk = Mock()
    mock_chunk.type = "content_block_delta"
    mock_chunk.delta = Mock(text="test response")

    with patch.object(provider.client.messages, "create") as mock_create:
        mock_create.return_value = [mock_chunk]

        responses = list(provider.stream_chat("system prompt", "user content"))
        assert responses == ["test response"]


def test_openai_provider_missing_api_key(mock_config):
    # Ensure console is properly mocked
    mock_console = Mock()
    mock_config["console"] = mock_console

    # Test empty API key
    mock_config["openai"] = {"api_key": ""}
    provider = OpenAIProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]OPENAI_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )

    # Reset mock
    mock_console.print.reset_mock()

    # Test missing API key
    mock_config["openai"] = {}
    provider = OpenAIProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]OPENAI_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )

    # Reset mock
    mock_console.print.reset_mock()

    # Test missing openai config section
    mock_config.pop("openai", None)
    provider = OpenAIProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]OPENAI_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )


def test_anthropic_provider_missing_api_key(mock_config):
    # Ensure console is properly mocked
    mock_console = Mock()
    mock_config["console"] = mock_console

    # Test empty API key
    mock_config["anthropic"] = {"api_key": ""}
    provider = AnthropicProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]ANTHROPIC_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )

    # Reset mock
    mock_console.print.reset_mock()

    # Test missing API key
    mock_config["anthropic"] = {}
    provider = AnthropicProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]ANTHROPIC_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )

    # Reset mock
    mock_console.print.reset_mock()

    # Test missing anthropic config section
    mock_config.pop("anthropic", None)
    provider = AnthropicProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]ANTHROPIC_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )


def test_google_provider_missing_api_key(mock_config):
    # Ensure console is properly mocked
    mock_console = Mock()
    mock_config["console"] = mock_console

    # Test empty API key
    mock_config["google"] = {"api_key": ""}
    provider = GoogleProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]GOOGLE_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )

    # Reset mock
    mock_console.print.reset_mock()

    # Test missing API key
    mock_config["google"] = {}
    provider = GoogleProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]GOOGLE_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )

    # Reset mock
    mock_console.print.reset_mock()

    # Test missing anthropic config section
    mock_config.pop("google", None)
    provider = GoogleProvider(mock_config)
    provider.console = mock_console
    provider.initialize_client()
    assert provider.client is None
    mock_console.print.assert_called_once_with(
        "[red]Set [bold]GOOGLE_API_KEY[/bold] in your environment, or run [bold]ipychat config[/bold].[/red]"
    )


@pytest.fixture
def mock_config_ollama():
    return {"current": {"model": "llama3"}}


def test_stream_chat_yields_content(mock_config_ollama):
    provider = OllamaProvider(mock_config_ollama)
    provider.model = "llama3"

    # Mock client.chat stream
    mock_response = [
        {"message": {"content": "Hello"}},
        {"message": {"content": " world!"}},
    ]

    provider.client = MagicMock()
    provider.client.chat.return_value = mock_response

    gen = provider.stream_chat("You are a bot", "Say something")
    result = list(gen)

    assert result == ["Hello", " world!"]
    provider.client.chat.assert_called_once()


def test_stream_chat_ignores_keyerror(mock_config_ollama):
    provider = OllamaProvider(mock_config_ollama)
    provider.model = "llama3"

    mock_response = [
        {"no_message": "oops"},  # Should be ignored due to KeyError
        {"message": {"content": "Recovered"}},
    ]

    provider.client = MagicMock()
    provider.client.chat.return_value = mock_response

    result = list(provider.stream_chat("You are a bot", "Continue"))
    assert result == ["Recovered"]
