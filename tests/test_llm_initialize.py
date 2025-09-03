import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from src.llm_initialize import LLMInitializer
from src.exception_handler import CustomException


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Set required environment variables for tests."""
    monkeypatch.setenv("AWS_ACCESS_KEY", "fake_aws_key")
    monkeypatch.setenv("AWS_SECRET_KEY", "fake_aws_secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("MODEL_ID_3", "fake-model-3")
    monkeypatch.setenv("MODEL_PROVIDER_3", "fake-provider-3")


def test_initialize_llm_model_3(monkeypatch):
    """Test model_num=3 initializes ChatBedrock with correct args."""
    mock_chat = MagicMock()
    monkeypatch.setattr("src.llm_initialize.ChatBedrock", lambda **kwargs: mock_chat)

    initializer = LLMInitializer()
    llm = initializer.initialize_llm(model_num=3, temperature=0.7, top_p=0.9, top_k=50)

    assert llm == mock_chat
    assert isinstance(initializer.llm, MagicMock)
    mock_chat.provider_stop_sequence_key_name_map["custom"] == "END_GENERATION"


def test_initialize_llm_model_4(monkeypatch):
    """Test model_num=4 initializes LLM from crewai."""
    mock_llm = MagicMock()
    monkeypatch.setattr("src.llm_initialize.LLM", lambda **kwargs: mock_llm)

    initializer = LLMInitializer()
    llm = initializer.initialize_llm(model_num=4, temperature=0.3)

    assert llm == mock_llm
    assert isinstance(initializer.llm, MagicMock)


def test_initialize_llm_default(monkeypatch):
    """Test default (else branch) initializes LLM."""
    mock_llm = MagicMock()
    monkeypatch.setattr("src.llm_initialize.LLM", lambda **kwargs: mock_llm)

    initializer = LLMInitializer()
    llm = initializer.initialize_llm(model_num=99)

    assert llm == mock_llm
    assert isinstance(initializer.llm, MagicMock)


def test_initialize_llm_raises(monkeypatch):
    """Test that initialization errors raise CustomException."""
    def broken_chatbedrock(**kwargs):
        raise RuntimeError("Fake failure")

    monkeypatch.setattr("src.llm_initialize.ChatBedrock", broken_chatbedrock)

    initializer = LLMInitializer()
    with pytest.raises(CustomException) as exc_info:
        initializer.initialize_llm(model_num=3)

    assert "Fake failure" in str(exc_info.value)
