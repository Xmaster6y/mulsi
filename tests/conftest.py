"""
Fixtures for tests.
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from mulsi import TdTokenizer


@pytest.fixture(scope="session")
def model():
    """
    Return a model.
    """
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    yield model


@pytest.fixture(scope="session")
def td_tokenizer():
    """
    Return a tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    td_tokenizer = TdTokenizer(tokenizer)
    yield td_tokenizer
