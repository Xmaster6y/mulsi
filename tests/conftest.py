"""
Fixtures for tests.
"""

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="session")
def model():
    """
    Return a model.
    """
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    yield model

def tokenizer():
    """
    Return a tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    yield tokenizer