"""
Fixtures for tests.
"""

import pytest
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

from mulsi import DiffCLIPImageProcessor, TdTokenizer


@pytest.fixture(scope="session")
def text_model():
    """Return a model."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    yield model


@pytest.fixture(scope="session")
def td_tokenizer():
    """Return a tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    td_tokenizer = TdTokenizer(tokenizer)
    yield td_tokenizer


@pytest.fixture(scope="session")
def clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    yield model


@pytest.fixture(scope="session")
def clip_processor():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = DiffCLIPImageProcessor(processor.image_processor)
    yield clip_processor


@pytest.fixture(scope="session")
def image():
    """Return an image."""
    image = Image.open("assets/orange.jpg")
    yield image
