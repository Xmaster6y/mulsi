"""Preprocessing functions for MULSI.
"""

from dataclasses import dataclass

import einops
import torch
from PIL import Image
from tensordict import TensorDict
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPImageProcessor, PreTrainedTokenizer, image_utils


@dataclass
class DiffCLIPImageProcessor:
    """Differentiable CLIP processor.

    TODO: Temporary solution. Should be inheriting from the processor.
    """

    processor: CLIPImageProcessor

    def preprocess(self, image) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = pil_to_tensor(image).float().unsqueeze(0)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if self.processor.resample != 3:
            raise NotImplementedError
        size = tuple(self.processor.crop_size.values())
        im_proc = torch.nn.functional.interpolate(
            image, size=size, antialias=True, mode="bicubic"
        )

        im_proc = im_proc * self.processor.rescale_factor

        mean = torch.Tensor(self.processor.image_mean)
        std = torch.Tensor(self.processor.image_std)

        im_proc = einops.rearrange(
            (einops.rearrange(im_proc, "b c h w -> b h w c") - mean) / std,
            "b h w c -> b c h w",
        )
        return im_proc

    def __call__(self, images, **kwargs):
        kwargs["return_tensors"] = "pt"
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
        else:
            images = image_utils.make_list_of_images(images)
        return TensorDict(
            {
                "pixel_values": [
                    self.preprocess(image).squeeze(0) for image in images
                ]
            },
            batch_size=len(images),
        )


@dataclass
class TdTokenizer:
    """Tokenizer for Tensordict.

    TODO: Temporary solution. Should be inheriting from the tokenizer.
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, *args, **kwargs):
        kwargs["return_tensors"] = "pt"
        inputs = self.tokenizer(*args, **kwargs)
        return TensorDict(
            {k: v for k, v in inputs.items()},
            batch_size=inputs["input_ids"].shape[0],
        )


@dataclass
class DiffCLIPProcessor:
    """Differentiable CLIP processor."""

    image_processor: DiffCLIPImageProcessor
    text_processor: TdTokenizer

    def __call__(self, images, text=None, **kwargs):
        outputs = TensorDict()
        outputs.update(self.image_processor(images, **kwargs))
        outputs.update(self.text_processor(text, **kwargs))
        return outputs
