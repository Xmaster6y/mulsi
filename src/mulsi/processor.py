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
class DiffClipProcessor:
    """Differentiable CLIP processor.

    TODO: Temporary solution. Should be inheriting from the processor.
    """

    processor: CLIPImageProcessor

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            image = pil_to_tensor(image).float().unsqueeze(0)
        if image.dim() != 4:
            raise NotImplementedError
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

    def __call__(self, images, return_tensors="pt", padding=True):
        if return_tensors != "pt":
            raise NotImplementedError
        images = image_utils.make_list_of_images(images)
        return TensorDict(
            {"pixel_values": [self.preprocess(image) for image in images]},
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
