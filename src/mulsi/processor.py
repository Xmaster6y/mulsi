"""Preprocessing functions for MULSI.
"""

import einops
import torch
from tensordict import TensorDict


class DiffClipProcessor:
    """Differentiable CLIP processor.

    TODO: Temporary solution. Should be inheriting from the processor.
    """

    def __init__(self, processor):
        self.processor = processor

    def preprocess(self, image):
        if image.dim() != 4:
            raise NotImplementedError
        if self.processor.resample != 3:
            raise NotImplementedError
        size = tuple(self.processor.crop_size.values())
        im_proc = torch.nn.functional.interpolate(
            image, size=size, antialias=True, mode="bilinear"
        )

        im_proc = im_proc * self.processor.rescale_factor

        mean = torch.Tensor(self.processor.image_mean)
        std = torch.Tensor(self.processor.image_std)

        im_proc = einops.rearrange(
            (einops.rearrange(im_proc, "b c h w -> b h w c") - mean) / std,
            "b h w c -> b c h w",
        )
        return im_proc

    def __call__(self, image):
        return self.preprocess(image)


class TdTokenizer:
    """Tokenizer for Tensordict.

    TODO: Temporary solution. Should be inheriting from the tokenizer.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        kwargs["return_tensors"] = "pt"
        inputs = self.tokenizer(*args, **kwargs)
        return TensorDict(
            {k: v for k, v in inputs.items()},
            batch_size=inputs["input_ids"].shape[0],
        )
