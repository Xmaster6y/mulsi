"""Test of the processor module."""

import pytest


class TestDiffClipProcessor:
    """Test the differentiable CLIP processor."""

    @pytest.mark.xfail
    def test_diff_clip_processor(self, clip_processor, image):
        """Test the differentiable CLIP processor."""
        im_proc = clip_processor.processor(images=image, return_tensors="pt", padding=True)
        diff_im_proc = clip_processor(image, return_tensors="pt", padding=True)
        assert diff_im_proc["pixel_values"].allclose(im_proc["pixel_values"], atol=1e-5)
