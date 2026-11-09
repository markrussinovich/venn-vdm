import copy

import numpy as np
import pytest


@pytest.mark.parametrize("width,representation", [(106, "image-thirds-70"), (320, "image-thirds-70"),
                                                 (106, "image-70"), (106, "image-280")])
def test_crop_metadata_tracks_each_grid_and_frame_on_cpu(monkeypatch, width, representation):
    import torch
    pytest.importorskip("transformers")
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

    from venn.images import ImageBatchReader

    processor = Gemma4ImageProcessor()
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: real_empty(
        *args, **{k: v for k, v in kwargs.items() if k != "pin_memory"}))

    class Reader:
        def __init__(self):
            self.widths, self.valid = [], []

        def preprocess(self, inputs, processing_kwargs):
            images = [item["image"][0] for item in inputs]
            self.widths.append([image.width for image in images])
            features = processor(images=images, return_tensors="pt", **processing_kwargs["image"])
            lengths = features["num_soft_tokens_per_image"] + 2
            features["input_ids"] = torch.arange(int(lengths.max())).expand(len(images), -1)
            features["attention_mask"] = features["input_ids"] < lengths[:, None]
            return features

        def __call__(self, features):
            positions, pixels = features["image_position_ids"], features["pixel_values"]
            valid = (positions[..., 0] >= 0).sum(1)
            self.valid.append(valid.tolist())
            assert len(positions) == len(pixels)
            assert torch.equal(valid // processor.pooling_kernel_size ** 2,
                               features["num_soft_tokens_per_image"])
            assert torch.equal(features["attention_mask"].sum(1), features["num_soft_tokens_per_image"] + 2)
            for row, count in enumerate(valid.tolist()):
                assert not pixels[row, count:].count_nonzero()
            return {"sentence_embedding": torch.stack(
                [pixels.mean((1, 2)), valid.float() / pixels.shape[1], torch.ones(len(pixels))], 1)}

    reader = Reader()
    batch = ImageBatchReader.__new__(ImageBatchReader)
    batch.reader, batch.processor, batch.device = reader, processor, torch.device("cpu")
    batch.host, batch.templates = {}, {}
    images = np.random.default_rng(0).integers(0, 256, (2, 100, width, 3), dtype=np.uint8)
    vectors = batch.encode_frames(images, representation)
    individual = np.concatenate([batch.encode_frames(image[None], representation) for image in images])
    np.testing.assert_allclose(vectors, individual, rtol=1e-6, atol=1e-7)
    assert len(reader.widths) == 1  # repeat requests reuse the complete crop template
    if representation == "image-thirds-70" and width == 106:
        assert reader.widths[0] == [35, 35, 36]
        assert reader.valid[0] == [504, 504, 585, 504, 504, 585]
    elif representation == "image-thirds-70":
        assert reader.widths[0] == [106, 107, 107]


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("with_scale", [False, True])
def test_compiled_norm_keeps_reference_rounding_and_weights(dtype_name, with_scale):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA normalization compilation requires a GPU")
    Gemma4RMSNorm = pytest.importorskip("transformers.models.gemma4.modeling_gemma4").Gemma4RMSNorm
    from venn.images import _compile_norms

    dtype = getattr(torch, dtype_name)
    norm = Gemma4RMSNorm(128, with_scale=with_scale).to(device="cuda", dtype=dtype).eval()
    generator = torch.Generator(device="cuda").manual_seed(7)
    if with_scale:
        with torch.no_grad():
            norm.weight.copy_(torch.randn(128, device="cuda", dtype=dtype, generator=generator))
    reference = copy.deepcopy(norm)
    _compile_norms(norm)
    with torch.inference_mode():
        for count in (8, 17):
            values = torch.randn(count, 7, 128, device="cuda", dtype=dtype, generator=generator)
            torch.testing.assert_close(norm(values), reference(values), rtol=0, atol=0)
    for name, value in norm.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
