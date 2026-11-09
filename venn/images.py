"""Pinned uint8 image batches with ordered field concatenation.

EmbeddingGemma 2's vision tower and image processor are the Gemma 4 classes in transformers
(``Gemma4VisionModel`` and ``Gemma4ImageProcessor``), so the norm and resize helpers used here are
imported from ``transformers.models.gemma4``.
"""

import numpy as np

from venn.profiles import IMAGE_TAGS, VENN_1


def _norm_square(values):
    return values.float().pow(2)


def _norm_scale(values, inverse, weight):
    normalized = values.float() * inverse
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized.type_as(values)


def _compile_norms(reader):
    import types
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    square = torch.compile(_norm_square, fullgraph=True, dynamic=True)
    scale = torch.compile(_norm_scale, fullgraph=True, dynamic=True)

    def forward(module, values):
        # Keep the original reduction and pow: fusing them changes bf16 embeddings.
        variance = square(values).mean(-1, keepdim=True) + module.eps
        inverse = torch.pow(variance, -0.5)
        return scale(values, inverse, module.weight if module.with_scale else None)

    for module in reader.modules():
        if isinstance(module, Gemma4RMSNorm) and not getattr(module, "_venn_compiled_norm", False):
            module.forward = types.MethodType(forward, module)
            module._venn_compiled_norm = True


class ImageBatchReader:
    """Encodes uint8 RGB image batches on a CUDA device with the pinned bf16 reader.

    Pass ``reader`` to reuse a loaded ``SentenceTransformer``; ``compile_norms`` swaps in compiled RMSNorm
    kernels. Raises ``ValueError`` without CUDA or a bf16 reader.
    """
    def __init__(self, device="cuda:0", reader=None, compile_norms=False):
        import torch

        if reader is None:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer

            snapshot = snapshot_download(VENN_1.reader, revision=VENN_1.revision,
                                         local_files_only=True)
            reader = SentenceTransformer(snapshot, device=device,
                                         model_kwargs={"dtype": torch.bfloat16})
        self.reader = reader.eval()
        self.device = reader.device
        if next(reader.parameters()).dtype != torch.bfloat16 or self.device.type != "cuda":
            raise ValueError("The uint8 GPU image path requires CUDA and the frozen bf16 reader")
        if compile_norms:
            _compile_norms(self.reader)
        self.processor = reader._first_module().processor.image_processor
        self.host, self.templates = {}, {}

    def encode_frames(self, images, representation="image-thirds-70"):
        """Return unit-norm float32 vectors for an ``(n, height, width, 3)`` uint8 batch.

        ``image-thirds-70`` embeds the left, center and right thirds and concatenates them (2304 dims);
        ``image-70`` and ``image-280`` embed the whole frame at 70 or 280 soft tokens (768 dims). Raises
        ``ValueError`` for an unknown representation or batches outside 1..128 images, 4M pixels or 64 MB.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvF
        from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

        if representation not in IMAGE_TAGS:
            raise ValueError("Unknown image representation")
        images = np.asarray(images)
        if (images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3
                or not 1 <= len(images) <= 128 or min(images.shape[1:3]) < 3
                or images.shape[1] * images.shape[2] > 4000000
                or images.nbytes > 64000000):
            raise ValueError("Provide 1..128 RGB uint8 images, at most 4M pixels/image and 64MB/batch")
        count, height, width, _ = images.shape
        thirds = representation == "image-thirds-70"
        parts = ((0, width // 3), (width // 3, 2 * width // 3), (2 * width // 3, width)) if thirds else ((0, width),)
        # Match the historical 320-column split exactly: 106, 107, 107.
        if thirds and width == 320:
            parts = ((0, 106), (106, 213), (213, 320))
        budget = 280 if representation == "image-280" else 70
        processor = self.processor
        maximum = budget * processor.pooling_kernel_size ** 2
        if int(processor.resample) != 3:
            raise ValueError("Unexpected reader resize interpolation")
        template_key = (height, width, representation)
        with torch.inference_mode():
            if template_key not in self.templates:
                template = self.reader.preprocess(
                    [{"text": "<|image|>", "image": [Image.new("RGB", (right - left, height))]}
                     for left, right in parts],
                    processing_kwargs={"image": {"max_soft_tokens": budget}})
                self.templates[template_key] = {
                    name: value.to(self.device) if torch.is_tensor(value) else value
                    for name, value in template.items() if name != "pixel_values"}
            shape = images.shape
            if shape not in self.host:
                if len(self.host) >= 4:
                    self.host.clear()
                self.host[shape] = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
            self.host[shape].copy_(torch.from_numpy(np.ascontiguousarray(images)))
            screens = self.host[shape].to(self.device, non_blocking=True).permute(0, 3, 1, 2)
            batches = []
            for left, right in parts:
                h, w = get_aspect_ratio_preserving_size(
                    height, right - left, processor.patch_size, maximum, processor.pooling_kernel_size)
                values = tvF.resize(screens[:, :, :, left:right], [h, w],
                                   interpolation=InterpolationMode.BICUBIC, antialias=True).float()
                if processor.do_rescale:
                    values = values * processor.rescale_factor
                if processor.do_normalize:
                    mean = torch.as_tensor(processor.image_mean, device=self.device).reshape(1, 3, 1, 1)
                    std = torch.as_tensor(processor.image_std, device=self.device).reshape(1, 3, 1, 1)
                    values = (values - mean) / std
                patch = processor.patch_size
                values = values.reshape(count, 3, h // patch, patch, w // patch, patch)
                values = values.permute(0, 2, 4, 3, 5, 1).reshape(count, -1, patch * patch * 3)
                batches.append(F.pad(values, (0, 0, 0, maximum - values.shape[1])))
            pixels = torch.stack(batches, dim=1).flatten(0, 1)
            features = {
                name: value.repeat(count, *([1] * (value.ndim - 1))) if torch.is_tensor(value) else value
                for name, value in self.templates[template_key].items()}
            features["pixel_values"] = pixels
            values = self.reader(features)["sentence_embedding"].float()
            values = F.normalize(values, dim=1).reshape(count, -1)
            return F.normalize(values, dim=1).cpu().numpy()
