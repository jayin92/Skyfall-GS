# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding models used in the CMMD calculation."""

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
import numpy as np

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CUDA_AVAILABLE = torch.cuda.is_available()


def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images


class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)

        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        if _CUDA_AVAILABLE:
            self._model = self._model.cuda()

        self.input_image_size = self.image_processor.crop_size["height"]

    @torch.no_grad()
    def embed(self, images):
        """Computes CLIP embeddings for the given images.

        Args:
          images: An image array of shape (batch_size, height, width, 3). Values are
            in range [0, 1].

        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """

        images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        image_embs = self._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs
