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

"""IO utilities."""

import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import tqdm


class CMMDDataset(Dataset):
    def __init__(self, path, reshape_to, max_count=-1):
        self.path = path
        self.reshape_to = reshape_to

        self.max_count = max_count
        img_path_list = self._get_image_list()
        if max_count > 0:
            img_path_list = img_path_list[:max_count]
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def _get_image_list(self):
        ext_list = ["png", "jpg", "jpeg"]
        image_list = []
        for ext in ext_list:
            image_list.extend(glob.glob(f"{self.path}/*{ext}"))
            image_list.extend(glob.glob(f"{self.path}/*.{ext.upper()}"))
        # Sort the list to ensure a deterministic output.
        image_list.sort()
        return image_list

    def _center_crop_and_resize(self, im, size):
        w, h = im.size
        l = min(w, h)
        top = (h - l) // 2
        left = (w - l) // 2
        box = (left, top, left + l, top + l)
        im = im.crop(box)
        # Note that the following performs anti-aliasing as well.
        # im = torch.nn.functional.interpolate(im, size=(size, size), mode="bicubic", align_corners=False, antialias=True)

        return im.resize((size, size), resample=Image.BICUBIC)  # pytype: disable=module-attr

    def _read_image(self, path, size):
        im = Image.open(path)
        if size > 0:
            im = self._center_crop_and_resize(im, size)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]

        x = self._read_image(img_path, self.reshape_to)
        if x.ndim == 3:
            return x
        elif x.ndim == 2:
            # Convert grayscale to RGB by duplicating the channel dimension.
            return np.tile(x[Ellipsis, np.newaxis], (1, 1, 3))


def compute_embeddings_for_dir(
    img_dir,
    embedding_model,
    batch_size,
    max_count=-1,
):
    """Computes embeddings for the images in the given directory.

    This drops the remainder of the images after batching with the provided
    batch_size to enable efficient computation on TPUs. This usually does not
    affect results assuming we have a large number of images in the directory.

    Args:
      img_dir: Directory containing .jpg or .png image files.
      embedding_model: The embedding model to use.
      batch_size: Batch size for the embedding model inference.
      max_count: Max number of images in the directory to use.

    Returns:
      Computed embeddings of shape (num_images, embedding_dim).
    """
    dataset = CMMDDataset(img_dir, reshape_to=embedding_model.input_image_size, max_count=max_count)
    count = len(dataset)
    print(f"Calculating embeddings for {count} images from {img_dir}.")

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16)

    all_embs = []
    for batch in tqdm.tqdm(dataloader, total=count // batch_size):
        image_batch = batch.numpy()

        # Normalize to the [0, 1] range.
        image_batch = image_batch / 255.0

        if np.min(image_batch) < 0 or np.max(image_batch) > 1:
            raise ValueError(
                "Image values are expected to be in [0, 1]. Found:" f" [{np.min(image_batch)}, {np.max(image_batch)}]."
            )

        # Compute the embeddings using a pmapped function.
        embs = np.asarray(
            embedding_model.embed(image_batch)
        )  # The output has shape (num_devices, batch_size, embedding_dim).
        all_embs.append(embs)

    all_embs = np.concatenate(all_embs, axis=0)

    return all_embs
