import os
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

from utils.image_utils import img_to_patch, calculate_top_k_indices

DIM_H = 3840
DIM_V = 2160


class EvalDataset(Dataset):
    def __init__(self, patch_size, top_k_patches, patch_selection_criterion, folder_path,
                 use_dct, transform_eval):
        super().__init__()

        self.patch_size = patch_size
        self.top_k_patches = top_k_patches
        self.patch_selection_criterion = patch_selection_criterion
        self.use_dct = use_dct
        self.filenames = []

        self.transform_eval = transform_eval

        folder_path = Path(folder_path)
        for frame in sorted(folder_path.iterdir()):

            self.filenames.append(frame)

        self.top_k_indices = self._compute_top_k_indices()

        print(f"Loaded {len(self.filenames)} images")

    def _compute_top_k_indices(self):

        top_k_indices = dict()
        if len(self.filenames) > 0:
            top_k_indices = calculate_top_k_indices(self.filenames, self.patch_size, 144,
                                                            self.patch_selection_criterion)
            
        return top_k_indices

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        image = np.array(Image.open(self.filenames[idx]), dtype=np.float32)
        image = image / 255.0

        image = self.transform_eval(image=image)["image"]
        if image.size(1) != DIM_V or image.size(2) != DIM_H:
            print(f"Cropping image {self.filenames[idx]} because it is not 4k")
            image = transforms.functional.crop(image, 0, 0, DIM_V, DIM_H)

        top_k = self.top_k_indices[self.filenames[idx]][:self.top_k_patches]
        image = image.unsqueeze(0)
        patches = img_to_patch(image, self.patch_size, flatten_channels=False)
        # Patches shape: torch.Size([1, 144, 3, 240, 240])
        patches = patches.squeeze(0)
        best_patches = patches[top_k, :]

        return best_patches
