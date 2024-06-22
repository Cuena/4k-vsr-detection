import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import pickle
from skimage.feature import graycomatrix, graycoprops
from torchvision import transforms
import torch
from tqdm import tqdm

DIM_H = 3840
DIM_V = 2160


def get_2d_dct(image: np.ndarray) -> np.ndarray:
    
    y, _, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))

    image = cv2.dct(np.float32(y))
    bias = 1e-4
    # Take the logarithm of DCT coefficients
    image = np.log(abs(image) + bias)
    return image

def get_2d_fft(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.fft.fft2(image)
    image = np.fft.fftshift(image)
    # image = np.log(abs(image) + 1)
    return image

def convert_image_to_3c_dct(image, normalize=False):

    y, cb, cr = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))

    image = cv2.dct(np.float32(y))
    bias = 1e-4
    # Take the logarithm of DCT coefficients
    image = np.log(abs(image) + bias)
    image = (image - image.min()) / (image.max() - image.min())
    # normalize cb and cr
    cb = (cb - cb.min()) / (cb.max() - cb.min())
    cr = (cr - cr.min()) / (cr.max() - cr.min())

    image = np.stack([image, cb, cr], axis=2)
    #normalize
    # image = (image - image.min()) / (image.max() - image.min())
    return image


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                        as a feature vector instead of a image grid.
    """
    patch_size = int(patch_size)
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
        # x = x.flatten(3, 4)  # [B, H'*W', C*p_H*p_W]
    return x


def get_top_k_patch_indices(patches, top_k_patches=3, patch_selection_criterion="contrast"):
    
    scores = [get_patch_complexity(p, patch_selection_criterion) for p in patches[:]]
    scores = np.array(scores, dtype=np.int16)
    # Getting indices of maximum values
    x = np.argsort(scores)[::-1][:top_k_patches]
    # print("Indices:",x)

    # Getting N maximum values
    # print("Values:",scores[x])
    return x


def get_patch_complexity(patch, patch_selection_criterion="contrast"):
    # Calculate GLCM properties
    # print("patch", patch.shape)
    scores = []
    glcm = graycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    dissimilarity = graycoprops(glcm, patch_selection_criterion)[0, 0]

    return dissimilarity


def get_patch_locations(image, patch_size):
    # Get image dimensions
    rows, cols = image.shape[:2]

    # Initialize list to store patch locations
    patch_locations = []

    # Iterate over the image with a sliding window
    for i in range(0, rows - patch_size[0] + 1, patch_size[0]):
        for j in range(0, cols - patch_size[1] + 1, patch_size[1]):
            patch_locations.append((i, j))

    return patch_locations


def calculate_top_k_indices(filenames, patch_size, top_k_patches, patch_selection_criterion):
        

    transform_patch = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.Grayscale(),
    ]
    )
    top_k_indices = {}
    for filename in tqdm(filenames):
        i1 = transform_patch(Image.open(filename))
        i1 = transforms.functional.crop(i1, 0, 0, DIM_V, DIM_H)
        i1 = i1.unsqueeze(0)

        ps = img_to_patch(i1, patch_size, flatten_channels=False).squeeze()
        top_k_indices[filename] = get_top_k_patch_indices(ps, top_k_patches=top_k_patches, patch_selection_criterion=patch_selection_criterion).copy()

    return top_k_indices


def batch_to_patches(x, indices=None, patch_size=240, top_k=3):
    patches = img_to_patch(x, patch_size, flatten_channels=False)
    
    new_batch = []
    for i in range(patches.size()[0]):
        pps = patches[i, :, :, :, :]
        best_patches = pps[indices[i], :]
        # print(best_patches.shape)
        new_batch.append(best_patches)

    i_patched = torch.stack(new_batch, dim=0)
    return i_patched


def get_resnet_features(model, x):
    features = []

    for name, module in model.named_children():
        if "avepool" in name or "fc" in name:
            continue
        if "layer" in name:
            x = module(x)
            features.append(x)
        else:
            x = module(x)

    # Apply spatial global average pooling to convert features to feature vectors
    pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    feature_vectors = [pool(f).squeeze() for f in features]
    feature_vectors = torch.concat(feature_vectors, dim=1)  # (48, 960)

    return feature_vectors
