from pathlib import Path
import glob

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path("/kaggle/input/datasets/studentkrithika/idrid-dataset/A. Segmentation")

TRAIN_IMAGE_DIR = BASE_DIR / "1. Original Images" / "a. Training Set"
TRAIN_GROUNDTRUTH_DIR = BASE_DIR / "2. All Segmentation Groundtruths" / "a. Training Set"

TRAIN_MASK_DIRS = {
    "MA": TRAIN_GROUNDTRUTH_DIR / "1. Microaneurysms",
    "HE": TRAIN_GROUNDTRUTH_DIR / "2. Haemorrhages",
    "EX": TRAIN_GROUNDTRUTH_DIR / "3. Hard Exudates",
    "SE": TRAIN_GROUNDTRUTH_DIR / "4. Soft Exudates",
}

TARGET_SIZE = (512, 512)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_files(folder):
    """Return sorted image files from a folder."""
    folder = Path(folder)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]

    files = []
    for extension in extensions:
        files.extend(glob.glob(str(folder / extension)))

    return sorted(files)


def get_image_id(path):
    """Extract the shared ID used to match an original image with masks."""
    stem = Path(path).stem
    parts = stem.split("_")

    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return stem


def make_id_map(files):
    """Create a mapping from image ID to file path."""
    return {get_image_id(file): file for file in files}


def load_rgb_image(path):
    """Load an original fundus image as RGB."""
    image = Image.open(path).convert("RGB")
    return np.array(image)


def load_mask(path):
    """Load a lesion mask as grayscale."""
    mask = Image.open(path).convert("L")
    return np.array(mask)


def create_zero_mask(image):
    """Create an all-zero mask with the same height and width as image."""
    height, width = image.shape[:2]
    return np.zeros((height, width), dtype=np.uint8)


def resize_rgb_image(image, target_size=TARGET_SIZE):
    """Resize an RGB image using bilinear interpolation."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def resize_mask(mask, target_size=TARGET_SIZE):
    """Resize a mask using nearest-neighbor interpolation."""
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


def apply_clahe_rgb(image):
    """Apply CLAHE to an RGB image in LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def imagenet_normalize(image):
    """Apply ImageNet normalization to an RGB image."""
    image = image.astype(np.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def mask_to_binary(mask):
    """Convert a lesion mask to binary float format."""
    return (mask > 0).astype(np.float32)


class IDRiDDataset(Dataset):
    """PyTorch dataset for IDRiD lesion segmentation inputs."""

    def __init__(
        self,
        image_dir=TRAIN_IMAGE_DIR,
        mask_dirs=TRAIN_MASK_DIRS,
        target_size=TARGET_SIZE,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dirs = {name: Path(path) for name, path in mask_dirs.items()}
        self.target_size = target_size

        self.image_files = list_files(self.image_dir)
        self.image_map = make_id_map(self.image_files)
        self.image_ids = sorted(self.image_map.keys())

        self.mask_maps = {}
        for mask_name, mask_dir in self.mask_dirs.items():
            mask_files = list_files(mask_dir)
            self.mask_maps[mask_name] = make_id_map(mask_files)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = load_rgb_image(self.image_map[image_id])

        masks = {}
        for mask_name in ["MA", "HE", "EX", "SE"]:
            if image_id in self.mask_maps[mask_name]:
                mask = load_mask(self.mask_maps[mask_name][image_id])
            else:
                mask = create_zero_mask(image)

            masks[mask_name] = mask

        image = resize_rgb_image(image, self.target_size)
        image = apply_clahe_rgb(image)
        image = imagenet_normalize(image)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)

        masks = {
            mask_name: resize_mask(mask, self.target_size)
            for mask_name, mask in masks.items()
        }

        ma_mask = torch.tensor(mask_to_binary(masks["MA"]), dtype=torch.float32).unsqueeze(0)
        he_mask = torch.tensor(mask_to_binary(masks["HE"]), dtype=torch.float32).unsqueeze(0)
        ex_mask = torch.tensor(mask_to_binary(masks["EX"]), dtype=torch.float32).unsqueeze(0)
        se_mask = torch.tensor(mask_to_binary(masks["SE"]), dtype=torch.float32).unsqueeze(0)

        return {
            "image": image,
            "MA": ma_mask,
            "HE": he_mask,
            "EX": ex_mask,
            "SE": se_mask,
            "image_id": image_id,
        }


def create_train_dataloader(batch_size=4):
    """Create the training dataset and dataloader for IDRiD."""
    train_dataset = IDRiDDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dirs=TRAIN_MASK_DIRS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_dataset, train_loader
