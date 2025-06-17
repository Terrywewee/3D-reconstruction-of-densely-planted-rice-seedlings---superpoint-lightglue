
import os
import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from glob import glob
from typing import Any


class RiceSegmentationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_ratio: float = 0.2,
        transforms: A.Compose = None,
        use_split: bool = False,
    ):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "Image")
        self.mask_dir = os.path.join(data_dir, "Mask")

        self.images = sorted(glob(os.path.join(self.image_dir, "*.jpg")))
        self.masks = sorted(glob(os.path.join(self.mask_dir, "*.png")))
        self.transforms = transforms

        assert len(self.images) == len(self.masks), "Images and masks must match!"

        self.indices = list(range(len(self.images)))
        np.random.seed(42)
        np.random.shuffle(self.indices)

        if use_split:
            self.indices = self.indices
        else:
            split_idx = int(len(self.indices) * (1 - split_ratio))
            if split == "train":
                self.indices = self.indices[:split_idx]
            elif split == "val":
                self.indices = self.indices[split_idx:]
            else:
                raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.images[real_idx]
        mask_path = self.masks[real_idx]

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        mask = np.expand_dims(mask, axis=0)
        return img, mask


def get_transforms(train=True, size=(256, 256)):
    if train:
        return A.Compose([
            A.Resize(*size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(*size),
            ToTensorV2()
        ])


if __name__ == "__main__":
    data_path = "data_rice"
    train_ds = RiceSegmentationDataset(data_path, split="train", transforms=get_transforms(train=True))
    val_ds = RiceSegmentationDataset(data_path, split="val", transforms=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    for imgs, masks in train_loader:
        print("Image:", imgs.shape, "Mask:", masks.shape)
        break
