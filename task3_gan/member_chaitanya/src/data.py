"""Unpaired photo / Monet datasets.

Domain A = photos (photo_jpg), domain B = Monet paintings (monet_jpg).
One epoch = one pass over the photos (shuffled); each photo is paired with a
Monet painting drawn uniformly at random, so pairs are never fixed.
Training augmentation (both domains, independently): resize to load_size,
random crop to crop_size, random horizontal flip. No colour jitter, rotation
or vertical flip.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    return files


def train_transform(load_size: int, crop_size: int, hflip: bool) -> T.Compose:
    ops = [T.Resize((load_size, load_size), interpolation=T.InterpolationMode.BICUBIC), T.RandomCrop(crop_size)]
    if hflip:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]  # -> [-1, 1] to match tanh output
    return T.Compose(ops)


def eval_transform(size: int) -> T.Compose:
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                      T.Normalize([0.5] * 3, [0.5] * 3)])


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class UnpairedDataset(Dataset):
    def __init__(self, photo_dir: Path, monet_dir: Path, transform_a, transform_b):
        self.photos, self.monets = list_images(photo_dir), list_images(monet_dir)
        self.tf_a, self.tf_b = transform_a, transform_b

    def __len__(self) -> int:
        return len(self.photos)

    def __getitem__(self, i: int):
        # torch RNG: DataLoader seeds each worker differently, so draws are not repeated across workers
        monet = self.monets[int(torch.randint(len(self.monets), (1,)).item())]
        return self.tf_a(load_rgb(self.photos[i])), self.tf_b(load_rgb(monet))


class FolderDataset(Dataset):
    """Eval-time images of one domain, returned with their file names."""

    def __init__(self, folder: Path, transform, files: list[Path] | None = None):
        self.files = files if files is not None else list_images(folder)
        self.tf = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        return self.tf(load_rgb(self.files[i])), self.files[i].name
