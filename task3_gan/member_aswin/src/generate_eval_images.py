"""Generate the missing Monet-to-photo images needed by Part 3 evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, base: int = 64, blocks: int = 9) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, base, 7, bias=False),
            nn.InstanceNorm2d(base, affine=True),
            nn.ReLU(True),
        ]
        channels = base
        for _ in range(2):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels * 2, 3, 2, 0, bias=False),
                nn.InstanceNorm2d(channels * 2, affine=True),
                nn.ReLU(True),
            ]
            channels *= 2
        layers += [ResBlock(channels) for _ in range(blocks)]
        for _ in range(2):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels // 2, 3, bias=False),
                nn.InstanceNorm2d(channels // 2, affine=True),
                nn.ReLU(True),
            ]
            channels //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.name


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    member_dir = Path(__file__).resolve().parents[1]
    task_dir = member_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=member_dir / "checkpoints/aswin_cyclegan_v2/best_model.pt",
    )
    parser.add_argument("--input", type=Path, default=task_dir / "data/monet_jpg")
    parser.add_argument("--output", type=Path, default=member_dir / "outputs/pred_A2B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    image_paths = sorted(
        path for path in args.input.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {args.input}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = Generator(config["generator_filters"], config["residual_blocks"])
    model.load_state_dict(checkpoint["EMA_A2B"])
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_size"], config["image_size"]),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    loader = DataLoader(
        ImageDataset(image_paths, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    generated = 0
    with torch.inference_mode():
        for images, names in loader:
            outputs = model(images.to(device)).cpu()
            for output, name in zip(outputs, names):
                save_image((output + 1) / 2, args.output / f"{Path(name).stem}.jpg")
                generated += 1
                if generated % 25 == 0 or generated == len(image_paths):
                    print(f"Generated {generated}/{len(image_paths)}", flush=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Saved {generated} images to {args.output}")


if __name__ == "__main__":
    main()
