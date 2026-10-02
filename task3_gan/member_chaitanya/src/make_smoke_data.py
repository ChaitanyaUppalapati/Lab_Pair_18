"""Write a few synthetic 'photo' and 'Monet' images for the smoke test (no download needed)."""
import numpy as np
from PIL import Image

from common import MEMBER_DIR

ROOT = MEMBER_DIR / "data_processed" / "smoke_data"


def main() -> None:
    rng = np.random.default_rng(0)
    for name, n, base in (("photo_jpg", 12, (90, 140, 200)), ("monet_jpg", 6, (200, 170, 90))):
        out = ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.clip(rng.normal(base, 40, size=(96, 96, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(img).save(out / f"{name[:5]}_{i:03d}.jpg")
    print(f"wrote smoke images to {ROOT.relative_to(MEMBER_DIR.parents[1]).as_posix()}")


if __name__ == "__main__":
    main()
