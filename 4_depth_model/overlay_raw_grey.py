from PIL import Image
import numpy as np
from pathlib import Path

def dim_with_depth_mask(
    original_path: str,
    mask_path: str,
    out_path: str,
    strength: float = 1.0,     # 0=no effect, 1=full mask control
    invert_mask: bool = False, # True if your mask meaning is reversed
    gamma: float = 1.4         # >1 makes dark regions darker faster
):
    original_path = Path(original_path)
    mask_path = Path(mask_path)
    out_path = Path(out_path)

    if not original_path.exists():
        raise FileNotFoundError(f"Original image not found: {original_path.resolve()}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path.resolve()}")

    # Load original (RGB) and mask (grayscale)
    img = Image.open(original_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Ensure same size
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.BILINEAR)

    img_np = np.asarray(img).astype(np.float32) / 255.0      # HxWx3 in [0,1]
    mask_np = np.asarray(mask).astype(np.float32) / 255.0    # HxW in [0,1]

    if invert_mask:
        mask_np = 1.0 - mask_np

    # Optional gamma shaping of the mask response
    if gamma != 1.0:
        mask_np = np.clip(mask_np, 0.0, 1.0) ** gamma

    # Build per-pixel factor so BLACK in mask darkens most:
    # strength=1.0 => factor == mask (0->black, 1->unchanged)
    # strength=0.0 => factor == 1.0 (no effect)
    factor = (1.0 - strength) + strength * mask_np
    factor = np.clip(factor, 0.0, 1.0)

    # Apply (broadcast factor HxW -> HxWx1)
    out_np = img_np * factor[..., None]
    out_np = (np.clip(out_np, 0.0, 1.0) * 255.0).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_np).save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dim_with_depth_mask(
        original_path=base_dir / "img_test/cyber_rabbit.jpg",
        mask_path=base_dir / "img_test/cyber_rabbit_depthmap.jpg",
        out_path=base_dir / "img_test/overlay_raw_grey/overlay_raw_grey_04.jpg",
        strength=0.4,
        invert_mask=False,
        gamma=1.4
    )
