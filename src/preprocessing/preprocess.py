import numpy as np
import rasterio
from pathlib import Path
from PIL import Image

def tif_to_png(tif_path, output_dir, filename):
    """GEE에서 받은 tif를 SAM2 입력용 PNG로 변환"""
    with rasterio.open(tif_path) as src:
        # RGB 밴드 (B4, B3, B2)
        r = src.read(3).astype(np.float32)  # B4 Red
        g = src.read(2).astype(np.float32)  # B3 Green
        b = src.read(1).astype(np.float32)  # B2 Blue

    # 정규화 (0~255)
    def normalize(band):
        band = np.clip(band, 0, 3000)
        return ((band / 3000) * 255).astype(np.uint8)

    rgb = np.stack([normalize(r), normalize(g), normalize(b)], axis=-1)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img = Image.fromarray(rgb)
    img.save(output_dir / filename.replace(".png", ".jpg"), quality=95)
    print(f"저장 완료: {output_dir / filename.replace('.png', '.jpg')}")

if __name__ == "__main__":
    raw_dir = Path("../../data/raw")
    output_dir = Path("../../data/processed")

    for i, name in enumerate(["pre_fire", "during_fire", "post_fire"]):
        tif_to_png(raw_dir / f"{name}.tif", output_dir, f"{i:04d}.jpg")
