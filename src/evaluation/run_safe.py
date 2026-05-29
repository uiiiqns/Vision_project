import numpy as np
import cv2
import rasterio
from segment_anything import sam_model_registry, SamPredictor
import torch
from pathlib import Path
from PIL import Image as PILImage

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = str(Path.home() / "Vision_project/checkpoints/sam_vit_l_0b3195.pth")
model_type = "vit_l"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

def run_safe(tif_path, output_path):
    with rasterio.open(tif_path) as src:
        image = src.read()

    # RGB 추출 및 정규화
    rgb = np.nan_to_num(image[:3].transpose(1, 2, 0))
    p_low = np.percentile(rgb, 2)
    p_high = np.percentile(rgb, 98)
    rgb = np.clip((rgb - p_low) / (p_high - p_low), 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    # dNBR 마스크에서 bounding box 생성
    burned_mask = np.array(PILImage.open(
        str(Path.home() / "Vision_project/data/processed/burned_mask.png")
    ).convert("L"))
    burned_mask_resized = cv2.resize(burned_mask, (rgb.shape[1], rgb.shape[0]))
    non_zero = np.argwhere(burned_mask_resized > 127)
    y_min, x_min = np.min(non_zero, axis=0)
    y_max, x_max = np.max(non_zero, axis=0)
    margin = 50
    box = np.array([[
        max(x_min - margin, 0),
        max(y_min - margin, 0),
        min(x_max + margin, rgb.shape[1]),
        min(y_max + margin, rgb.shape[0])
    ]])

    print(f"Bounding Box: {box}")

    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)

    best_mask = masks[np.argmax(scores)]
    mask_img = (best_mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(output_path), mask_img)
    print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    raw_dir = Path.home() / "Vision_project/data/raw"
    output_dir = Path.home() / "Vision_project/results"

    for name in ["during_fire", "post_fire"]:
        run_safe(raw_dir / f"{name}.tif", output_dir / f"safe_{name}_mask.png")

