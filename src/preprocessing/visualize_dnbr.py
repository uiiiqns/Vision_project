import numpy as np
from PIL import Image
from pathlib import Path

processed_dir = Path("/home/user/Vision_project/data/processed")
results_dir = Path("/home/user/Vision_project/results")

# 원본 이미지에 dNBR 마스크 오버레이
orig = Image.open(processed_dir / "0001.jpg").convert("RGB")
mask = Image.open(processed_dir / "burned_mask.png").convert("L")
mask = mask.resize(orig.size, Image.NEAREST)

orig_arr = np.array(orig)
mask_arr = np.array(mask)

overlay = orig_arr.copy()
overlay[mask_arr > 127] = [255, 165, 0]  # 주황색으로 표시

result = Image.fromarray(((orig_arr * 0.5) + (overlay * 0.5)).astype(np.uint8))
result.save(results_dir / "dnbr_overlay.jpg")
print("dNBR 오버레이 저장 완료!")
