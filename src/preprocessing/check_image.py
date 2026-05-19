from PIL import Image
import numpy as np

for name in ["pre_fire", "during_fire", "post_fire"]:
    img = Image.open(f"../../data/processed/{name}.png")
    arr = np.array(img)
    print(f"{name}: 크기={img.size}, 최솟값={arr.min()}, 최댓값={arr.max()}, 평균={arr.mean():.1f}")
