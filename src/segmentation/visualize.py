import numpy as np
from PIL import Image

processed_dir = "/home/user/Vision_project/data/processed"
results_dir = "/home/user/Vision_project/results"

for i, name in enumerate(["pre_fire", "during_fire", "post_fire"]):
    orig = Image.open(f"{processed_dir}/{i:04d}.jpg").convert("RGB")
    mask = Image.open(f"{results_dir}/{name}_mask.jpg").convert("L")
    mask = mask.resize(orig.size, Image.NEAREST) 
    
    orig_arr = np.array(orig)
    mask_arr = np.array(mask)
    
    overlay = orig_arr.copy()
    overlay[mask_arr > 127] = [255, 0, 0]
    
    result = Image.fromarray(((orig_arr * 0.6) + (overlay * 0.4)).astype(np.uint8))
    result.save(f"{results_dir}/{name}_overlay.jpg")
    print(f"{name} 오버레이 저장 완료")
