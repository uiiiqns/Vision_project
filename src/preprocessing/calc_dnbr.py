import numpy as np
import rasterio
from pathlib import Path
from PIL import Image
from PIL import Image as PILImage

def calc_nbr(tif_path):
    """이미 계산된 NBR 밴드 직접 사용"""
    with rasterio.open(tif_path) as src:
        nbr = src.read(7).astype(np.float32)  # NBR 밴드 직접
    return nbr

def calc_dnbr(pre_tif, post_tif, output_dir):
    """dNBR = pre_NBR - post_NBR, 양수일수록 피해 심각"""
    pre_nbr = calc_nbr(pre_tif)
    post_nbr = calc_nbr(post_tif)
    
    dnbr = pre_nbr - post_nbr
    
    # 피해 영역 마스크 (dNBR > 0.1 이면 피해 지역)
    dnbr_clean = np.nan_to_num(dnbr, nan=0.0)
    burned_mask = (dnbr > 0.27).astype(np.uint8) * 255
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # dNBR 저장
    np.save(output_dir / "dnbr.npy", dnbr)
    
    # 마스크 이미지 저장
    jpg_size = (2790, 1483)  # 0001.jpg 크기
    mask_img = PILImage.fromarray(burned_mask)
    mask_img = mask_img.resize(jpg_size, PILImage.NEAREST)
    mask_img.save(output_dir / "burned_mask.png")
    print(f"dNBR 계산 완료! 피해 픽셀 수: {(burned_mask > 0).sum()}")
    
    return dnbr, burned_mask

if __name__ == "__main__":
    raw_dir = Path("/home/user/Vision_project/data/raw")
    output_dir = Path("/home/user/Vision_project/data/processed")
    
    dnbr, mask = calc_dnbr(
        raw_dir / "pre_fire.tif",
        raw_dir / "post_fire.tif",
        output_dir
    )
