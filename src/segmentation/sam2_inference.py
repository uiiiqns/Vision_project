import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.append(str(Path.home() / "sam2"))
from sam2.build_sam import build_sam2_video_predictor

def run_sam2_wildfire(frames_dir, burned_mask_path, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(Path.home() / "sam2/checkpoints/sam2.1_hiera_large.pt"),
        device=device
    )

    frame_names = ["pre_fire", "during_fire", "post_fire"]
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_img = Image.open(frames_dir / "0000.jpg")
    img_size = sample_img.size
    print(f"이미지 크기: {img_size}")

    burned_mask = Image.open(burned_mask_path).convert("L")
    burned_mask = burned_mask.resize(img_size, Image.NEAREST)
    mask_arr = (np.array(burned_mask) > 127).astype(bool)
    print(f"마스크 피해 픽셀 수: {mask_arr.sum()}")

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir))

        _, _, masks = predictor.add_new_mask(
            inference_state=state,
            frame_idx=1,
            obj_id=1,
            mask=mask_arr
        )

        # during_fire 마스크 저장
        mask = (masks[0] > 0).cpu().numpy().squeeze()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(output_dir / f"{frame_names[1]}_mask.jpg")
        print(f"during_fire 마스크 저장 완료")

        # 순방향 전파 (during → post)
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0] > 0).cpu().numpy().squeeze()
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(output_dir / f"{frame_names[frame_idx]}_mask.jpg")
            print(f"순방향 프레임 {frame_idx} ({frame_names[frame_idx]}) 마스크 저장 완료")

        # 역방향 전파 (during → pre)
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=True):
            mask = (mask_logits[0] > 0).cpu().numpy().squeeze()
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(output_dir / f"{frame_names[frame_idx]}_mask.jpg")
            print(f"역방향 프레임 {frame_idx} ({frame_names[frame_idx]}) 마스크 저장 완료")

if __name__ == "__main__":
    run_sam2_wildfire(
        frames_dir=str(Path.home() / "Vision_project/data/processed"),
        burned_mask_path=str(Path.home() / "Vision_project/data/processed/burned_mask.png"),
        output_dir=str(Path.home() / "Vision_project/results")
    )
