import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.append(str(Path.home() / "sam2"))
from sam2.build_sam import build_sam2_video_predictor

def run_sam2_wildfire(frames_dir, output_dir):
    """
    frames_dir: pre_fire.png, during_fire.png, post_fire.png 있는 폴더
    output_dir: 마스크 저장 폴더
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")

    # SAM2 로드
    predictor = build_sam2_video_predictor(
	"configs/sam2.1/sam2.1_hiera_l.yaml",
        str(Path.home() / "sam2/checkpoints/sam2.1_hiera_large.pt"),
        device=device
    )

    # 프레임 로드
    frame_names = ["pre_fire.jpg", "during_fire.jpg", "post_fire.jpg"]
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SAM2 비디오 세션 시작
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir))

        _, _, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,  # 1 → 0으로 변경
            obj_id=1,
            points=np.array([[256, 256]]),
            labels=np.array([1])
        )

        # 프레임 0 마스크 저장
        mask = (masks[0] > 0).cpu().numpy().squeeze()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(output_dir / frame_names[0].replace(".jpg", "_mask.jpg"))
        print(f"프레임 0 마스크 저장 완료")

        # 나머지 프레임 전파
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0] > 0).cpu().numpy().squeeze()
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(output_dir / frame_names[frame_idx].replace(".jpg", "_mask.jpg"))
            print(f"프레임 {frame_idx} 마스크 저장 완료")

if __name__ == "__main__":
    run_sam2_wildfire(
        frames_dir=str(Path.home() / "Vision_project/data/processed"),
        output_dir=str(Path.home() / "Vision_project/results")
    )
