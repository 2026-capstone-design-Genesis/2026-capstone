from pathlib import Path
import cv2
from tqdm import tqdm


def sample_frames(video_path: str, sample_fps: float, output_frame_dir: str, image_ext: str = ".jpg") -> list[dict]:
    output_dir = Path(output_frame_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # If sample_fps <= 0, treat as request to use all frames (no skipping)
    if sample_fps is None or float(sample_fps) <= 0.0:
        step = 1
    else:
        step = max(int(round(original_fps / float(sample_fps))), 1)

    frames = []
    frame_idx = 0
    saved_idx = 0

    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc="Sampling frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            timestamp = frame_idx / original_fps
            frame_name = f"frame_{saved_idx:06d}{image_ext}"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            frames.append({
                "frame_id": saved_idx,
                "source_frame_index": frame_idx,
                "timestamp": float(timestamp),
                "frame_path": str(frame_path),
            })
            saved_idx += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return frames
