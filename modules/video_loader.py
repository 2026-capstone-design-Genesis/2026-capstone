from pathlib import Path
import cv2


def load_video_metadata(video_path: str) -> dict:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "video_path": str(path),
        "fps": float(fps),
        "frame_count": frame_count,
        "duration_sec": float(duration_sec),
        "width": width,
        "height": height,
    }
