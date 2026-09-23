from pathlib import Path
import json
import shutil
from typing import Callable, Optional

import cv2
import numpy as np

from modules.video_loader import load_video_metadata
from modules.frame_sampler import sample_frames
from modules.result_writer import write_json
from modules.visualizer import save_score_plot


def _gray_hist(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read frame: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    return hist.astype(np.float32)


def _frame_diff_score(prev_path: str | None, cur_path: str) -> float:
    if prev_path is None:
        return 1.0
    h1 = _gray_hist(prev_path)
    h2 = _gray_hist(cur_path)
    return float(np.mean(np.abs(h2 - h1)))


def extract_keyframes_fast(
    video_path: str,
    output_dir: str,
    sample_fps: float = 1.0,
    top_k: int = 10,
    min_gap_sec: float = 2.0,
    image_ext: str = ".jpg",
    progress_callback: Optional[Callable[[str, dict | None], None]] = None,
) -> dict:
    """Lightweight keyframe extraction without LLaVA/Llama.

    It samples frames, scores temporal novelty by grayscale histogram difference,
    then greedily keeps high-score frames while enforcing a minimum time gap.
    Useful as a fast UI/demo mode when the full LLM pipeline is too heavy.
    """
    out = Path(output_dir)
    frames_dir = out / "frames"
    selected_dir = out / "selected_frames"
    out.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    def notify(msg: str, payload: dict | None = None):
        if progress_callback:
            progress_callback(msg, payload)
        print(msg)

    notify("[FAST 1/4] 비디오 메타데이터 읽는 중")
    metadata = load_video_metadata(video_path)
    write_json(out / "metadata.json", metadata)

    notify("[FAST 2/4] 프레임 샘플링 중")
    frames = sample_frames(video_path, sample_fps, str(frames_dir), image_ext=image_ext)
    write_json(out / "frames.json", frames)

    notify("[FAST 3/4] 프레임 변화량 점수 계산 중")
    scored = []
    prev = None
    for frame in frames:
        score = _frame_diff_score(prev, frame["frame_path"])
        item = {
            "segment_id": frame["frame_id"],
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "start_time": frame["timestamp"],
            "end_time": frame["timestamp"],
            "representative_frame_path": frame["frame_path"],
            "raw_score_0_10": round(score * 10, 4),
            "local_score": score,
            "final_score": score,
            "caption": "",
        }
        scored.append(item)
        prev = frame["frame_path"]

    ranked = sorted(scored, key=lambda x: x["final_score"], reverse=True)
    selected = []
    for item in ranked:
        if len(selected) >= int(top_k):
            break
        if all(abs(item["timestamp"] - other["timestamp"]) >= float(min_gap_sec) for other in selected):
            selected.append(item)
    selected = sorted(selected, key=lambda x: x["timestamp"])

    for idx, item in enumerate(selected, start=1):
        src = Path(item["representative_frame_path"])
        dst = selected_dir / f"keyframe_{idx:03d}_t{item['timestamp']:.2f}s{src.suffix}"
        shutil.copy2(src, dst)
        item["selected_frame_copy"] = str(dst)

    write_json(out / "scores.json", scored)
    write_json(out / "summary.json", selected)
    save_score_plot(scored, selected, str(out))

    notify("[FAST 4/4] 완료", {"num_selected": len(selected), "output_dir": str(out)})
    return {
        "status": "success",
        "mode": "fast_cv",
        "output_dir": str(out),
        "num_frames": len(frames),
        "num_selected": len(selected),
    }
