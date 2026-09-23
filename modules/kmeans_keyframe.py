from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Callable, Optional

import cv2
import numpy as np

from modules.video_loader import load_video_metadata
from modules.frame_sampler import sample_frames
from modules.result_writer import write_json
from modules.visualizer import save_score_plot, save_html_report

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - fallback for minimal envs
    KMeans = None
    StandardScaler = None


def _notify(callback: Optional[Callable[[str, dict | None], None]], message: str, payload: dict | None = None) -> None:
    if callback:
        callback(message, payload)
    print(message)


def _extract_visual_feature(image_path: str) -> np.ndarray:
    """Small deterministic frame descriptor for lightweight K-Means selection.

    It combines color histogram, edge histogram, and a downsampled thumbnail.
    This avoids loading CLIP/DINO, so it is suitable for a local web demo.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read frame: {image_path}")

    small = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    edge = cv2.Canny(gray, 80, 160)
    edge_hist = cv2.calcHist([edge], [0], None, [8], [0, 256]).flatten()

    thumb = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).flatten().astype(np.float32) / 255.0
    feat = np.concatenate([h_hist, s_hist, v_hist, edge_hist, thumb]).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat


def _fallback_even_selection(features: np.ndarray, k: int) -> list[int]:
    if len(features) == 0:
        return []
    if len(features) <= k:
        return list(range(len(features)))
    return sorted(set(np.linspace(0, len(features) - 1, k, dtype=int).tolist()))


def _select_nearest_to_centers(features: np.ndarray, k: int, random_state: int = 42) -> tuple[list[int], np.ndarray]:
    n = len(features)
    k = max(1, min(int(k), n))
    if n == 0:
        return [], np.zeros((0,), dtype=np.int32)
    if n <= k:
        return list(range(n)), np.arange(n, dtype=np.int32)
    if KMeans is None or StandardScaler is None:
        selected = _fallback_even_selection(features, k)
        return selected, np.zeros(n, dtype=np.int32)

    x = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(x)

    selected: list[int] = []
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        center = km.cluster_centers_[cluster_id]
        distances = np.linalg.norm(x[members] - center, axis=1)
        selected.append(int(members[int(np.argmin(distances))]))

    # If empty clusters or library behavior leaves us short, add farthest remaining frames.
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected]
        center = x.mean(axis=0)
        ranked = sorted(remaining, key=lambda i: float(np.linalg.norm(x[i] - center)), reverse=True)
        selected.extend(ranked[: k - len(selected)])

    return sorted(set(selected)), labels.astype(np.int32)


def extract_keyframes_kmeans(
    video_path: str,
    output_dir: str,
    sample_fps: float = 1.0,
    top_k: int = 10,
    min_gap_sec: float = 0.0,
    image_ext: str = ".jpg",
    progress_callback: Optional[Callable[[str, dict | None], None]] = None,
) -> dict:
    """Extract keyframes by clustering sampled frames in visual feature space."""
    out = Path(output_dir)
    frames_dir = out / "frames"
    selected_dir = out / "selected_frames"
    out.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, "[KMEANS 1/5] 비디오 메타데이터 읽는 중")
    metadata = load_video_metadata(video_path)
    metadata["method"] = "kmeans_visual_clustering"
    write_json(out / "metadata.json", metadata)

    _notify(progress_callback, "[KMEANS 2/5] 프레임 샘플링 중", {"fps": sample_fps})
    frames = sample_frames(video_path, sample_fps, str(frames_dir), image_ext=image_ext)
    write_json(out / "frames.json", frames)

    if not frames:
        write_json(out / "scores.json", [])
        write_json(out / "summary.json", [])
        return {"status": "success", "mode": "kmeans", "output_dir": str(out), "num_frames": 0, "num_selected": 0}

    _notify(progress_callback, "[KMEANS 3/5] 프레임 특징 추출 중")
    features = np.vstack([_extract_visual_feature(f["frame_path"]) for f in frames]).astype(np.float32)
    np.save(out / "kmeans_features.npy", features)

    _notify(progress_callback, "[KMEANS 4/5] K-Means 클러스터링으로 대표 프레임 선택 중", {"k": int(top_k)})
    selected_indices, labels = _select_nearest_to_centers(features, int(top_k))

    # Optional temporal gap filter. If it removes too many, fill back with time-sorted cluster representatives.
    if min_gap_sec and min_gap_sec > 0:
        filtered: list[int] = []
        for idx in sorted(selected_indices, key=lambda i: frames[i]["timestamp"]):
            if all(abs(frames[idx]["timestamp"] - frames[j]["timestamp"]) >= float(min_gap_sec) for j in filtered):
                filtered.append(idx)
        selected_indices = filtered or selected_indices

    selected_set = set(selected_indices)
    scores = []
    for idx, frame in enumerate(frames):
        # Selected cluster representatives get 1.0; others use inverse distance to closest selected feature for plotting.
        if idx in selected_set:
            final_score = 1.0
        else:
            d = min(float(np.linalg.norm(features[idx] - features[j])) for j in selected_set) if selected_set else 1.0
            final_score = float(1.0 / (1.0 + d))
        scores.append({
            "segment_id": frame["frame_id"],
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "start_time": frame["timestamp"],
            "end_time": frame["timestamp"],
            "representative_frame_path": frame["frame_path"],
            "cluster_id": int(labels[idx]) if len(labels) == len(frames) else 0,
            "raw_score_0_10": round(final_score * 10, 4),
            "local_score": final_score,
            "final_score": final_score,
            "caption": "",
        })

    summary = []
    for idx in sorted(selected_indices, key=lambda i: frames[i]["timestamp"]):
        item = dict(scores[idx])
        summary.append(item)

    for rank, item in enumerate(summary, start=1):
        src = Path(item["representative_frame_path"])
        dst = selected_dir / f"keyframe_{rank:03d}_t{item['timestamp']:.2f}s{src.suffix}"
        shutil.copy2(src, dst)
        item["selected_frame_copy"] = str(dst)

    write_json(out / "scores.json", scores)
    write_json(out / "summary.json", summary)
    save_score_plot(scores, summary, str(out))
    save_html_report(metadata, summary, scores, summary, str(out))

    _notify(progress_callback, "[KMEANS 5/5] 완료", {"num_selected": len(summary), "output_dir": str(out)})
    return {
        "status": "success",
        "mode": "kmeans",
        "output_dir": str(out),
        "num_frames": len(frames),
        "num_selected": len(summary),
    }
