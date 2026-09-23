from __future__ import annotations

"""CluFrame-style redundancy removal before LLMVS scoring.

This module ports the key idea from the private key-frame extraction code into
this LLMVS web project:

1. Frame differencing removes nearly-static/redundant sampled frames.
2. Visual features are extracted from the remaining motion frames.
3. K-Means selects cluster representatives, sorted by timestamp.
4. The selected representatives are returned in the same frame-dict format that
   the original LLMVS pipeline expects.

The default feature backend is a lightweight OpenCV descriptor so this module can
run in the existing web UI environment without forcing another large model load.
If torchvision is installed, config can set feature_backend: efficientnet_b0 to
use an ImageNet EfficientNetB0 backbone similar to the private TC module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import json
import math
import shutil

import cv2
import numpy as np

try:  # scikit-learn is already listed in this project requirements.
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    KMeans = None
    StandardScaler = None


ProgressCallback = Optional[Callable[[str, dict | None], None]]


@dataclass
class CluFrameConfig:
    enabled: bool = False
    sample_fps: float = 1.0
    image_ext: str = ".jpg"
    max_candidates: int = 64
    min_motion_frames: int = 2
    diff_threshold: Optional[float] = None
    resize_width: int = 320
    blur_ksize: int = 5
    use_morph: bool = True
    use_contours: bool = True
    min_contour_area: int = 800
    min_total_motion: float = 1500.0
    feature_backend: str = "lightweight"  # lightweight | efficientnet_b0
    random_state: int = 0


def _notify(callback: ProgressCallback, message: str, payload: dict | None = None) -> None:
    if callback:
        callback(message, payload)
    print(message)


def _coerce_cfg(cfg: dict, sampling_cfg: dict) -> CluFrameConfig:
    p = dict(cfg or {})
    return CluFrameConfig(
        enabled=bool(p.get("enabled", False)),
        sample_fps=float(p.get("sample_fps", sampling_cfg.get("fps", 1.0))),
        image_ext=str(p.get("image_ext", sampling_cfg.get("image_ext", ".jpg"))),
        max_candidates=int(p.get("max_candidates", p.get("top_k", 64))),
        min_motion_frames=int(p.get("min_motion_frames", 2)),
        diff_threshold=p.get("diff_threshold", None),
        resize_width=int(p.get("resize_width", 320)),
        blur_ksize=int(p.get("blur_ksize", 5)),
        use_morph=bool(p.get("use_morph", True)),
        use_contours=bool(p.get("use_contours", True)),
        min_contour_area=int(p.get("min_contour_area", 800)),
        min_total_motion=float(p.get("min_total_motion", 1500.0)),
        feature_backend=str(p.get("feature_backend", "lightweight")),
        random_state=int(p.get("random_state", 0)),
    )


def _default_diff_thresh() -> float:
    # Same reference formula style as private/modules/frame_differencing.py.
    object_distance_m = 1.0
    object_size_factor = 1.0
    angle_rad = 20.0
    return 10.0 * object_size_factor * math.cos(angle_rad) / (object_distance_m**2.6) + 0.1


def _is_motion_frame(prev_bgr: np.ndarray | None, frame_bgr: np.ndarray, cfg: CluFrameConfig) -> bool:
    if prev_bgr is None:
        return True

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, float(cfg.resize_width) / float(w)) if w > 0 else 1.0
    if scale != 1.0:
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        prev = cv2.resize(prev_bgr, size, interpolation=cv2.INTER_AREA)
        cur = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    else:
        prev, cur = prev_bgr, frame_bgr

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    if cfg.blur_ksize and cfg.blur_ksize > 1:
        k = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
        prev_gray = cv2.GaussianBlur(prev_gray, (k, k), 0)
        cur_gray = cv2.GaussianBlur(cur_gray, (k, k), 0)

    diff = cv2.absdiff(prev_gray, cur_gray)
    threshold = float(cfg.diff_threshold) if cfg.diff_threshold is not None else _default_diff_thresh()
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    if cfg.use_morph:
        th = cv2.medianBlur(th, 5)
        th = cv2.dilate(th, None, iterations=2)

    scale_sq = scale * scale
    if cfg.use_contours:
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= cfg.min_contour_area * scale_sq:
                total_area += area
        return total_area >= cfg.min_total_motion * scale_sq

    motion_pixels = float(cv2.countNonZero(th))
    return motion_pixels >= cfg.min_total_motion * scale_sq


def _write_image(path: Path, frame_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame_bgr)
    if not ok:
        # Windows/non-ASCII path fallback style.
        ext = ".jpg" if path.suffix.lower() in {".jpg", ".jpeg"} else path.suffix.lower() or ".jpg"
        ok, buf = cv2.imencode(ext, frame_bgr)
        if not ok:
            raise RuntimeError(f"Failed to encode frame: {path}")
        path.write_bytes(buf.tobytes())


def _sample_motion_candidates(video_path: str, cfg: CluFrameConfig) -> tuple[list[dict], list[np.ndarray], dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # If cfg.sample_fps <= 0, treat as request to use all frames (no skipping)
    if cfg.sample_fps is None or float(cfg.sample_fps) <= 0.0:
        step = 1
    else:
        step = max(int(round(original_fps / max(float(cfg.sample_fps), 1e-6))), 1)

    sampled_meta: list[dict] = []
    sampled_bgr: list[np.ndarray] = []
    motion_meta: list[dict] = []
    motion_bgr: list[np.ndarray] = []
    prev_sampled: np.ndarray | None = None

    frame_idx = 0
    sampled_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            meta = {
                "sample_id": sampled_idx,
                "source_frame_index": int(frame_idx),
                "timestamp": float(frame_idx / original_fps),
            }
            sampled_meta.append(meta)
            sampled_bgr.append(frame.copy())
            if _is_motion_frame(prev_sampled, frame, cfg):
                motion_meta.append(meta)
                motion_bgr.append(frame.copy())
            prev_sampled = frame.copy()
            sampled_idx += 1
        frame_idx += 1

    cap.release()
    report = {
        "original_fps": float(original_fps),
        "source_frame_count": int(frame_count),
        "sample_step": int(step),
        "sampled_frames": len(sampled_meta),
        "motion_candidates": len(motion_meta),
        "skipped_static_sampled_frames": max(len(sampled_meta) - len(motion_meta), 0),
    }

    if len(motion_meta) < cfg.min_motion_frames:
        # Avoid empty downstream LLMVS input on very static videos.
        report["fallback"] = "motion_candidates_too_few_use_sampled_frames"
        return sampled_meta, sampled_bgr, report

    report["fallback"] = None
    return motion_meta, motion_bgr, report


def _lightweight_feature(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    edge = cv2.Canny(gray, 80, 160)
    edge_hist = cv2.calcHist([edge], [0], None, [8], [0, 256]).flatten()
    thumb = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).flatten().astype(np.float32) / 255.0
    feat = np.concatenate([h_hist, s_hist, v_hist, edge_hist, thumb]).astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-8)


def _efficientnet_features(frames_bgr: list[np.ndarray]) -> np.ndarray:
    # Lazy import keeps the default path light and prevents torchvision from being a hard requirement.
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frames_bgr), 16):
            batch = frames_bgr[start:start + 16]
            x = torch.stack([transform(bgr[:, :, ::-1].copy()) for bgr in batch], dim=0).to(device)
            feat = model(x)
            feat = F.normalize(feat, p=2, dim=1)
            out.append(feat.cpu().numpy().astype(np.float32))
    return np.vstack(out) if out else np.zeros((0, 1280), dtype=np.float32)


def _extract_features(frames_bgr: list[np.ndarray], backend: str) -> np.ndarray:
    if not frames_bgr:
        return np.zeros((0, 1), dtype=np.float32)
    if backend == "efficientnet_b0":
        return _efficientnet_features(frames_bgr)
    return np.vstack([_lightweight_feature(frame) for frame in frames_bgr]).astype(np.float32)


def _select_cluster_representatives(features: np.ndarray, max_candidates: int, random_state: int) -> tuple[list[int], np.ndarray]:
    n = int(features.shape[0])
    if n == 0:
        return [], np.zeros((0,), dtype=np.int32)
    k = max(1, min(int(max_candidates), n))
    if n <= k:
        return list(range(n)), np.arange(n, dtype=np.int32)
    if KMeans is None or StandardScaler is None:
        return sorted(set(np.linspace(0, n - 1, k, dtype=int).tolist())), np.zeros(n, dtype=np.int32)

    x = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(x).astype(np.int32)

    selected: list[tuple[int, int]] = []  # (earliest member position, selected position)
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        center = km.cluster_centers_[cluster_id]
        distances = np.linalg.norm(x[members] - center, axis=1)
        selected_pos = int(members[int(np.argmin(distances))])
        earliest_pos = int(np.min(members))
        selected.append((earliest_pos, selected_pos))

    # Deduplicate and sort by first temporal appearance of each visual cluster.
    dedup: dict[int, int] = {}
    for earliest, selected_pos in selected:
        if selected_pos not in dedup or earliest < dedup[selected_pos]:
            dedup[selected_pos] = earliest
    return [pos for pos, _ in sorted(dedup.items(), key=lambda item: item[1])], labels


def run_cluframe_preprocessing(
    video_path: str,
    output_frame_dir: str,
    preprocessing_cfg: dict,
    sampling_cfg: dict,
    progress_callback: ProgressCallback = None,
) -> tuple[list[dict], dict]:
    """Return LLMVS-compatible frame dicts after CluFrame-style filtering."""
    cfg = _coerce_cfg(preprocessing_cfg, sampling_cfg)
    out_dir = Path(output_frame_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, "[CluFrame 1/4] Frame Differencing으로 중복/정적 프레임 제거 중", {"fps": cfg.sample_fps})
    candidate_meta, candidate_bgr, report = _sample_motion_candidates(video_path, cfg)

    _notify(progress_callback, "[CluFrame 2/4] 후보 프레임 특징 추출 중", {"backend": cfg.feature_backend, "candidates": len(candidate_meta)})
    features = _extract_features(candidate_bgr, cfg.feature_backend)

    _notify(progress_callback, "[CluFrame 3/4] K-Means로 대표 후보 프레임 선택 중", {"max_candidates": cfg.max_candidates})
    selected_positions, labels = _select_cluster_representatives(features, cfg.max_candidates, cfg.random_state)

    frames: list[dict] = []
    selected_dir = out_dir
    for new_id, pos in enumerate(sorted(selected_positions, key=lambda i: candidate_meta[i]["timestamp"])):
        src_meta = candidate_meta[pos]
        frame_path = selected_dir / f"frame_{new_id:06d}{cfg.image_ext}"
        _write_image(frame_path, candidate_bgr[pos])
        frames.append({
            "frame_id": int(new_id),
            "source_frame_index": int(src_meta["source_frame_index"]),
            "timestamp": float(src_meta["timestamp"]),
            "frame_path": str(frame_path),
            "cluframe_candidate_position": int(pos),
            "cluframe_cluster_id": int(labels[pos]) if len(labels) == len(candidate_meta) else None,
        })

    report.update({
        "enabled": True,
        "method": "cluframe_fd_kmeans_preprocessing",
        "feature_backend": cfg.feature_backend,
        "max_candidates": int(cfg.max_candidates),
        "selected_candidate_frames": len(frames),
        "selected_source_frame_indices": [f["source_frame_index"] for f in frames],
        "selected_timestamps": [f["timestamp"] for f in frames],
    })

    np.save(out_dir.parent / "cluframe_features.npy", features)
    (out_dir.parent / "cluframe_preprocessing.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Keep a clearly named copy for quick inspection in the output zip.
    inspect_dir = out_dir.parent / "cluframe_candidates"
    if inspect_dir.exists():
        shutil.rmtree(inspect_dir)
    inspect_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        src = Path(f["frame_path"])
        dst = inspect_dir / f"candidate_{f['frame_id']:03d}_t{f['timestamp']:.2f}s{src.suffix}"
        shutil.copy2(src, dst)

    _notify(progress_callback, "[CluFrame 4/4] LLMVS 입력 후보 프레임 구성 완료", {"selected": len(frames)})
    return frames, report
