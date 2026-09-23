"""Recorded-segment analysis for the realtime CCTV dashboard."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import httpx
import numpy as np

from modules.hietaskim import extract_descriptors, hierarchical_regions, sample_indices, select_keyshots


RECORDING_FPS = 4.0
SUMMARY_FPS = 4.0
MOTION_THRESHOLD_PERCENT = 0.65
MAX_KEYFRAME_CANDIDATES = 96
KEYFRAME_COUNT = 8
MOTION_END_PERCENT = 0.3
QUIET_SECONDS = 1.5
MERGE_GAP_SECONDS = 2.0
FEATURE_DISTANCE = 0.015
MAX_ACTION_SECONDS = 6.0
ACTION_CONTEXT_SECONDS = 1.5
CLIP_PRE_SECONDS = 3.0
CLIP_POST_SECONDS = 3.0
BROWSER_VIDEO_CODEC = "VP80"
BROWSER_VIDEO_SUFFIX = ".webm"


def _video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Create a VP8/WebM writer that Chromium-based browsers can decode."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*BROWSER_VIDEO_CODEC),
        max(1.0, float(fps)),
        size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("WebM 영상 저장 파일을 열 수 없습니다.")
    return writer


def _atomic_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_project_env(root: Path) -> dict[str, str]:
    values = dict(os.environ)
    for candidate in (root / ".env", root.parent / ".env"):
        if not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in values:
                values[key] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class RecordedSegment:
    path: Path
    start: float
    end: float
    frame_count: int
    fps: float
    timestamps: tuple[float, ...] = ()


class SegmentRecorder:
    """Persist incoming JPEG frames as browser-compatible WebM segments."""

    def __init__(self, output_dir: Path, fps: float = RECORDING_FPS):
        self.folder = output_dir / "recordings"
        self.fps = fps
        self.index = 1
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.size: tuple[int, int] | None = None
        self.start = 0.0
        self.frames = 0
        self.timestamps: list[float] = []

    @property
    def has_frames(self) -> bool:
        return self.frames > 0

    def write(self, jpeg: bytes, second: float) -> None:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        height, width = frame.shape[:2]
        if self.writer is None:
            self.folder.mkdir(parents=True, exist_ok=True)
            self.path = self.folder / f"recording_{self.index:04d}{BROWSER_VIDEO_SUFFIX}"
            self.size = (max(2, width - width % 2), max(2, height - height % 2))
            self.start = max(0.0, float(second))
            self.writer = _video_writer(self.path, self.fps, self.size)
        if self.size != (width, height):
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(frame)
        self.frames += 1
        self.timestamps.append(float(second))

    def rotate(self, end: float) -> RecordedSegment | None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if not self.path or self.frames <= 0:
            return None
        segment = RecordedSegment(
            path=self.path,
            start=self.start,
            end=max(self.start, float(end)),
            frame_count=self.frames,
            fps=self.fps,
            timestamps=tuple(self.timestamps),
        )
        _atomic_json(self.path.with_suffix('.timestamps.json'), self.timestamps)
        self.index += 1
        self.path = None
        self.size = None
        self.frames = 0
        self.timestamps = []
        return segment

    def close(self, end: float) -> RecordedSegment | None:
        return self.rotate(end)


@dataclass
class FrameCandidate:
    frame_index: int
    local_second: float
    timestamp: float
    motion_percent: float
    jpeg: bytes
    feature: np.ndarray
    quality: dict[str, Any] = field(default_factory=dict)


def _feature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    color = cv2.resize(small, (8, 5), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    histogram = cv2.calcHist([small], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3).ravel()
    histogram /= max(float(histogram.sum()), 1.0)
    return np.concatenate((color, histogram * 4.0))


def _jpeg(frame: np.ndarray, quality: int = 86) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("키프레임 JPEG 생성에 실패했습니다.")
    return encoded.tobytes()


def frame_quality(frame: np.ndarray) -> dict[str, Any]:
    """Measure whether a frame is usable as visual evidence, not whether it is important."""
    if frame.size == 0:
        return {'status': 'rejected', 'usable': False, 'score': 0.0, 'reasons': ['empty_frame']}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    edge_ratio = float(np.mean(np.abs(laplacian) >= 12))
    dark_ratio = float(np.mean(gray <= 12))
    bright_ratio = float(np.mean(gray >= 243))
    reasons = []
    if brightness < 18 or dark_ratio > .96:
        reasons.append('too_dark')
    if brightness > 238 or bright_ratio > .96:
        reasons.append('too_bright')
    if contrast < 8 and edge_ratio < .003:
        reasons.append('low_information')
    elif sharpness < 12:
        reasons.append('blurred')
    exposure_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    score = (.30 * min(1.0, sharpness / 120) + .25 * exposure_score
             + .25 * min(1.0, contrast / 45) + .20 * min(1.0, edge_ratio / .08))
    return {
        'status': 'usable' if not reasons else 'rejected', 'usable': not reasons,
        'score': round(score, 4), 'reasons': reasons,
        'sharpness': round(sharpness, 2), 'brightness': round(brightness, 2),
        'contrast': round(contrast, 2), 'edge_ratio': round(edge_ratio, 4),
        'dark_ratio': round(dark_ratio, 4), 'bright_ratio': round(bright_ratio, 4),
    }


def compensated_motion(previous_gray: np.ndarray | None, gray: np.ndarray) -> dict[str, Any]:
    """Estimate foreground motion after aligning the previous frame to the camera view."""
    if previous_gray is None:
        return {'motion_percent': 0.0, 'raw_motion_percent': 0.0, 'camera_motion': False,
                'camera_shift_pixels': 0.0, 'alignment': 'first_frame'}
    changed = cv2.absdiff(gray, previous_gray) > 18
    raw = float(np.mean(changed) * 100.0)
    tile_changes = []
    for row in np.array_split(changed, 4, axis=0):
        tile_changes.extend(float(np.mean(tile)) for tile in np.array_split(row, 6, axis=1))
    global_change_ratio = float(np.mean(np.asarray(tile_changes) >= .12))
    widespread_change = bool(raw >= MOTION_THRESHOLD_PERCENT and global_change_ratio >= .75)

    def fallback(alignment: str) -> dict[str, Any]:
        return {
            'motion_percent': 0.0 if widespread_change else raw,
            'raw_motion_percent': raw,
            'camera_motion': widespread_change,
            'camera_shift_pixels': 0.0,
            'global_change_ratio': round(global_change_ratio, 3),
            'alignment': alignment,
        }

    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=160, qualityLevel=.01, minDistance=7)
    if points is None or len(points) < 8:
        return fallback('insufficient_features')
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
    if tracked is None or status is None:
        return fallback('flow_failed')
    valid = status.ravel().astype(bool)
    if int(valid.sum()) < 8:
        return fallback('insufficient_matches')
    matrix, inliers = cv2.estimateAffinePartial2D(
        points[valid], tracked[valid], method=cv2.RANSAC, ransacReprojThreshold=2.5
    )
    if matrix is None:
        return fallback('transform_failed')
    aligned = cv2.warpAffine(previous_gray, matrix, (gray.shape[1], gray.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    residual = float(np.mean(cv2.absdiff(gray, aligned) > 18) * 100.0)
    shift = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    inlier_ratio = float(np.mean(inliers)) if inliers is not None else 0.0
    camera_motion = bool(widespread_change or (inlier_ratio >= .55 and shift >= 1.5 and residual < raw * .65))
    effective_motion = 0.0 if widespread_change else residual
    return {
        'motion_percent': effective_motion, 'raw_motion_percent': raw,
        'camera_motion': camera_motion, 'camera_shift_pixels': round(shift, 3),
        'global_change_ratio': round(global_change_ratio, 3),
        'alignment': 'aligned', 'alignment_inlier_ratio': round(inlier_ratio, 3),
    }


def _select_keyframes(candidates: list[FrameCandidate], count: int = KEYFRAME_COUNT) -> list[FrameCandidate]:
    if len(candidates) <= count:
        return sorted(candidates, key=lambda item: item.timestamp)
    features = np.stack([item.feature for item in candidates])
    chosen = [int(np.argmax([item.motion_percent for item in candidates]))]
    while len(chosen) < count:
        distances = np.min(
            np.stack([np.mean((features - features[index]) ** 2, axis=1) for index in chosen]),
            axis=0,
        )
        temporal = np.array(
            [min(abs(item.timestamp - candidates[index].timestamp) for index in chosen) for item in candidates]
        )
        distances += 0.003 * temporal / max(1.0, float(temporal.max()))
        distances[chosen] = -1.0
        chosen.append(int(np.argmax(distances)))
    return sorted((candidates[index] for index in chosen), key=lambda item: item.timestamp)


def _write_clip(video_path: Path, output_path: Path, center: float, fps: float, bounds=None, timestamps=()) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_index = max(0, int(round((center - CLIP_PRE_SECONDS) * fps)))
    end_index = min(max(0, total - 1), int(round((center + CLIP_POST_SECONDS) * fps)))
    if bounds is not None:
        start_index, end_index = bounds
    if total <= 0 or width <= 0 or height <= 0 or end_index < start_index:
        capture.release()
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = _video_writer(output_path, fps, (width, height))
    except RuntimeError:
        capture.release()
        return False
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    written = 0
    for index in range(start_index, end_index + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        target = written + 1
        if timestamps:
            next_time = timestamps[index + 1] if index < end_index else timestamps[index] + 1 / fps
            target = max(written, round((next_time - timestamps[start_index]) * fps))
        while written < target:
            writer.write(frame)
            written += 1
    writer.release()
    capture.release()
    return output_path.is_file() and output_path.stat().st_size > 0


def context_sample_seconds(start: float, center: float, end: float, count: int) -> list[float]:
    """Distribute evidence across before, transition, and after states."""
    if count <= 0 or end < start:
        return []
    center = min(end, max(start, center))
    third = max(1, count // 3)
    before = np.linspace(start, center, num=third, endpoint=False)
    transition_start, transition_end = max(start, center - .5), min(end, center + .5)
    middle = np.linspace(transition_start, transition_end, num=third)
    after_count = max(1, count - len(before) - len(middle))
    after = np.linspace(center, end, num=after_count + 1)[1:] if end > center else np.array([center])
    values = sorted({round(float(value), 6) for value in np.concatenate((before, middle, after))})
    if len(values) < count:
        values = sorted(set(values) | {round(float(value), 6) for value in np.linspace(start, end, count)})
    if len(values) > count:
        chosen = {0, len(values) - 1, min(range(len(values)), key=lambda index: abs(values[index] - center))}
        for index in np.linspace(0, len(values) - 1, count, dtype=int):
            chosen.add(int(index))
        if len(chosen) > count:
            removable = sorted(chosen - {0, len(values) - 1}, key=lambda index: abs(values[index] - center), reverse=True)
            while len(chosen) > count:
                chosen.remove(removable.pop(0))
        values = [values[index] for index in sorted(chosen)]
    return values


def _sample_context(video_path: Path, center: float, fps: float, count: int = 6, bounds=None) -> list[tuple[float, bytes]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps > 0 else 0.0
    start = max(0.0, center - 3.0)
    end = min(duration, center + 3.0)
    if bounds is not None:
        start, end = bounds[0] / fps, bounds[1] / fps
    samples: list[tuple[float, bytes]] = []
    for second in context_sample_seconds(start, center, end, count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(second) * fps)))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        if frame.shape[1] > 480:
            scale = 480 / frame.shape[1]
            frame = cv2.resize(frame, (480, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        samples.append((float(second), _jpeg(frame, quality=72)))
    capture.release()
    return samples


class OpenAIAnalyzer:
    def __init__(self, root: Path):
        env = _load_project_env(root)
        ai_enabled = env.get("REALTIME_AI_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        self.api_key = env.get("OPENAI_API_KEY", "").strip() if ai_enabled else ""
        self.model = (env.get("OPENAI_VISION_MODEL") or env.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
        self.base_url = (env.get("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _chat(self, content: Any, max_tokens: int = 400) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "당신은 CCTV 영상 관제 보조자입니다. 화면에 보이는 근거만 설명하고 "
                            "보이지 않는 행동이나 의도를 추측하지 마세요."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()

    def describe(self, video_path: Path, local_second: float, global_second: float, fps: float, bounds=None) -> tuple[str, str | None]:
        if not self.enabled:
            return "이 시간대에서 장면 변화가 감지되었습니다. AI 행동 분석은 API 설정 후 제공됩니다.", "openai_api_key_missing"
        frames = _sample_context(video_path, local_second, fps, count=12, bounds=bounds)
        if not frames:
            return "분석할 전후 프레임을 불러오지 못했습니다.", "context_frames_unavailable"
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"CCTV의 {global_second:.1f}초 주변에서 이전 상태, 변화 순간, 이후 상태를 보여 주는 "
                    "프레임이 시간순으로 제공됩니다. 프레임 전체의 순서를 비교하여 직접 확인되는 핵심 변화를 "
                    "한국어 한 문장으로 설명하세요. 주체와 관찰된 동작 또는 상태 변화만 쓰세요. "
                    "보이지 않는 원인, 의도, 감정, 신원이나 이후 결과를 추측하지 마세요. "
                    "카메라가 이동하거나 가려진 것만 보이면 그 화면 변화 자체를 설명하세요. "
                    "사람·차량·물체의 변화가 명확하지 않으면 관찰 가능한 화면 상태를 구체적으로 설명하세요. "
                    "서문, 목록, 타임스탬프 없이 한 문장만 출력하세요."
                ),
            }
        ]
        for second, jpeg in frames:
            content.append({"type": "text", "text": f"+{second:.2f}초 프레임"})
            encoded = base64.b64encode(jpeg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"}})
        try:
            return self._chat(content, max_tokens=160), None
        except Exception as error:
            return "장면 변화는 감지되었으나 AI 행동 설명을 생성하지 못했습니다.", type(error).__name__

    def summarize(self, observations: Iterable[dict[str, Any]], duration: float) -> tuple[str, str | None]:
        rows = [
            f"- {item.get('timestamp', 0):.1f}초: {item.get('caption', '')}"
            for item in observations
            if item.get("caption") and not item.get('caption_error') and item.get('caption_status', 'ok') == 'ok'
        ]
        if not rows:
            return "아직 요약할 행동 관찰 결과가 없습니다.", None
        if not self.enabled:
            return f"총 {duration:.0f}초의 영상에서 {len(rows)}개 대표 시간대를 추출했습니다.", "openai_api_key_missing"
        prompt = (
            f"총 {duration:.1f}초 CCTV 영상의 시간대별 관찰 결과입니다.\n"
            + "\n".join(rows[-40:])
            + "\n전체 흐름을 2~4문장의 간결한 한국어 관제 요약으로 작성하세요. "
            "동일한 내용을 반복하지 말고, 관찰 결과에 없는 사건은 추가하지 마세요."
        )
        try:
            return self._chat(prompt, max_tokens=280), None
        except Exception as error:
            return f"총 {duration:.0f}초의 영상에서 {len(rows)}개 대표 시간대를 추출했습니다.", type(error).__name__


def _url(session_id: str, relative_path: Path) -> str:
    return f"/api/sessions/{session_id}/media/{relative_path.as_posix()}"


def motion_regions(times, motions, features):
    """Retain brief changes; merge only nearby, visually compatible regions."""
    regions = []
    start = last = None
    for index, motion in enumerate(motions):
        if start is None:
            if motion >= MOTION_THRESHOLD_PERCENT:
                start = last = index
        elif motion >= MOTION_END_PERCENT:
            last = index
        elif times[index] - times[last] >= QUIET_SECONDS:
            regions.append((start, last))
            start = last = None
    if start is not None:
        regions.append((start, last))
    merged = []
    for start, end in regions:
        if merged and times[start] - times[merged[-1][1]] <= MERGE_GAP_SECONDS and float(
            np.mean((features[start] - features[merged[-1][1]]) ** 2)
        ) < FEATURE_DISTANCE:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    # A sustained visual transition can divide a long interval; duration alone cannot.
    result = []
    for start, end in merged:
        anchor = start
        for index in range(start + 1, end):
            if times[index] - times[anchor] < 15:
                continue
            future = min(end, int(np.searchsorted(times, times[index] + 1)))
            before = features[max(anchor, index - 1)]
            if np.mean((before - features[index]) ** 2) > FEATURE_DISTANCE and np.mean(
                (before - features[future]) ** 2
            ) > FEATURE_DISTANCE:
                result.append((anchor, index - 1))
                anchor = index
        result.append((anchor, end))
    return result


def refine_action_bounds(times, motions, start: int, end: int) -> tuple[int, int, int]:
    """Center a bounded evidence window on the strongest transition in a motion region."""
    peak = max(range(start, end + 1), key=lambda index: motions[index])
    threshold = max(MOTION_END_PERCENT, motions[peak] * .25)
    active = [index for index in range(start, end + 1) if motions[index] >= threshold]
    core_start, core_end = (active[0], active[-1]) if active else (peak, peak)
    desired_start = times[core_start] - ACTION_CONTEXT_SECONDS
    desired_end = times[core_end] + ACTION_CONTEXT_SECONDS
    if desired_end - desired_start > MAX_ACTION_SECONDS:
        desired_start = times[peak] - MAX_ACTION_SECONDS / 2
        desired_end = times[peak] + MAX_ACTION_SECONDS / 2
    left = max(0, int(np.searchsorted(times, desired_start, side='left')))
    right = min(len(times) - 1, int(np.searchsorted(times, desired_end, side='right') - 1))
    return left, right, peak


def exclude_camera_regions(regions, camera_motions, minimum_ratio: float = .5):
    """Drop regions dominated by whole-frame camera movement or occlusion."""
    kept = []
    for start, end in regions:
        ratio = float(np.mean([bool(item['camera_motion']) for item in camera_motions[start:end + 1]]))
        if ratio < minimum_ratio:
            kept.append((start, end))
    return kept


def temporal_keyshot_regions(labels) -> list[tuple[int, int, int]]:
    """Return one keyshot window for every chronological visual-state run.

    A hierarchical component may occur more than once (for example, a camera
    moves from a desk to a doorway and then back to the desk). Treating all
    occurrences as one component is fine for a static representative-image
    gallery, but it erases the order of events in a CCTV skim. Keep each
    contiguous occurrence here so the summary video can show A -> B -> A.
    """
    labels = np.asarray(labels)
    if not len(labels):
        return []
    starts = [0] + (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()
    regions = []
    for start, stop in zip(starts, starts[1:] + [len(labels)]):
        end = stop - 1
        regions.append((start, end, (start + end) // 2))
    return regions


def join_clips(paths, destination, fps=RECORDING_FPS):
    writer = None
    try:
        for path in paths:
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise RuntimeError('Cannot read continuation clip')
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if writer is None:
                        size = (frame.shape[1], frame.shape[0])
                        writer = _video_writer(destination, fps, size)
                    writer.write(cv2.resize(frame, size))
            finally:
                capture.release()
    finally:
        if writer is not None:
            writer.release()


def deduplicate_keyshots(shots, features, qualities, threshold=FEATURE_DISTANCE):
    """Compare each selected clip's start/middle/end; keep the clearer duplicate.

    This is application postprocessing after HieTaSkim selection. Compare against
    retained clips only, so chains of weak matches cannot erase distinct shots.
    """
    if not shots:
        return []
    signatures = [np.asarray([features[index] for index in
                              (left, (left + right) // 2, right)])
                  for left, right, _ in shots]
    ranked = sorted(range(len(shots)), key=lambda index: (
        -int(qualities[shots[index][2]].get('usable', False)),
        -float(qualities[shots[index][2]].get('score', 0)),
        shots[index][0],
    ))
    kept = []
    for index in ranked:
        if not any(np.all(np.mean((signatures[index] - signatures[other]) ** 2, axis=1) < threshold)
                   for other in kept):
            kept.append(index)
    return sorted((shots[index] for index in kept), key=lambda shot: shot[0])


def analyze_recorded_segment(
    *,
    root: Path,
    output_dir: Path,
    session_id: str,
    segment: RecordedSegment,
    batch_id: int,
    prior_observations: list[dict[str, Any]],
    interval: int,
    reason: str,
    captions_enabled: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    analysis_dir = output_dir / "analysis"
    keyframe_dir = analysis_dir / "keyframes"
    clip_dir = analysis_dir / "clips"
    summary_dir = analysis_dir / "summary_videos"
    for folder in (keyframe_dir, clip_dir, summary_dir):
        folder.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(segment.path))
    if not capture.isOpened():
        raise RuntimeError("저장된 실시간 영상을 다시 열 수 없습니다.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or segment.fps or RECORDING_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    summary_path = summary_dir / f"summary_{batch_id:04d}{BROWSER_VIDEO_SUFFIX}"
    previous_gray: np.ndarray | None = None
    motion_frames = 0
    first_frame: np.ndarray | None = None
    buckets: dict[int, FrameCandidate] = {}
    times, motions, raw_motions, camera_motions, features, qualities = [], [], [], [], [], []
    duration = max(segment.end - segment.start, total_frames / fps if fps > 0 else 0.0)
    bucket_seconds = max(1.0, duration / MAX_KEYFRAME_CANDIDATES)

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if first_frame is None:
            first_frame = frame.copy()
        small_width = min(320, frame.shape[1])
        scale = small_width / frame.shape[1]
        small = cv2.resize(frame, (small_width, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        motion_result = compensated_motion(previous_gray, gray)
        motion = motion_result['motion_percent']
        previous_gray = gray
        local_second = frame_index / fps if fps > 0 else 0.0
        timestamp = segment.timestamps[frame_index] if frame_index < len(segment.timestamps) else segment.start + local_second
        times.append(timestamp)
        motions.append(motion)
        raw_motions.append(motion_result['raw_motion_percent'])
        camera_motions.append(motion_result)
        features.append(_feature(frame))
        quality = frame_quality(small)
        qualities.append(quality)
        if motion >= MOTION_THRESHOLD_PERCENT:
            motion_frames += 1
            bucket = int(local_second / bucket_seconds)
            candidate = FrameCandidate(
                frame_index=frame_index,
                local_second=local_second,
                timestamp=timestamp,
                motion_percent=motion,
                jpeg=_jpeg(frame),
                feature=_feature(frame),
                quality=quality,
            )
            if quality['usable'] and (bucket not in buckets or motion + quality['score'] >
                                      buckets[bucket].motion_percent + buckets[bucket].quality['score']):
                buckets[bucket] = candidate
        frame_index += 1
    capture.release()

    candidates = sorted(buckets.values(), key=lambda item: item.timestamp)
    if not candidates and first_frame is not None:
        candidates = [
            FrameCandidate(0, 0.0, segment.start, 0.0, _jpeg(first_frame), _feature(first_frame))
        ]
    caption_env = _load_project_env(root)
    delta_t = float(caption_env.get('HIETASKIM_DELTA_T', '4.0'))
    gamma = float(caption_env.get('HIETASKIM_GAMMA', '0.75'))
    summary_ratio = float(caption_env.get('HIETASKIM_SUMMARY_RATIO', '0.15'))
    hierarchy = caption_env.get('HIETASKIM_HIERARCHY', 'area')
    min_components = int(caption_env.get('HIETASKIM_MIN_COMPONENTS', '3'))
    sampled = sample_indices(times)
    descriptors = extract_descriptors(segment.path, sampled)
    sampled_labels = hierarchical_regions(np.asarray(times)[sampled], descriptors, delta_t=delta_t,
                                         gamma=gamma, min_components=min_components, hierarchy=hierarchy,
                                         return_labels=True)
    labels = np.repeat(sampled_labels, np.diff(np.append(sampled, len(times))))
    # Summary video is a chronological story, not a deduplicated image gallery.
    # A return to an earlier-looking view therefore remains meaningful.
    regions = temporal_keyshot_regions(labels)
    shots = select_keyshots(times, regions, fps, summary_ratio)
    shots_before_dedup = len(shots)
    visual_duplicate_count = shots_before_dedup - len(deduplicate_keyshots(shots, features, qualities))
    selected = []
    region_bounds = []
    selected_regions = []
    capture = cv2.VideoCapture(str(segment.path))
    for left, right, transition_peak in shots:
        peak = transition_peak
        capture.set(cv2.CAP_PROP_POS_FRAMES, peak)
        ok, frame = capture.read()
        if not ok:
            continue
        selected.append(FrameCandidate(peak, peak / fps, times[peak], motions[peak], _jpeg(frame), features[peak], qualities[peak]))
        region_bounds.append((left, right))
        selected_regions.append((left, right, transition_peak))
    capture.release()
    if not selected and times and not any(item['usable'] for item in qualities):
        # Keep one diagnostic result, but do not send an unusable frame to a VLM.
        peak = max(range(len(times)), key=lambda index: qualities[index]['score'])
        capture = cv2.VideoCapture(str(segment.path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, peak)
        ok, frame = capture.read()
        capture.release()
        if ok:
            selected.append(FrameCandidate(peak, peak / fps, times[peak], motions[peak], _jpeg(frame), features[peak], qualities[peak]))
            region_bounds.append((peak, peak))
            selected_regions.append((peak, peak, peak))
    ai = OpenAIAnalyzer(root) if captions_enabled else None
    use_livecc = captions_enabled and caption_env.get('CAPTION_BACKEND', 'openai').lower() == 'livecc'
    ai_ready = captions_enabled and total_frames >= 8
    records: list[dict[str, Any]] = []
    skim_paths = []
    shot_bounds = {(left, right) for left, right, _ in shots}
    for index, item in enumerate(selected, start=1):
        identifier = f"b{batch_id:04d}_{index:02d}"
        image_path = keyframe_dir / f"{identifier}.jpg"
        clip_path = clip_dir / f"{identifier}{BROWSER_VIDEO_SUFFIX}"
        image_path.write_bytes(item.jpeg)
        bounds = region_bounds[index - 1]
        transition_index = selected_regions[index - 1][2]
        transition_local_second = transition_index / fps
        transition_timestamp = times[transition_index]
        clip_saved = _write_clip(segment.path, clip_path, item.local_second, fps, bounds=bounds, timestamps=times)
        if clip_saved and bounds in shot_bounds:
            skim_paths.append(clip_path)
        caption_details = None
        if not captions_enabled:
            caption, caption_error = '분석 로직 테스트 · 캡션 생성 생략', None
            caption_details = {'status': 'disabled'}
        elif not item.quality.get('usable', True):
            reasons = ', '.join(item.quality.get('reasons', [])) or 'low_quality'
            caption, caption_error = "화면 품질이 낮아 행동을 확인할 수 없습니다.", 'quality_rejected:' + reasons
            caption_details = {'status': 'quality_rejected', 'quality': item.quality}
        elif use_livecc:
            from livecc_jobs import enqueue
            from livecc_caption import MODEL_ID
            job_id = enqueue(session_id, batch_id, identifier, output_dir, item.timestamp, dict(
                mode=caption_env.get('LIVECC_CLIP_MODE', 'centered'),
                seconds=float(caption_env.get('LIVECC_CLIP_SECONDS', '2')),
                count=int(caption_env.get('LIVECC_FRAME_COUNT', '4'))))
            caption_details = {'status': 'pending', 'job_id': job_id, 'model_id': MODEL_ID}
            caption, caption_error = '설명 생성 대기 중', 'pending'
        elif ai_ready:
            caption, caption_error = ai.describe(
                segment.path, transition_local_second, transition_timestamp, fps, bounds=bounds
            )
        else:
            caption = "수집 시간이 짧아 장면 변화만 기록했습니다. 다음 분석 구간에서 행동 설명을 생성합니다."
            caption_error = None
        records.append(
            {
                "id": identifier,
                "segment_id": len(prior_observations) + index - 1,
                "batch_id": batch_id,
                "timestamp": round(item.timestamp, 2),
                "start_time": round(times[bounds[0]], 2),
                "end_time": round(times[bounds[1]], 2),
                "source_frame_bounds": list(bounds),
                "brief_change": times[bounds[1]] - times[bounds[0]] < 0.5,
                "continues_to_boundary": selected_regions[index - 1][1] >= len(times) - 2,
                "begins_at_boundary": selected_regions[index - 1][0] <= 1,
                "ending_feature": features[-1].tolist() if selected_regions[index - 1][1] >= len(times) - 2 else None,
                "captured_at": None,
                "title": "시간대별 행동 관찰",
                "caption": caption,
                "status": "unreviewed",
                "motion_percent": round(item.motion_percent, 2),
                "raw_motion_percent": round(raw_motions[item.frame_index], 2),
                "camera_motion": camera_motions[item.frame_index]['camera_motion'],
                "camera_shift_pixels": camera_motions[item.frame_index]['camera_shift_pixels'],
                "transition_timestamp": round(transition_timestamp, 2),
                "quality_status": item.quality.get('status', 'unknown'),
                "quality_score": item.quality.get('score'),
                "quality_reasons": item.quality.get('reasons', []),
                "frame_quality": item.quality,
                "analysis_method": "hietaskim_vlm" if captions_enabled else "hietaskim",
                "selected_frame_copy": str(image_path),
                "image_url": _url(session_id, image_path.relative_to(output_dir)),
                "clip_url": _url(session_id, clip_path.relative_to(output_dir)) if clip_saved else None,
                "caption_error": caption_error,
                "caption_details": caption_details,
                "caption_status": caption_details['status'] if caption_details else ('error' if caption_error else 'ok'),
                "caption_model": caption_details.get('model_id') if caption_details else ai.model,
            }
        )

    # Concatenate only selected continuous shots, preserving their playback rate.
    if skim_paths:
        join_clips(skim_paths, summary_path, fps=fps)
    replaced_ids = []
    observations = [row for row in prior_observations if row['id'] not in replaced_ids] + records
    if ai_ready:
        ai_summary, summary_error = ai.summarize(records, segment.end - segment.start)
    else:
        ai_summary = f"현재까지 {len(observations)}개 대표 시간대를 추출했습니다."
        summary_error = None
    recording_relative = segment.path.relative_to(output_dir)
    summary_relative = summary_path.relative_to(output_dir)
    batch = {
        "id": batch_id,
        "start": round(segment.start, 2),
        "end": round(segment.end, 2),
        "count": len(records),
        "candidate_count": len(candidates),
        "interval_seconds": interval,
        "reason": reason,
        "recording_url": _url(session_id, recording_relative),
        "summary_video_url": _url(session_id, summary_relative) if summary_path.exists() else None,
        "source_frames": total_frames,
        "skimming": {"method": "hietaskim", "delta_t_seconds": delta_t, "gamma": gamma,
                     "hierarchy": hierarchy, "min_components": min_components,
                     "descriptor": "keras_resnet50_imagenet_predictions", "sample_fps": 2.0,
                     "summary_ratio": summary_ratio, "region_count": len(regions),
                     "clips_before_dedup": shots_before_dedup, "clips_after_dedup": len(shots),
                     "duplicates_removed": 0,
                     "visual_duplicates_retained_for_timeline": visual_duplicate_count,
                     "dedup_feature_mse_threshold": FEATURE_DISTANCE},
        "motion_frames": motion_frames,
        "camera_motion_frames": sum(bool(item['camera_motion']) for item in camera_motions),
        "quality_rejected_frames": sum(not item['usable'] for item in qualities),
        "completed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    report = {
        "status": "ready",
        "ai_summary": ai_summary,
        "ai_model": ai.model if ai and ai.enabled else None,
        "ai_error": summary_error,
        "latest_summary_video_url": batch["summary_video_url"],
        "latest_recording_url": batch["recording_url"],
        "recorded_seconds": round(segment.end, 2),
        "source_frames": total_frames,
        "motion_frames": motion_frames,
        "observation_count": len(observations),
        "updated_at": batch["completed_at"],
        "replaced_observation_ids": replaced_ids,
    }
    _atomic_json(output_dir / 'history' / f'batch_{batch_id:04d}.json', {
        'batch': batch, 'observations': records, 'report': report,
    })
    return batch, records, report


def save_analysis_files(output_dir: Path, observations: list[dict[str, Any]], batches: list[dict[str, Any]], report: dict[str, Any], created_at: str) -> None:
    _atomic_json(output_dir / "summary.json", observations)
    _atomic_json(output_dir / "batches.json", batches)
    _atomic_json(output_dir / "analysis.json", {**report, "observations": observations, "batches": batches})
    _atomic_json(
        output_dir / "metadata.json",
        {
            "source": "phone_camera",
            "duration_sec": report.get("recorded_seconds", 0),
            "created_at": created_at,
            "method": "realtime_recording_hietaskim_vlm_summary",
            "continuous_video_saved": True,
            "recording_segments": [item.get("recording_url") for item in batches],
            "summary_videos": [item.get("summary_video_url") for item in batches],
        },
    )
