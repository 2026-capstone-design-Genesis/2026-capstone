"""QR 카메라 연결, 실시간 프레임 수신, 주기별 키프레임 분석 API."""
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from functools import partial
from pathlib import Path
import secrets
import time
from urllib.parse import urlsplit

import cv2
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from realtime_analysis import RECORDING_FPS, SegmentRecorder, analyze_recorded_segment, save_analysis_files

ROOT = Path(__file__).resolve().parent
INTERVALS = (30, 60, 300, 600, 1800, 3600)
MAX_FRAME_BYTES = 400_000
MAX_CANDIDATES = 360
MAX_ACTIVE_SESSIONS = 4
COOKIE = "clullm_controller"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Candidate:
    second: float
    captured_at: str
    jpeg: bytes
    feature: np.ndarray
    motion: float


def decode_frame(jpeg: bytes) -> tuple[np.ndarray, np.ndarray]:
    """크기를 먼저 검사해 제한된 JPEG만 분석한다."""
    if not jpeg or len(jpeg) > MAX_FRAME_BYTES:
        raise ValueError("영상 프레임 크기가 허용 범위를 넘었습니다.")
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.format != "JPEG" or max(image.size) > 1280 or min(image.size) < 16:
                raise ValueError("1280px 이하 JPEG 프레임만 전송할 수 있습니다.")
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("영상 프레임을 읽을 수 없습니다.") from error
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("영상 프레임을 해석할 수 없습니다.")
    small = cv2.resize(frame, (160, 90))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # 공간 구조와 색상 분포를 함께 비교한다. 사건 위험도는 추론하지 않는다.
    thumb = cv2.resize(small, (16, 9)).astype(np.float32).ravel() / 255
    histogram = cv2.calcHist([small], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3).ravel()
    histogram /= max(float(histogram.sum()), 1)
    return gray, np.concatenate((thumb, histogram * 4))


def select_keyframes(candidates: list[Candidate], count: int = 8) -> list[Candidate]:
    """시간 간격을 확보하며 서로 다른 시각 특징을 가진 대표 장면을 고른다."""
    if len(candidates) <= count:
        return sorted(candidates, key=lambda item: item.second)
    features = np.stack([item.feature for item in candidates])
    chosen = [int(np.argmax([item.motion for item in candidates]))]
    while len(chosen) < count:
        distance = np.min(np.stack([np.mean((features - features[i]) ** 2, axis=1) for i in chosen]), axis=0)
        # 거의 동일한 장면에서는 시간적으로 고르게 분산된 프레임을 우선한다.
        temporal = np.array([min(abs(item.second - candidates[i].second) for i in chosen) for item in candidates])
        distance += .002 * temporal / max(1, float(temporal.max()))
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    return sorted((candidates[i] for i in chosen), key=lambda item: item.second)


class Session:
    def __init__(self, interval: int, output_root: Path, *, label: str = "카메라 01", slot: int = 1,
                 frame_executor: ThreadPoolExecutor | None = None,
                 analysis_executor: ProcessPoolExecutor | None = None,
                 summary_executor: ThreadPoolExecutor | None = None):
        self.id = secrets.token_urlsafe(16)
        self.phone_token = secrets.token_urlsafe(32)
        self.created_at = utc_now()
        self.created_mono = time.monotonic()
        self.expires_mono = self.created_mono + 86400
        self.interval = interval
        # 표시 전용 정보다. 경로와 인증 토큰에는 절대 사용하지 않는다.
        self.label = label
        self.slot = slot
        self.frame_executor = frame_executor
        self.analysis_executor = analysis_executor
        self.summary_executor = summary_executor
        self.elapsed = 0.0
        self.next_at = float(interval)
        self.window_start = 0.0
        self.last_frame_mono: float | None = None
        self.last_recorded_mono: float | None = None
        self.previous_gray: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
        self.frame_version = 0
        self.received = 0
        self.motion = 0.0
        self.phone: WebSocket | None = None
        self.viewers = 0
        self.stopped = False
        self.busy = False
        self.error: str | None = None
        self.candidates: list[Candidate] = []
        self.batches: list[dict] = []
        self.summary: list[dict] = []
        self.analysis_report: dict = {
            "status": "idle",
            "ai_summary": "아직 완료된 영상 분석이 없습니다.",
            "observation_count": 0,
        }
        self.task: asyncio.Task | None = None
        self.run_name = f"phone_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.id[:8]}"
        self.output_dir = output_root / self.run_name
        self.recorder = SegmentRecorder(self.output_dir)
        self.lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self.phone is not None and self.last_frame_mono is not None and time.monotonic() - self.last_frame_mono < 3

    def snapshot(self) -> dict:
        self.refresh_caption_jobs()
        from livecc_jobs import get_worker_status
        state = "ended" if self.stopped else ("live" if self.connected else ("reconnecting" if self.received else "waiting"))
        return {
            "id": self.id, "label": self.label, "slot": self.slot, "created_at": self.created_at, "status": state,
            "interval_seconds": self.interval, "elapsed_seconds": round(self.elapsed, 2),
            "next_analysis_in": max(0, round(self.next_at - self.elapsed)),
            "pending_candidates": len(self.candidates), "received_frames": self.received,
            "motion_percent": round(self.motion, 1), "analyzing": self.busy,
            "analysis_status": self.analysis_report.get("status", "idle"), "error": self.error,
            "batches": [batch for batch in self.batches if not batch.get('deleted')], "keyframes": self.summary[-200:],
            "total_keyframes": len(self.summary), "run_name": self.run_name,
            "video_saved": bool(self.batches) or self.recorder.has_frames,
            "analysis_report": self.analysis_report,
            "caption_queue_count": sum(row.get('caption_status') in ('pending', 'waiting', 'running') for row in self.summary),
            "caption_worker": get_worker_status(),
            "phone_expires_in": max(0, int(self.expires_mono - time.monotonic())),
        }

    def refresh_caption_jobs(self):
        changed = False
        completed_batches = set()
        for record in self.summary:
            details = record.get('caption_details') or {}
            if not details.get('job_id') or record.get('caption_status') not in ('pending', 'waiting', 'running'):
                continue
            from livecc_jobs import get_job
            job = get_job(details['job_id'])
            if not job:
                continue
            result = json.loads(job['result']) if job['result'] else details
            status = result.get('status') if job['state'] == 'done' else job['state']
            if status == record.get('caption_status'):
                continue
            changed = True
            if job['state'] == 'done':
                completed_batches.add(record['batch_id'])
            record['caption_status'] = status
            record['caption_details'] = result
            record['caption_error'] = None if status == 'ok' else status
            record['caption'] = result.get('caption') or {'pending': '설명 생성 대기 중', 'running': '설명 생성 중', 'waiting': '다음 영상 프레임 대기 중'}.get(status, '설명 생성 실패: ' + status)
            if result.get('caption_clip_url'):
                record['action_clip_url'] = record.get('action_clip_url') or record.get('clip_url')
                record['clip_url'] = result['caption_clip_url']
            elif result.get('clip_export_error'):
                record['action_clip_url'] = record.get('action_clip_url') or record.get('clip_url')
                record['clip_url'] = None
            history = self.output_dir / 'history' / f"batch_{record['batch_id']:04d}.json"
            if history.exists():
                data = json.loads(history.read_text(encoding='utf-8'))
                data['observations'] = [record if item['id'] == record['id'] else item for item in data['observations']]
                from realtime_analysis import _atomic_json
                _atomic_json(history, data)
        if changed:
            save_analysis_files(self.output_dir, self.summary, self.batches, self.analysis_report, self.created_at)
        for batch_id in completed_batches:
            records = [record for record in self.summary if record['batch_id'] == batch_id]
            if any(record.get('caption_status') in ('pending', 'waiting', 'running') for record in records):
                continue
            if not hasattr(self, 'caption_summary_tasks'):
                self.caption_summary_tasks = {}
            if batch_id not in self.caption_summary_tasks:
                self.caption_summary_tasks[batch_id] = asyncio.create_task(self.finish_caption_summary(batch_id, records))

    async def finish_caption_summary(self, batch_id, records):
        from realtime_analysis import OpenAIAnalyzer, _atomic_json
        batch = next(item for item in self.batches if item['id'] == batch_id)
        work = partial(OpenAIAnalyzer(ROOT).summarize, records, batch['end'] - batch['start'])
        summary, error = await asyncio.get_running_loop().run_in_executor(self.summary_executor, work)
        async with self.lock:
            history = self.output_dir / 'history' / f'batch_{batch_id:04d}.json'
            if history.exists():
                data = json.loads(history.read_text(encoding='utf-8'))
                data['report'].update(ai_summary=summary, ai_error=error, caption_summary_finalized=True)
                _atomic_json(history, data)
            if self.batches[-1]['id'] == batch_id:
                self.analysis_report.update(ai_summary=summary, ai_error=error)
                save_analysis_files(self.output_dir, self.summary, self.batches, self.analysis_report, self.created_at)

    async def receive(self, jpeg: bytes) -> None:
        # Publish every incoming frame, but decode and archive only at the analysis FPS.
        if not jpeg or len(jpeg) > MAX_FRAME_BYTES:
            raise ValueError("프레임 크기가 허용 범위를 벗어났습니다.")
        async with self.lock:
            if self.stopped:
                return
            now = time.monotonic()
            if self.last_frame_mono is not None:
                gap = now - self.last_frame_mono
                # 수신이 끊긴 시간은 분석 구간에 포함하지 않는다.
                if gap < 3:
                    self.elapsed += gap
                else:
                    self.previous_gray = None
                    self.last_recorded_mono = None
            self.last_frame_mono = now
            self.latest_jpeg = jpeg
            self.frame_version += 1
            self.received += 1
            should_process = self.last_recorded_mono is None or now - self.last_recorded_mono >= 1 / RECORDING_FPS
            if not should_process:
                if self.elapsed >= self.next_at and not self.busy and not self.error:
                    self.start_analysis("scheduled")
                return

        # 수신 WebSocket은 즉시 다음 프레임을 받는다. JPEG 검증·디코딩은 전용 스레드에서 한다.
        gray, feature = await asyncio.get_running_loop().run_in_executor(self.frame_executor, decode_frame, jpeg)
        async with self.lock:
            if self.stopped:
                return
            self.motion = float(np.mean(cv2.absdiff(gray, self.previous_gray) > 18) * 100) if self.previous_gray is not None else 0.0
            self.previous_gray = gray
            await asyncio.get_running_loop().run_in_executor(self.frame_executor, self.recorder.write, jpeg, self.elapsed)
            self.last_recorded_mono = now
            candidate = Candidate(self.elapsed, utc_now(), jpeg, feature, self.motion)
            # 10초 단위로 변화가 큰 프레임 하나만 보관한다. 최대 한 시간에도 메모리가 제한된다.
            if self.candidates and int(self.candidates[-1].second / 10) == int(self.elapsed / 10):
                if candidate.motion >= self.candidates[-1].motion:
                    self.candidates[-1] = candidate
            else:
                self.candidates.append(candidate)
                if len(self.candidates) > MAX_CANDIDATES:
                    self.candidates.pop(0)
            if self.elapsed >= self.next_at and not self.busy and not self.error:
                self.start_analysis("scheduled")

    def start_analysis(self, reason: str, *, allow_stopped: bool = False) -> bool:
        if self.busy or not self.candidates:
            return False
        samples, self.candidates = self.candidates, []
        start, end = self.window_start, self.elapsed
        self.window_start, self.next_at = end, end + self.interval
        self.busy, self.error = True, None
        self.analysis_report = {**self.analysis_report, "status": "running"}
        recording = self.recorder.rotate(end)
        self.task = asyncio.create_task(self.analyze(samples, recording, start, end, self.interval, reason))
        return True

    async def analyze(self, samples: list[Candidate], recording, start: float, end: float, interval: int, reason: str):
        try:
            if recording is None:
                raise RuntimeError("분석할 저장 영상이 없습니다.")
            work = partial(
                analyze_recorded_segment,
                root=ROOT,
                output_dir=self.output_dir,
                session_id=self.id,
                segment=recording,
                batch_id=len(self.batches) + 1,
                prior_observations=list(self.summary),
                interval=interval,
                reason=reason,
            )
            # 카메라별 요약은 독립 프로세스에서 실행한다. GIL과 프레임 수신 루프를 막지 않는다.
            batch, records, report = await asyncio.get_running_loop().run_in_executor(self.analysis_executor, work)
            async with self.lock:
                self.batches.append(batch)
                replaced = set(report.get('replaced_observation_ids', []))
                self.summary = [row for row in self.summary if row['id'] not in replaced]
                self.summary.extend(records)
                report["source_frames"] = sum(int(item.get("source_frames", 0)) for item in self.batches)
                report["motion_frames"] = sum(int(item.get("motion_frames", 0)) for item in self.batches)
                self.analysis_report = report
                save_analysis_files(
                    self.output_dir,
                    self.summary,
                    self.batches,
                    self.analysis_report,
                    self.created_at,
                )
                self.busy = False
                self.analysis_report = {**self.analysis_report, "status": "complete"}
        except Exception:
            async with self.lock:
                self.candidates = sorted(samples + self.candidates, key=lambda item: item.second)[-MAX_CANDIDATES:]
                self.window_start = start
                self.busy = False
                self.analysis_report = {**self.analysis_report, "status": "error"}
                self.error = "키프레임 저장에 실패했습니다. 저장 공간을 확인하고 ‘지금 분석’을 눌러 다시 시도하세요."

    def save_batch(self, samples: list[Candidate], start: float, end: float, interval: int, reason: str):
        selected = select_keyframes(samples)
        batch_id = len(self.batches) + 1
        folder = self.output_dir / "selected_frames"
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for index, frame in enumerate(selected):
            identifier = f"b{batch_id:04d}_{index + 1:02d}"
            filename = identifier + ".jpg"
            destination = folder / filename
            destination.write_bytes(frame.jpeg)
            records.append({
                "id": identifier, "segment_id": len(self.summary) + index, "batch_id": batch_id,
                "timestamp": round(frame.second, 2), "start_time": round(frame.second, 2),
                "end_time": round(frame.second, 2), "captured_at": frame.captured_at,
                "title": "장면 변화" if frame.motion >= 3 else "대표 장면",
                "caption": "프레임 변화량과 시각적 다양성으로 선택한 키프레임입니다.",
                "status": "unreviewed", "motion_percent": round(frame.motion, 2),
                "analysis_method": "visual_diversity", "selected_frame_copy": str(destination),
                "image_url": f"/api/sessions/{self.id}/frames/{filename}",
            })
        batch = {"id": batch_id, "start": round(start, 2), "end": round(end, 2), "count": len(records),
                 "candidate_count": len(samples), "interval_seconds": interval, "reason": reason, "completed_at": utc_now()}
        # 기존 Streamlit 결과 조회와 호환되는 JSON 구조로 기록한다.
        files = {
            "summary.json": self.summary + records,
            "batches.json": self.batches + [batch],
            "metadata.json": {"source": "phone_camera", "video_path": None, "duration_sec": round(end, 2),
                              "created_at": self.created_at, "method": "실시간 시각 특징 기반 키프레임 추출", "continuous_video_saved": False},
        }
        for name, content in files.items():
            temporary = self.output_dir / (name + ".tmp")
            temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.output_dir / name)
        return batch, records

    async def finish(self):
        async with self.lock:
            self.stopped = True
        if self.phone:
            with suppress(RuntimeError, WebSocketDisconnect):
                await self.phone.close(code=1000, reason="관제 화면에서 연동을 종료했습니다.")
        if self.task and self.task is not asyncio.current_task():
            await self.task
        async with self.lock:
            self.start_analysis("stop", allow_stopped=True)
        if self.task and self.task is not asyncio.current_task():
            await self.task
        self.latest_jpeg = None
        self.previous_gray = None

        if self.stopped:
            from realtime_analysis import _atomic_json
            _atomic_json(self.output_dir / 'session-ended.json', {'ended': True})


class IntervalSetting(BaseModel):
    interval_seconds: int = 300
    label: str | None = Field(default=None, max_length=80)


class Login(BaseModel):
    token: str = Field(min_length=20, max_length=200)


def create_app(*, admin_token: str | None = None, public_origin: str | None = None, output_root: Path | None = None) -> FastAPI:
    controller_token = admin_token or secrets.token_urlsafe(32)
    public_origin = (public_origin or os.environ.get("CLULLM_PUBLIC_ORIGIN", "http://localhost:8765")).rstrip("/")
    sessions: dict[str, Session] = {}
    frame_executor = ThreadPoolExecutor(max_workers=MAX_ACTIVE_SESSIONS, thread_name_prefix="camera-frame")
    summary_executor = ThreadPoolExecutor(max_workers=MAX_ACTIVE_SESSIONS, thread_name_prefix="camera-summary")
    analysis_workers = max(1, min(MAX_ACTIVE_SESSIONS, int(os.environ.get("CLULLM_ANALYSIS_PROCESSES", MAX_ACTIVE_SESSIONS))))
    analysis_executor = ProcessPoolExecutor(max_workers=analysis_workers)
    output_root = output_root or ROOT / "outputs"

    @asynccontextmanager
    async def lifespan(app):
        async def cleanup():
            while True:
                await asyncio.sleep(30)
                for session_id, session in list(sessions.items()):
                    if time.monotonic() > session.expires_mono:
                        await session.finish()
                        sessions.pop(session_id, None)
        cleaner = asyncio.create_task(cleanup())
        yield
        cleaner.cancel()
        with suppress(asyncio.CancelledError):
            await cleaner
        for session in sessions.values():
            await session.finish()
        frame_executor.shutdown(wait=True, cancel_futures=True)
        summary_executor.shutdown(wait=True, cancel_futures=True)
        analysis_executor.shutdown(wait=True, cancel_futures=True)
        if app.state.upload_tasks:
            await asyncio.gather(*app.state.upload_tasks, return_exceptions=True)

    app = FastAPI(title="CluLLM 실시간 관제", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.sessions = sessions
    app.state.frame_executor = frame_executor
    app.state.summary_executor = summary_executor
    app.state.analysis_executor = analysis_executor
    app.state.upload_tasks = set()

    def same_origin(connection) -> bool:
        origin = connection.headers.get("origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        return parsed.netloc == connection.headers.get("host") and parsed.scheme in ("http", "https")

    def require_owner(connection):
        if not same_origin(connection) or not secrets.compare_digest(connection.cookies.get(COOKIE, ""), controller_token):
            raise HTTPException(401, "실행기에 표시된 관제 주소로 접속해 주세요.")

    def get_session(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "연결 세션이 없거나 만료되었습니다. 새 QR 코드를 발급해 주세요.")
        return session

    from upload_test import register_upload_tests
    register_upload_tests(app, require_owner, output_root, ROOT)

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
        return response

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/login")
    async def login(body: Login, request: Request, response: Response):
        if not same_origin(request) or not secrets.compare_digest(body.token, controller_token):
            raise HTTPException(401, "관제 접속 키가 유효하지 않습니다.")
        response.set_cookie(COOKIE, controller_token, httponly=True, secure=request.url.scheme == "https", samesite="strict", max_age=86400)
        return {"ok": True}

    @app.get("/api/config")
    async def config(request: Request):
        require_owner(request)
        return {"public_origin": public_origin, "intervals": INTERVALS,
                "sessions": [s.snapshot() for s in sessions.values() if not s.stopped]}

    @app.post("/api/sessions", status_code=201)
    async def new_session(body: IntervalSetting, request: Request):
        require_owner(request)
        if body.interval_seconds not in INTERVALS:
            raise HTTPException(422, "분석 주기는 30초, 1분, 5분, 10분, 30분, 1시간 중 선택해 주세요.")
        active = [s for s in sessions.values() if not s.stopped]
        if len(active) >= MAX_ACTIVE_SESSIONS:
            raise HTTPException(429, "동시에 연결할 수 있는 카메라는 최대 4대입니다.")
        used_slots = {s.slot for s in active}
        slot = next(slot for slot in range(1, MAX_ACTIVE_SESSIONS + 1) if slot not in used_slots)
        label = (body.label or "").strip() or f"카메라 {slot:02d}"
        session = Session(body.interval_seconds, output_root, label=label, slot=slot,
                          frame_executor=frame_executor, analysis_executor=analysis_executor,
                          summary_executor=summary_executor)
        sessions[session.id] = session
        return {**session.snapshot(), "phone_url": f"{public_origin}/phone#session={session.id}&token={session.phone_token}"}

    @app.get("/api/sessions/{session_id}")
    async def session_info(session_id: str, request: Request):
        require_owner(request)
        session = get_session(session_id)
        return {**session.snapshot(), "phone_url": f"{public_origin}/phone#session={session.id}&token={session.phone_token}"}

    @app.patch("/api/sessions/{session_id}/interval")
    async def change_interval(session_id: str, body: IntervalSetting, request: Request):
        require_owner(request)
        session = get_session(session_id)
        if body.interval_seconds not in INTERVALS or session.stopped:
            raise HTTPException(422, "연동 중인 카메라에서 허용된 주기를 선택해 주세요.")
        async with session.lock:
            session.interval = body.interval_seconds
            # 모아 둔 장면을 보존하고 변경 시점부터 다음 분석을 예약한다.
            session.next_at = session.elapsed + session.interval
        return session.snapshot()

    @app.post("/api/sessions/{session_id}/analyze")
    async def analyze_now(session_id: str, request: Request):
        require_owner(request)
        session = get_session(session_id)
        async with session.lock:
            if not session.start_analysis("manual"):
                raise HTTPException(409, "분석 중이거나 아직 수집된 장면이 없습니다.")
        return session.snapshot()

    @app.post("/api/sessions/{session_id}/stop")
    async def stop(session_id: str, request: Request):
        require_owner(request)
        session = get_session(session_id)
        await session.finish()
        return session.snapshot()

    def archive_path(run_name: str):
        path = (output_root / run_name).resolve()
        if path.parent != output_root.resolve() or not path.is_dir():
            raise HTTPException(404, 'Archive not found')
        return path

    def deleted_batches(root):
        return {int(path.stem) for path in (root / '.deleted_analysis').glob('*.json') if path.stem.isdigit()}

    @app.post('/api/history/delete')
    async def delete_history(request: Request):
        require_owner(request)
        payload = await request.json()
        items = payload.get('items') if isinstance(payload, dict) else None
        if not isinstance(items, list) or not 1 <= len(items) <= 500:
            raise HTTPException(400, '삭제할 분석을 1~500개 선택해 주세요.')
        targets = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('run'), str) or not isinstance(item.get('file'), str):
                raise HTTPException(400, '잘못된 분석 항목입니다.')
            root = archive_path(item['run'])
            name = item['file']
            prefix = 'batch_' if name.startswith('batch_') else 'legacy_'
            number = name[len(prefix):-5]
            if not name.startswith(prefix) or not name.endswith('.json') or not number.isdigit():
                raise HTTPException(400, '잘못된 분석 파일입니다.')
            batch_id = int(number)
            active = next((session for session in sessions.values() if session.output_dir.resolve() == root), None)
            if active and (active.busy or any(row.get('batch_id') == batch_id and row.get('caption_status') in ('pending', 'running', 'waiting') for row in active.summary)):
                raise HTTPException(409, '분석 또는 캡션 생성 중입니다. 완료 후 삭제해 주세요.')
            if batch_id not in deleted_batches(root):
                if prefix == 'batch_':
                    path = (root / 'history' / name).resolve()
                    if path.parent != root / 'history' or not path.is_file():
                        raise HTTPException(404, 'Archive not found')
                else:
                    source = root / 'analysis.json'
                    if not source.is_file() or not any(row.get('id') == batch_id for row in json.loads(source.read_text(encoding='utf-8')).get('batches', [])):
                        raise HTTPException(404, 'Archive not found')
            targets.append((root, batch_id))
        from realtime_analysis import _atomic_json
        from history_deletion import delete_batch_files
        removed_files = 0
        for root, batch_id in set(targets):
            try:
                removed_files += delete_batch_files(root, batch_id)
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            except OSError as exc:
                raise HTTPException(409, '파일을 삭제하지 못했습니다. 재생 중인 클립을 닫고 다시 시도해 주세요.') from exc
            _atomic_json(root / '.deleted_analysis' / f'{batch_id}.json', {'id': batch_id, 'deleted_at': utc_now()})
            for session in sessions.values():
                if session.output_dir.resolve() == root:
                    session.summary = [row for row in session.summary if row.get('batch_id') != batch_id]
                    for batch in session.batches:
                        if batch['id'] == batch_id:
                            batch.update(deleted=True, count=0, summary_video_url=None)
                    session.analysis_report.update(observation_count=len(session.summary), latest_summary_video_url=None)
            for job in getattr(app.state, 'upload_jobs', {}).values():
                if job.get('run') == root.name:
                    job.pop('result', None)
                    job.update(status='deleted', error='삭제된 분석입니다.')
        return {'deleted': len(set(targets)), 'removed_files': removed_files}

    @app.get('/api/history')
    async def history(request: Request):
        require_owner(request)
        rows = []
        for path in output_root.glob('*/history/batch_*.json'):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if data['batch']['id'] in deleted_batches(path.parent.parent):
                    continue
                rows.append({'run': path.parent.parent.name, 'file': path.name, **data['batch']})
            except (OSError, ValueError, KeyError):
                continue
        known = {(row['run'], row['id']) for row in rows}
        for path in output_root.glob('*/analysis.json'):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                for batch in data.get('batches', []):
                    if batch['id'] in deleted_batches(path.parent):
                        continue
                    if (path.parent.name, batch['id']) not in known:
                        rows.append({'run': path.parent.name, 'file': f"legacy_{batch['id']}.json", **batch})
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, key=lambda row: row.get('completed_at', ''), reverse=True)

    @app.get('/api/history/{run_name}/{filename}')
    async def history_detail(run_name: str, filename: str, request: Request):
        require_owner(request)
        root = archive_path(run_name)
        number = filename.removeprefix('batch_').removeprefix('legacy_').removesuffix('.json')
        if number.isdigit() and int(number) in deleted_batches(root):
            raise HTTPException(404, '삭제된 분석입니다.')
        path = (root / 'history' / filename).resolve()
        if filename.startswith('legacy_') and filename.endswith('.json') and filename[7:-5].isdigit():
            source = root / 'analysis.json'
            if not source.is_file():
                raise HTTPException(404, 'Archive not found')
            saved = json.loads(source.read_text(encoding='utf-8'))
            batch = next((row for row in saved.get('batches', []) if row['id'] == int(filename[7:-5])), None)
            if batch is None:
                raise HTTPException(404, 'Archive not found')
            data = {'batch': batch, 'observations': [row for row in saved.get('observations', []) if row.get('batch_id') == batch['id']],
                    'report': {'ai_summary': '이전 형식의 기록으로 회차별 AI 요약은 저장되어 있지 않습니다.'}}
        elif path.parent != root / 'history' or path.suffix != '.json' or not path.is_file():
            raise HTTPException(404, 'Archive not found')
        else:
            data = json.loads(path.read_text(encoding='utf-8'))
        from livecc_jobs import get_job
        for record in data.get('observations', []):
            details = record.get('caption_details') or {}
            job = get_job(details['job_id']) if details.get('job_id') else None
            if not job:
                continue
            result = json.loads(job['result']) if job['result'] else details
            status = result.get('status') if job['state'] == 'done' else job['state']
            record.update(caption_details=result, caption_status=status,
                caption_error=None if status == 'ok' else status,
                caption=result.get('caption') or '설명 상태: ' + status)
            if result.get('caption_clip_url'):
                record['action_clip_url'] = record.get('action_clip_url') or record.get('clip_url')
                record['clip_url'] = result['caption_clip_url']
            elif result.get('clip_export_error'):
                record['action_clip_url'] = record.get('action_clip_url') or record.get('clip_url')
                record['clip_url'] = None
        observations = data.get('observations', [])
        if (path.is_file() and observations and not data.get('report', {}).get('caption_summary_finalized')
                and any((row.get('caption_details') or {}).get('job_id') for row in observations)
                and all(row.get('caption_status') not in ('pending', 'waiting', 'running') for row in observations)):
            from realtime_analysis import OpenAIAnalyzer, _atomic_json
            batch = data['batch']
            summary, error = await asyncio.to_thread(OpenAIAnalyzer(ROOT).summarize, observations, batch['end'] - batch['start'])
            data['report'].update(ai_summary=summary, ai_error=error, caption_summary_finalized=True)
            _atomic_json(path, data)
        def rewrite(value):
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, str) and value.startswith('/api/sessions/') and '/media/' in value:
                return f'/api/history-media/{run_name}/' + value.split('/media/', 1)[1]
            return value
        return rewrite(data)

    @app.get('/api/history-media/{run_name}/{relative_path:path}')
    async def history_media(run_name: str, relative_path: str, request: Request):
        require_owner(request)
        root = archive_path(run_name)
        path = (root / relative_path).resolve()
        types = {'.webm': 'video/webm', '.mp4': 'video/mp4', '.jpg': 'image/jpeg', '.png': 'image/png'}
        if root not in path.parents or path.suffix not in types or not path.is_file():
            raise HTTPException(404, 'Media not found')
        return FileResponse(path, media_type=types[path.suffix])

    @app.get("/api/sessions/{session_id}/frames/{filename}")
    async def image(session_id: str, filename: str, request: Request):
        require_owner(request)
        session = get_session(session_id)
        allowed = {Path(record["selected_frame_copy"]).name for record in session.summary}
        if filename not in allowed:
            raise HTTPException(404, "키프레임이 없습니다.")
        return FileResponse(session.output_dir / "selected_frames" / filename, media_type="image/jpeg")

    @app.get("/api/sessions/{session_id}/media/{relative_path:path}")
    async def analysis_media(session_id: str, relative_path: str, request: Request):
        require_owner(request)
        session = get_session(session_id)
        root = session.output_dir.resolve()
        path = (root / relative_path).resolve()
        allowed_suffixes = {".mp4", ".webm", ".jpg", ".jpeg", ".png"}
        if root not in path.parents or path.suffix.lower() not in allowed_suffixes or not path.is_file():
            raise HTTPException(404, "분석 미디어를 찾을 수 없습니다.")
        media_types = {".mp4": "video/mp4", ".webm": "video/webm", ".png": "image/png"}
        media_type = media_types.get(path.suffix.lower(), "image/jpeg")
        return FileResponse(path, media_type=media_type)

    @app.get("/api/sessions/{session_id}/report")
    async def download_report(session_id: str, request: Request):
        require_owner(request)
        session = get_session(session_id)
        path = session.output_dir / "summary.json"
        if not path.exists():
            raise HTTPException(404, "아직 완료된 분석 결과가 없습니다.")
        return FileResponse(path, media_type="application/json", filename=session.run_name + ".json")

    @app.websocket("/ws/viewer/{session_id}")
    async def viewer(socket: WebSocket, session_id: str):
        try:
            require_owner(socket)
            session = get_session(session_id)
        except HTTPException:
            await socket.close(code=1008)
            return
        if session.viewers >= 4:
            await socket.close(code=1013)
            return
        await socket.accept()
        session.viewers += 1

        async def publish():
            previous_version = -1
            last_snapshot = 0.0
            while True:
                now = time.monotonic()
                if now - last_snapshot >= .5:
                    await asyncio.wait_for(socket.send_json(session.snapshot()), timeout=4)
                    last_snapshot = now
                if session.frame_version != previous_version and session.latest_jpeg:
                    await asyncio.wait_for(socket.send_bytes(session.latest_jpeg), timeout=4)
                    previous_version = session.frame_version
                await asyncio.sleep(.04)

        async def observe_disconnect():
            while True:
                await socket.receive_text()

        publisher = asyncio.create_task(publish())
        receiver = asyncio.create_task(observe_disconnect())
        try:
            completed, _ = await asyncio.wait((publisher, receiver), return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                with suppress(asyncio.CancelledError):
                    task.result()
        except (WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
            pass
        finally:
            publisher.cancel()
            receiver.cancel()
            await asyncio.gather(publisher, receiver, return_exceptions=True)
            session.viewers -= 1
            with suppress(Exception):
                await socket.close()

    @app.websocket("/ws/phone/{session_id}")
    async def phone(socket: WebSocket, session_id: str):
        if not same_origin(socket):
            await socket.close(code=1008)
            return
        await socket.accept()
        session = sessions.get(session_id)
        try:
            credentials = await asyncio.wait_for(socket.receive_json(), timeout=8)
            if not session or session.stopped or time.monotonic() > session.expires_mono:
                await socket.close(code=1008, reason="QR 코드가 만료되었거나 연동이 종료되었습니다.")
                return
            if not isinstance(credentials, dict) or not isinstance(credentials.get("token"), str) or not secrets.compare_digest(credentials["token"], session.phone_token):
                await socket.close(code=1008, reason="유효하지 않은 연결 코드입니다.")
                return
            if session.phone is not None:
                await socket.close(code=1008, reason="이 코드에 이미 다른 카메라가 연결되어 있습니다.")
                return
            session.phone = socket
            await socket.send_json({"type": "ready", "interval_seconds": session.interval})
            while not session.stopped:
                payload = await asyncio.wait_for(socket.receive_bytes(), timeout=12)
                await session.receive(payload)
                # 수신 확인 후 다음 프레임을 보내므로 느린 네트워크에도 프레임이 쌓이지 않는다.
                await socket.send_json({"type": "ack", "interval_seconds": session.interval})
        except ValueError as error:
            with suppress(Exception):
                await socket.close(code=1003, reason=str(error))
        except (WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
            with suppress(Exception):
                await socket.close(code=1011, reason="전송이 중단되었습니다. 다시 연결해 주세요.")
        finally:
            if session and session.phone is socket:
                session.phone = None
                session.previous_gray = None
                session.last_frame_mono = None

    assets = ROOT / "realtime_frontend" / "dist"
    if (assets / "assets").exists():
        app.mount("/assets", StaticFiles(directory=assets / "assets"), name="assets")

    @app.get("/")
    @app.get("/phone")
    async def frontend():
        if not (assets / "index.html").exists():
            raise HTTPException(503, "React 화면을 먼저 빌드해 주세요: npm.cmd --prefix realtime_frontend run build")
        return FileResponse(assets / "index.html", media_type="text/html")

    return app
