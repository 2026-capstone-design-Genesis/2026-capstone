"""Authenticated file-upload tests using the realtime skimmer, without captions."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import secrets

import cv2
import numpy as np
from fastapi import HTTPException, Request

from realtime_analysis import RecordedSegment, analyze_recorded_segment, save_analysis_files

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def register_upload_tests(app, require_owner, output_root, project_root):
    jobs = {}
    app.state.upload_jobs = jobs
    tasks = app.state.upload_tasks

    def process(path, folder, identifier):
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not capture.isOpened() or not np.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError('영상을 읽을 수 없습니다. 정상적인 MP4 또는 WebM 파일을 선택해 주세요.')
        finally:
            capture.release()
        segment = RecordedSegment(path, 0., count / fps, count, fps)
        batch, records, report = analyze_recorded_segment(
            root=project_root, output_dir=folder, session_id=identifier,
            segment=segment, batch_id=1, prior_observations=[], interval=0,
            reason='upload_test', captions_enabled=True,
        )
        save_analysis_files(folder, records, [batch], report, datetime.now().isoformat())
        metadata_path = folder / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(source='uploaded_video', captions_enabled=True, method='hietaskim_upload_test')
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

        def rewrite(value):
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, str) and value.startswith(f'/api/sessions/{identifier}/media/'):
                return f'/api/history-media/{folder.name}/' + value.split('/media/', 1)[1]
            return value

        return rewrite({'batch': batch, 'observations': records, 'report': report})

    async def run(identifier, path, folder):
        try:
            jobs[identifier]['result'] = await asyncio.to_thread(process, path, folder, identifier)
            jobs[identifier]['status'] = 'done'
        except Exception as exc:
            jobs[identifier].update(status='error', error=f'{type(exc).__name__}: {exc}')

    @app.post('/api/upload-tests', status_code=202)
    async def upload(request: Request, filename: str = 'video.mp4'):
        require_owner(request)
        if any(job['status'] in ('uploading', 'analyzing') for job in jobs.values()):
            raise HTTPException(409, '현재 업로드 영상의 분석이 끝난 뒤 다시 시도해 주세요.')
        if len(jobs) >= 32:
            raise HTTPException(429, '테스트 실행 한도에 도달했습니다. 서버를 재시작해 주세요.')
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(415, 'MP4, MOV, AVI, MKV, WebM 파일을 선택해 주세요.')
        identifier = secrets.token_hex(12)
        folder = output_root / f'upload_test_{datetime.now():%Y%m%d_%H%M%S}_{identifier}'
        folder.mkdir(parents=True)
        path = folder / f'source{suffix}'
        jobs[identifier] = {'id': identifier, 'filename': filename[:200], 'status': 'uploading', 'run': folder.name}
        try:
            size = 0
            with path.open('wb') as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, '영상은 최대 1GB까지 업로드할 수 있습니다.')
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(400, '빈 파일은 분석할 수 없습니다.')
        except BaseException:
            path.unlink(missing_ok=True)
            jobs.pop(identifier, None)
            raise
        jobs[identifier]['status'] = 'analyzing'
        task = asyncio.create_task(run(identifier, path, folder))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return dict(jobs[identifier])

    @app.get('/api/upload-tests/{identifier}')
    async def status(identifier: str, request: Request):
        require_owner(request)
        if identifier not in jobs:
            raise HTTPException(404, '테스트 결과가 없거나 서버가 재시작되었습니다.')
        return jobs[identifier]
