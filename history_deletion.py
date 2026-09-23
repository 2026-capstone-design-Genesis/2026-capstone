"""Delete generated batch media, keeping recordings and shared files."""
import json
from pathlib import Path
from realtime_analysis import _atomic_json


def delete_batch_files(root, batch_id):
    root = Path(root).resolve()
    history = root / 'history' / f'batch_{batch_id:04d}.json'
    aggregate = root / 'analysis.json'
    saved = json.loads(aggregate.read_text(encoding='utf-8')) if aggregate.is_file() else {}
    allowed = [root / 'analysis' / part for part in ('clips', 'keyframes', 'summary_videos', 'caption_clips')]
    allowed.append(root / 'selected_frames')

    def collect(value):
        paths = set()
        if isinstance(value, dict):
            for item in value.values():
                paths.update(collect(item))
        elif isinstance(value, list):
            for item in value:
                paths.update(collect(item))
        elif isinstance(value, str):
            candidate = None
            if value.startswith('/api/sessions/') and '/media/' in value:
                candidate = root / value.split('/media/', 1)[1]
            elif value.startswith(f'/api/history-media/{root.name}/'):
                candidate = root / value.split(f'/api/history-media/{root.name}/', 1)[1]
            elif Path(value).is_absolute():
                candidate = Path(value)
            if candidate is not None:
                candidate = candidate.resolve()
                if root in candidate.parents and any(folder in candidate.parents for folder in allowed):
                    paths.add(candidate)
        return paths

    selected = {'observations': [row for row in saved.get('observations', []) if row.get('batch_id') == batch_id],
                'batches': [row for row in saved.get('batches', []) if row.get('id') == batch_id]}
    paths = collect(selected)
    if history.is_file():
        detail = json.loads(history.read_text(encoding='utf-8'))
        paths.update(collect(detail))
        selected['observations'].extend(detail.get('observations', []))
    for observation in selected['observations']:
        job_id = (observation.get('caption_details') or {}).get('job_id')
        if job_id:
            from livecc_jobs import get_job
            job = get_job(job_id)
            if job and job['state'] in ('pending', 'waiting', 'running'):
                raise RuntimeError('캡션 생성 중인 분석입니다. 완료 후 삭제해 주세요.')
            if job and job.get('result'):
                paths.update(collect(json.loads(job['result'])))
    for folder in allowed:
        paths.update(folder.glob(f'b{batch_id:04d}_*'))
    paths.update((root / 'analysis/summary_videos').glob(f'summary_{batch_id:04d}.*'))
    protected = collect({'observations': [row for row in saved.get('observations', []) if row.get('batch_id') != batch_id],
                         'batches': [row for row in saved.get('batches', []) if row.get('id') != batch_id]})
    for other in (root / 'history').glob('batch_*.json'):
        if other != history:
            protected.update(collect(json.loads(other.read_text(encoding='utf-8'))))
    removed = 0
    for path in paths - protected:
        resolved = path.resolve()
        if root in resolved.parents and any(folder in resolved.parents for folder in allowed) and resolved.is_file():
            resolved.unlink()
            removed += 1
    if history.is_file() and history.resolve().parent == root / 'history':
        history.unlink()
    for filename in ('analysis.json', 'summary.json', 'batches.json'):
        path = root / filename
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        if filename == 'summary.json':
            value = [row for row in value if row.get('batch_id') != batch_id]
        elif filename == 'batches.json':
            value = [row for row in value if row.get('id') != batch_id]
        else:
            value['observations'] = [row for row in value.get('observations', []) if row.get('batch_id') != batch_id]
            value['batches'] = [row for row in value.get('batches', []) if row.get('id') != batch_id]
            value['observation_count'] = len(value['observations'])
            value['latest_summary_video_url'] = (value['batches'][-1] if value['batches'] else {}).get('summary_video_url')
        _atomic_json(path, value)
    return removed
