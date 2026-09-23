"""Durable shared queue. Only the locked worker owns a GPU model."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from fractions import Fraction
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent
STATE = ROOT / '.livecc'


@contextmanager
def connect():
    STATE.mkdir(exist_ok=True)
    db = sqlite3.connect(STATE / 'jobs.sqlite3', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT, state TEXT, result TEXT, created REAL, updated REAL)')
    try:
        with db:
            yield db
    finally:
        db.close()


def enqueue(session_id, batch_id, record_id, output_dir, keyframe, settings):
    payload = dict(session_id=session_id, batch_id=batch_id, record_id=record_id,
        output_dir=str(Path(output_dir).resolve()), keyframe=keyframe, settings=settings)
    identifier = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with connect() as db:
        db.execute('INSERT OR IGNORE INTO jobs VALUES (?, ?, ?, NULL, ?, ?)',
            (identifier, json.dumps(payload), 'pending', time.time(), time.time()))
    return identifier


def get_job(identifier):
    with connect() as db:
        row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
        return dict(row) if row else None


def write_worker_status(state, **details):
    STATE.mkdir(exist_ok=True)
    content = {'state': state, 'pid': os.getpid(), 'heartbeat': time.time(), **details}
    destination = STATE / 'worker-status.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(destination)


def get_worker_status(max_age=10):
    path = STATE / 'worker-status.json'
    if not path.is_file():
        return {'online': False, 'state': 'not_started'}
    try:
        content = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {'online': False, 'state': 'unreadable'}
    content['online'] = content.get('state') != 'stopped' and time.time() - content.get('heartbeat', 0) <= max_age
    return content


def export_caption_clip(paths, destination, start, end):
    import av
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.webm')
    stream = None
    last_pts = -1
    with av.open(str(temporary), 'w') as output:
        for path in paths:
            capture_times = json.loads(path.with_suffix('.timestamps.json').read_text(encoding='utf-8'))
            with av.open(str(path)) as source:
                for index, frame in enumerate(source.decode(video=0)):
                    second = capture_times[index]
                    if second < start or second > end:
                        continue
                    if stream is None:
                        stream = output.add_stream('libvpx', rate=4)
                        stream.width, stream.height = frame.width, frame.height
                        stream.pix_fmt = 'yuv420p'
                        stream.time_base = Fraction(1, 1000)
                    pts = round((second - start) * 1000)
                    if pts <= last_pts:
                        continue
                    last_pts = pts
                    frame = frame.reformat(width=stream.width, height=stream.height, format='yuv420p')
                    frame.pts, frame.time_base = pts, Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        output.mux(packet)
        if stream is None:
            raise ValueError('No frames for caption clip')
        for packet in stream.encode():
            output.mux(packet)
    temporary.replace(destination)


def worker(model_path):
    import msvcrt
    from livecc_caption import LiveCCCaptioner
    STATE.mkdir(exist_ok=True)
    # The OS releases this lock after crashes; never delete the lock file.
    with (STATE / 'worker.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit('A LiveCC worker already owns this queue/GPU')
        model = LiveCCCaptioner(model_path, profile='nf4')
        write_worker_status('idle', model_loaded=False)
        with connect() as db:
            db.execute("UPDATE jobs SET state='pending' WHERE state='running'")
        print('LiveCC worker ready', flush=True)
        heartbeat = 0
        try:
            while True:
                if time.time() - heartbeat >= 2:
                    write_worker_status('idle', model_loaded=model.model is not None)
                    heartbeat = time.time()
                with connect() as db:
                    row = db.execute("SELECT * FROM jobs WHERE state IN ('pending','waiting') ORDER BY updated LIMIT 1").fetchone()
                if not row:
                    time.sleep(.5)
                    continue
                payload = json.loads(row['payload'])
                folder = Path(payload['output_dir'])
                paths = [p for p in sorted((folder / 'recordings').glob('*.webm')) if p.with_suffix('.timestamps.json').exists()]
                ended = (folder / 'session-ended.json').exists()
                signature = [ended, [[p.name, p.stat().st_size] for p in paths]]
                prior = json.loads(row['result']) if row['result'] else {}
                if row['state'] == 'waiting' and prior.get('available_files') == signature:
                    with connect() as db:
                        db.execute("UPDATE jobs SET state='waiting', updated=? WHERE id=?", (time.time(), row['id']))
                    time.sleep(.5)
                    continue
                with connect() as db:
                    db.execute("UPDATE jobs SET state='running', updated=? WHERE id=?", (time.time(), row['id']))
                write_worker_status('running', model_loaded=model.model is not None, job_id=row['id'])
                inference_started = time.time()
                try:
                    result = model.caption(paths, payload['keyframe'], ended=ended, **payload['settings'])
                    state = 'waiting' if result['status'] == 'waiting' else 'done'
                    if 'clip_start_sec' in result and result['status'] != 'waiting':
                        relative = Path('analysis/clips') / (row['id'] + '.webm')
                        try:
                            export_caption_clip(paths, folder / relative, result['clip_start_sec'], result['clip_end_sec'])
                            result['caption_clip_url'] = f"/api/sessions/{payload['session_id']}/media/{relative.as_posix()}"
                        except Exception as error:
                            result['clip_export_error'] = str(error)
                except Exception as error:
                    result, state = {'status': 'worker_error', 'caption': '', 'error': str(error)}, 'done'
                result['job_id'] = row['id']
                result['session_id'] = payload['session_id']
                result['batch_id'] = payload['batch_id']
                result['available_files'] = signature
                result['observation_wait_started'] = prior.get('observation_wait_started', inference_started if state == 'waiting' else None)
                wait_started = result['observation_wait_started']
                result['observation_wait_seconds'] = max(0, inference_started - wait_started) if wait_started else 0
                result['queue_seconds'] = max(0, inference_started - row['created'] - result['observation_wait_seconds'])
                with connect() as db:
                    db.execute('UPDATE jobs SET state=?, result=?, updated=? WHERE id=?',
                        (state, json.dumps(result, ensure_ascii=False), time.time(), row['id']))
                if state == 'waiting':
                    time.sleep(.5)
        finally:
            write_worker_status('stopped', model_loaded=model.model is not None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    worker(args.model)
