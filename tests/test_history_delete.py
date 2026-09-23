import json
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from realtime_server import create_app


class HistoryDelete(unittest.TestCase):
    def test_bulk_delete_persists_and_keeps_unselected_and_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'saved'
            (run / 'history').mkdir(parents=True)
            batches = [{'id': i, 'start': 0, 'end': 30, 'count': 1} for i in (1, 2, 3)]
            (run / 'history/batch_0001.json').write_text(json.dumps({'batch': batches[0]}))
            (run / 'analysis.json').write_text(json.dumps({'batches': batches}))
            (run / 'original.mp4').write_bytes(b'keep original')
            clips = run / 'analysis/clips'
            clips.mkdir(parents=True)
            (clips / 'b0001_01.webm').write_bytes(b'delete')
            (clips / 'b0002_01.webm').write_bytes(b'delete')
            (clips / 'b0003_01.webm').write_bytes(b'keep')
            frames = run / 'analysis/keyframes'
            frames.mkdir()
            (frames / 'b0001_01.jpg').write_bytes(b'delete')
            summaries = run / 'analysis/summary_videos'
            summaries.mkdir()
            (summaries / 'summary_0001.webm').write_bytes(b'delete')
            token = 'test-history-delete-controller'
            with TestClient(create_app(admin_token=token, output_root=root)) as client:
                payload = {'items': [{'run': 'saved', 'file': 'batch_0001.json'}, {'run': 'saved', 'file': 'legacy_2.json'}]}
                self.assertEqual(client.post('/api/history/delete', json=payload).status_code, 401)
                self.assertEqual(client.post('/api/login', json={'token': token}).status_code, 200)
                invalid = {'items': [payload['items'][0], {'run': 'saved', 'file': '../analysis.json'}]}
                self.assertEqual(client.post('/api/history/delete', json=invalid).status_code, 400)
                self.assertEqual(len(client.get('/api/history').json()), 3)
                result = client.post('/api/history/delete', json=payload).json()
                self.assertEqual(result['deleted'], 2)
                self.assertEqual(result['removed_files'], 4)
                self.assertFalse((clips / 'b0001_01.webm').exists())
                self.assertFalse((clips / 'b0002_01.webm').exists())
                self.assertFalse((frames / 'b0001_01.jpg').exists())
                self.assertFalse((summaries / 'summary_0001.webm').exists())
                self.assertTrue((clips / 'b0003_01.webm').exists())
                self.assertEqual([row['id'] for row in client.get('/api/history').json()], [3])
                self.assertEqual(client.get('/api/history/saved/batch_0001.json').status_code, 404)
                self.assertEqual(client.get('/api/history/saved/legacy_1.json').status_code, 404)
                self.assertEqual(client.post('/api/history/delete', json=payload).status_code, 200)
            with TestClient(create_app(admin_token=token, output_root=root)) as client:
                client.post('/api/login', json={'token': token})
                self.assertEqual([row['id'] for row in client.get('/api/history').json()], [3])
            self.assertEqual((run / 'original.mp4').read_bytes(), b'keep original')

    def test_shared_and_outside_files_are_protected(self):
        from history_deletion import delete_batch_files
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            (root / 'history').mkdir(parents=True)
            (root / 'analysis/clips').mkdir(parents=True)
            shared = root / 'analysis/clips/shared.webm'
            shared.write_bytes(b'keep')
            outside = Path(directory) / 'outside.webm'
            outside.write_bytes(b'keep')
            for batch_id in (1, 2):
                (root / 'history' / f'batch_{batch_id:04d}.json').write_text(json.dumps({
                    'batch': {'id': batch_id}, 'observations': [{'clip_url': str(shared), 'image_url': str(outside)}]}))
            self.assertEqual(delete_batch_files(root, 1), 0)
            self.assertTrue(shared.exists())
            self.assertTrue(outside.exists())
