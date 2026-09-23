"""Upload through HTTP, run skimming, and serve results without AI calls."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from realtime_server import create_app


class UploadTest(unittest.TestCase):
    def test_upload_analysis_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'fixture.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 4, (160, 120))
            self.assertTrue(writer.isOpened())
            for index in range(80):
                frame = np.full((120, 160, 3), 70, np.uint8)
                cv2.rectangle(frame, (index % 100, 30), (index % 100 + 40, 80), (230, 180, 100), -1)
                writer.write(frame)
            writer.release()
            app = create_app(admin_token='upload-test-token-for-verification', output_root=root / 'outputs')
            with patch('realtime_analysis.extract_descriptors', side_effect=lambda path, indices: np.tile([1., 0.], (len(indices), 1))), patch('realtime_analysis.OpenAIAnalyzer') as ai, TestClient(app, base_url='https://testserver') as client:
                ai.return_value.enabled = True
                ai.return_value.model = 'test-model'
                ai.return_value.describe.return_value = ('업로드 영상 설명', None)
                ai.return_value.summarize.return_value = ('업로드 영상 전체 요약', None)
                self.assertEqual(client.post('/api/upload-tests?filename=test.avi', content=b'x').status_code, 401)
                client.post('/api/login', json={'token': 'upload-test-token-for-verification'})
                self.assertEqual(client.post('/api/upload-tests?filename=test.txt', content=b'x').status_code, 415)
                self.assertEqual(client.post('/api/upload-tests?filename=test.avi', content=b'').status_code, 400)
                with patch('upload_test.MAX_UPLOAD_BYTES', 2):
                    self.assertEqual(client.post('/api/upload-tests?filename=test.avi', content=b'123').status_code, 413)
                response = client.post('/api/upload-tests?filename=test.avi', content=source.read_bytes())
                self.assertEqual(response.status_code, 202, response.text)
                job = response.json()
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    job = client.get('/api/upload-tests/' + job['id']).json()
                    if job['status'] != 'analyzing':
                        break
                    time.sleep(.1)
                self.assertEqual(job['status'], 'done', job)
                result = job['result']
                self.assertEqual(result['batch']['source_frames'], 80)
                self.assertGreater(len(result['observations']), 0)
                self.assertTrue(all(row['caption'] == '업로드 영상 설명' for row in result['observations']))
                self.assertEqual(result['report']['ai_summary'], '업로드 영상 전체 요약')
                ai.assert_called_once()
                self.assertTrue(ai.return_value.describe.called)
                ai.return_value.summarize.assert_called_once()
                self.assertEqual(client.get(result['observations'][0]['image_url']).status_code, 200)
                self.assertEqual(client.get(result['batch']['summary_video_url']).status_code, 200)
                self.assertEqual(client.get('/api/history/' + job['run'] + '/batch_0001.json').status_code, 200)


if __name__ == '__main__':
    unittest.main()
