import unittest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch
from livecc_caption import select_indices, valid_caption


class SamplingTests(unittest.TestCase):
    def test_invalid_keyframe(self):
        self.assertEqual(select_indices([0, .25, .5], 10)[0]['status'], 'invalid_input')

    def test_cross_file_mapping_and_future_wait(self):
        from realtime_analysis import SegmentRecorder
        from tests.test_realtime_analysis import jpeg_frame
        from livecc_caption import decode_session
        from livecc_jobs import export_caption_clip
        import av
        with tempfile.TemporaryDirectory() as directory:
            recorder = SegmentRecorder(Path(directory))
            for index in range(8):
                recorder.write(jpeg_frame(index), index / 4)
            first = recorder.rotate(2)
            result, _ = decode_session([first.path], 1.5, ended=False)
            self.assertEqual(result['status'], 'waiting')
            for index in range(8):
                recorder.write(jpeg_frame(index), 2 + index / 4)
            second = recorder.rotate(4)
            result, frames = decode_session([first.path, second.path], 1.5, ended=False)
            self.assertEqual(result['status'], 'sampled')
            self.assertEqual(len(frames), 4)
            self.assertEqual(len(set(item['file_id'] for item in result['frame_mapping'])), 2)
            clip = Path(directory) / 'caption.webm'
            export_caption_clip([first.path, second.path], clip, .5, 2.5)
            with av.open(str(clip)) as container:
                decoded = list(container.decode(video=0))
                self.assertEqual(len(decoded), 9)
                self.assertAlmostEqual(float(decoded[-1].pts * decoded[-1].time_base), 2)

    def test_durable_enqueue_idempotency(self):
        import livecc_jobs
        with tempfile.TemporaryDirectory() as directory, patch.object(livecc_jobs, 'STATE', Path(directory)):
            args = ('session', 1, 'frame', directory, 1., {'count': 4})
            first = livecc_jobs.enqueue(*args)
            self.assertEqual(first, livecc_jobs.enqueue(*args))
            self.assertEqual(livecc_jobs.get_job(first)['state'], 'pending')

    def test_worker_heartbeat(self):
        import livecc_jobs
        with tempfile.TemporaryDirectory() as directory, patch.object(livecc_jobs, 'STATE', Path(directory)):
            self.assertFalse(livecc_jobs.get_worker_status()['online'])
            livecc_jobs.write_worker_status('idle', model_loaded=False)
            self.assertTrue(livecc_jobs.get_worker_status()['online'])
            self.assertEqual(livecc_jobs.get_worker_status()['state'], 'idle')
    def test_keyframe_and_order(self):
        times = np.arange(0, 5, .25)
        result, indices = select_indices(times, 2.25)
        self.assertIn(9, indices)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(len(indices), 4)
        self.assertEqual(result['frame_times_sec'], times[indices].tolist())

    def test_boundaries_without_extension(self):
        result, _ = select_indices([0, .25, .5], 0)
        self.assertEqual(result['clip_start_sec'], 0)
        self.assertEqual(result['clip_end_sec'], .5)
        self.assertEqual(result['observed_frame_count'], 3)

    def test_past_only(self):
        result, _ = select_indices(np.arange(0, 5, .25), 2.1, mode='past_only', count=8)
        self.assertTrue(all(t <= 2.1 for t in result['frame_times_sec']))

    def test_waiting_is_not_ended(self):
        result, indices = select_indices([0, .5, 1], 1, ended=False)
        self.assertEqual(result['status'], 'waiting')
        self.assertEqual(indices, [])
        result, _ = select_indices([0, .5, 1], 1, ended=True)
        self.assertEqual(result['status'], 'sampled')

    def test_empty_and_format(self):
        self.assertEqual(select_indices([], 0)[0]['status'], 'no_frames')
        self.assertTrue(valid_caption('사람이 걷고 있다.'))
        self.assertFalse(valid_caption('사람이 걷는다. 멈춘다.'))
        self.assertFalse(valid_caption('A person is walking.'))
