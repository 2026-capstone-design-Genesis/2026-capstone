"""Integration checks for recorded realtime video analysis."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from realtime_analysis import SegmentRecorder, analyze_recorded_segment, save_analysis_files
from realtime_analysis import compensated_motion, context_sample_seconds, exclude_camera_regions, frame_quality, motion_regions
from realtime_analysis import refine_action_bounds
from realtime_analysis import deduplicate_keyshots, temporal_keyshot_regions


def jpeg_frame(index: int) -> bytes:
    canvas = np.full((180, 320, 3), 30, dtype=np.uint8)
    left = min(260, 12 + index * 18)
    canvas[60:125, left:left + 42] = (230, 190, 70)
    content = io.BytesIO()
    Image.fromarray(canvas).save(content, "JPEG", quality=90)
    return content.getvalue()


class RecordedAnalysis(unittest.TestCase):
    def test_duplicate_clips_keep_better_quality_and_temporal_order(self):
        shots = [(0, 2, 1), (3, 5, 4), (6, 8, 7)]
        features = np.array([[0.]] * 3 + [[1.]] * 3 + [[0.]] * 3)
        qualities = [{'usable': True, 'score': 1.} for _ in range(9)]
        qualities[7]['score'] = 2.
        self.assertEqual(deduplicate_keyshots(shots, features, qualities), [shots[1], shots[2]])

    def test_same_middle_with_different_clip_ending_is_retained(self):
        shots = [(0, 2, 1), (3, 5, 4)]
        features = np.array([[0.], [0.], [0.], [0.], [0.], [1.]])
        qualities = [{'usable': True, 'score': 1.} for _ in range(6)]
        self.assertEqual(deduplicate_keyshots(shots, features, qualities), shots)

    def test_duplicate_clips_prefer_usable_frame_over_rejected(self):
        shots = [(0, 0, 0), (1, 1, 1)]
        qualities = [{'usable': False, 'score': 100.}, {'usable': True, 'score': 1.}]
        self.assertEqual(deduplicate_keyshots(shots, np.zeros((2, 4)), qualities), [shots[1]])
        self.assertEqual(deduplicate_keyshots([], [], []), [])

    def test_temporal_keyshots_keep_return_to_prior_view(self):
        # The same visual component at the beginning and end must not erase
        # the return trip from a chronological CCTV summary.
        self.assertEqual(
            temporal_keyshot_regions(np.asarray([0, 0, 1, 1, 0, 0])),
            [(0, 1, 0), (2, 3, 2), (4, 5, 4)],
        )

    def setUp(self):
        # Graph/clip integration is offline; the actual pretrained descriptor
        # is verified separately to avoid model downloads in the unit suite.
        descriptor_patch = patch('realtime_analysis.extract_descriptors',
                                 side_effect=lambda path, indices: np.tile([1., 0.], (len(indices), 1)))
        descriptor_patch.start()
        self.addCleanup(descriptor_patch.stop)

    def test_context_samples_cover_before_transition_and_after(self):
        samples = context_sample_seconds(7.0, 10.0, 13.0, 12)
        self.assertEqual(len(samples), 12)
        self.assertEqual(samples, sorted(samples))
        self.assertLess(samples[0], 9.0)
        self.assertTrue(any(9.5 <= value <= 10.5 for value in samples))
        self.assertGreater(samples[-1], 11.0)

    def test_action_bounds_are_limited_around_transition(self):
        times = np.arange(0, 30, .25)
        motions = np.full(len(times), .4)
        motions[60] = 8.0
        left, right, peak = refine_action_bounds(times, motions, 0, len(times) - 1)
        self.assertEqual(peak, 60)
        self.assertLessEqual(times[right] - times[left], 6.0)
        self.assertLessEqual(times[left], times[peak])
        self.assertGreaterEqual(times[right], times[peak])

    def test_camera_translation_is_suppressed(self):
        previous = np.zeros((180, 320), dtype=np.uint8)
        for y in range(15, 170, 25):
            for x in range(15, 310, 25):
                cv2.circle(previous, (x, y), 3, 255, -1)
        matrix = np.float32([[1, 0, 8], [0, 1, 4]])
        current = cv2.warpAffine(previous, matrix, (320, 180), borderMode=cv2.BORDER_REPLICATE)
        result = compensated_motion(previous, current)
        self.assertEqual(result['alignment'], 'aligned')
        self.assertLess(result['motion_percent'], result['raw_motion_percent'])
        self.assertTrue(result['camera_motion'])

    def test_widespread_scene_change_is_not_treated_as_subject_motion(self):
        previous = np.tile(np.arange(320, dtype=np.uint8), (180, 1))
        current = 255 - previous
        result = compensated_motion(previous, current)
        self.assertTrue(result['camera_motion'])
        self.assertEqual(result['motion_percent'], 0.0)

    def test_local_subject_change_survives_global_filter(self):
        previous = np.full((180, 320), 60, dtype=np.uint8)
        current = previous.copy()
        current[60:120, 120:190] = 230
        result = compensated_motion(previous, current)
        self.assertFalse(result['camera_motion'])
        self.assertGreater(result['motion_percent'], 0.65)

    def test_camera_dominated_region_is_excluded(self):
        camera = {'camera_motion': True}
        subject = {'camera_motion': False}
        regions = exclude_camera_regions([(0, 3), (4, 7)], [camera] * 3 + [subject] * 5)
        self.assertEqual(regions, [(4, 7)])

    def test_frame_quality_rejects_blank_exposure_and_accepts_detail(self):
        self.assertFalse(frame_quality(np.zeros((180, 320, 3), dtype=np.uint8))["usable"])
        self.assertFalse(frame_quality(np.full((180, 320, 3), 255, dtype=np.uint8))["usable"])
        detailed = np.full((180, 320, 3), 80, dtype=np.uint8)
        detailed[30:150:8, :] = 220
        self.assertTrue(frame_quality(detailed)["usable"])

    def test_continuous_motion_is_one_region(self):
        times = np.arange(0, 40, .25)
        self.assertEqual(motion_regions(times, [2.] * len(times), [np.zeros(4)] * len(times)), [(0, len(times) - 1)])

    def test_separate_and_brief_changes_survive(self):
        times = np.arange(0, 12, .25)
        motions = np.zeros(len(times))
        motions[4] = 2
        motions[32:36] = 2
        self.assertEqual(motion_regions(times, motions, [np.zeros(4)] * len(times)), [(4, 4), (32, 35)])

    def test_nearby_matching_regions_merge(self):
        times = np.arange(0, 8, .25)
        motions = np.zeros(len(times))
        motions[4:8] = 2
        motions[14:18] = 2
        self.assertEqual(motion_regions(times, motions, [np.zeros(4)] * len(times)), [(4, 17)])

    def test_record_summary_keyframes_and_shorts(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            recorder = SegmentRecorder(output_dir)
            for index in range(48):
                recorder.write(jpeg_frame(index), index / 4)
            segment = recorder.rotate(12.0)
            self.assertIsNotNone(segment)
            self.assertTrue(segment.path.exists())

            batch, records, report = analyze_recorded_segment(
                root=Path(directory),
                output_dir=output_dir,
                session_id="test-session",
                segment=segment,
                batch_id=1,
                prior_observations=[],
                interval=30,
                reason="test",
            )
            self.assertGreater(len(records), 0)
            self.assertTrue(Path(records[0]["selected_frame_copy"]).exists())
            self.assertTrue(records[0]["clip_url"].endswith(".webm"))
            self.assertNotIn("event_type", records[0])
            self.assertEqual(records[0]["quality_status"], "usable")
            self.assertIn("sharpness", records[0]["frame_quality"])
            self.assertTrue(batch["recording_url"].endswith(".webm"))
            self.assertTrue(batch["summary_video_url"].endswith(".webm"))
            self.assertEqual(records[0]['analysis_method'], 'hietaskim_vlm')
            self.assertEqual(batch['skimming']['summary_ratio'], .15)
            self.assertEqual(batch['skimming']['clips_before_dedup'] - batch['skimming']['duplicates_removed'],
                             batch['skimming']['clips_after_dedup'])
            self.assertEqual(batch['skimming']['clips_after_dedup'], len(records))
            skim = cv2.VideoCapture(str(output_dir / 'analysis/summary_videos/summary_0001.webm'))
            skim_frames = int(skim.get(cv2.CAP_PROP_FRAME_COUNT))
            skim.release()
            self.assertGreater(skim_frames, 0)
            self.assertLessEqual(skim_frames, int(48 * .15))
            clip_frames = 0
            for row in records:
                clip = cv2.VideoCapture(str(output_dir / row['clip_url'].split('/media/', 1)[1]))
                clip_frames += int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
                clip.release()
            self.assertEqual(skim_frames, clip_frames)
            self.assertEqual(report["status"], "ready")
            self.assertIn("quality_rejected_frames", batch)

            save_analysis_files(output_dir, records, [batch], report, "test-created-at")
            for filename in ("summary.json", "batches.json", "analysis.json", "metadata.json"):
                self.assertTrue((output_dir / filename).exists())

    def test_rejected_frames_skip_caption_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            recorder = SegmentRecorder(output_dir)
            blank = io.BytesIO()
            Image.fromarray(np.zeros((180, 320, 3), dtype=np.uint8)).save(blank, "JPEG")
            for index in range(8):
                recorder.write(blank.getvalue(), index / 4)
            segment = recorder.rotate(2.0)
            _, records, _ = analyze_recorded_segment(
                root=Path(directory), output_dir=output_dir, session_id="test-session",
                segment=segment, batch_id=1, prior_observations=[], interval=30, reason="test",
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["caption_status"], "quality_rejected")
            self.assertEqual(records[0]["quality_reasons"], ["too_dark", "low_information"])
            self.assertIsNone(records[0]["caption_model"])


if __name__ == "__main__":
    unittest.main()
