"""Synthetic scene boundaries and timestamp-aware skim budgets."""

import unittest

import numpy as np

from modules.hietaskim import _adaptive_labels, component_keyshot_regions, hierarchical_regions, sample_indices, select_keyshots


class HierarchicalSkimming(unittest.TestCase):
    def test_empty_single_and_constant(self):
        self.assertEqual(hierarchical_regions([], []), [])
        self.assertEqual(hierarchical_regions([0], [[1]]), [(0, 0)])
        self.assertEqual(hierarchical_regions(np.arange(20) / 4, np.ones((20, 2)), min_components=1), [(0, 19)])

    def test_abrupt_scenes_split(self):
        features = [[1., 0.]] * 8 + [[0., 1.]] * 8 + [[1., 1.]] * 8
        self.assertEqual(hierarchical_regions(np.arange(24) / 4, features), [(0, 7), (8, 15), (16, 23)])

    def test_identical_frames_across_time_gap_stay_separate(self):
        self.assertEqual(hierarchical_regions([0, .25, 10, 10.25], [[1]] * 4, min_components=1), [(0, 1), (2, 3)])

    def test_interleaved_component_is_split_into_contiguous_runs(self):
        self.assertEqual(hierarchical_regions([0, .25, .5], [[1, 0], [0, 1], [1, 0]]), [(0, 0), (1, 1), (2, 2)])

    def test_gamma_controls_cutting(self):
        features = [[1., 0.]] * 8 + [[0., 1.]] * 8
        self.assertEqual(hierarchical_regions(np.arange(16) / 4, features, gamma=100, min_components=1), [(0, 15)])

    def test_minimum_components_override_stability(self):
        regions = hierarchical_regions(np.arange(16) / 4, [[1., 0.]] * 16, gamma=100)
        self.assertGreaterEqual(len(regions), 3)

    def test_cosine_is_invariant_to_descriptor_magnitude(self):
        features = np.array([[1., 0.]] * 8 + [[0., 1.]] * 8)
        self.assertEqual(hierarchical_regions(np.arange(16) / 4, features),
                         hierarchical_regions(np.arange(16) / 4, features * np.arange(1, 17)[:, None]))

    def test_priority_is_saliency_but_threshold_uses_original_distances(self):
        labels = _adaptive_labels(4, np.array([0, 1, 2]), np.array([1, 2, 3]),
                                  np.array([.1, .9, .1]), np.array([10., 9., 8.]), .75, 1)
        self.assertEqual(len(set(labels)), 1)

    def test_threshold_is_recomputed_in_each_component(self):
        labels = _adaptive_labels(6, np.arange(5), np.arange(1, 6),
                                  np.array([.1, .8, .2, .9, .1]), np.array([0., 2., 0., 3., 0.]), .75, 1)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertEqual(labels[4], labels[5])
        self.assertEqual(len(set(labels)), 3)

    def test_sampling_uses_capture_time(self):
        np.testing.assert_array_equal(sample_indices([0, .2, .5, .7, 1., 4.]), [0, 2, 4, 5])

    def test_noncontiguous_component_produces_one_centered_keyshot(self):
        labels = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        regions = component_keyshot_regions(labels)
        self.assertEqual(regions, [(4, 5, 5), (6, 7, 6)])
        shots = select_keyshots(np.arange(10) / 4, regions, 4, .8)
        self.assertEqual(len(shots), 2)
        for left, right, center in shots:
            self.assertTrue(left <= center <= right)
            self.assertEqual(len(set(labels[left:right + 1])), 1)

    def test_all_four_paper_watersheds_partition_every_frame(self):
        features = np.random.default_rng(4).uniform(.1, 1, (24, 8))
        for hierarchy in ('area', 'volume', 'dynamics', 'number_of_parents'):
            with self.subTest(hierarchy=hierarchy):
                regions = hierarchical_regions(np.arange(24) / 2, features, hierarchy=hierarchy)
                self.assertEqual([i for start, end in regions for i in range(start, end + 1)], list(range(24)))

    def test_midpoint_and_total_budget(self):
        times = np.arange(120) / 4
        shots = select_keyshots(times, [(0, 39), (40, 79), (80, 119)], 4)
        self.assertEqual(len(shots), 3)
        self.assertLessEqual(sum(right - left + 1 for left, right, _ in shots), 18)
        for (left, right, center), (start, end) in zip(shots, [(0, 39), (40, 79), (80, 119)]):
            self.assertTrue(start <= left <= center <= right <= end)
            self.assertLessEqual(abs(times[center] - (times[start] + times[end]) / 2), .125)

    def test_irregular_timestamps_budget(self):
        times = [0, .1, .3, 1, 4, 4.2, 7, 10]
        shots = select_keyshots(times, [(0, 3), (4, 7)], 4, .3)
        playback_frames = sum(round((times[right] - times[left]) * 4 + 1) for left, right, _ in shots)
        self.assertLessEqual(playback_frames, int(41 * .3))

    def test_tiny_budget_and_many_shots(self):
        self.assertEqual(select_keyshots([0], [(0, 0)], 4), [])
        shots = select_keyshots(np.arange(20) / 4, [(i, i) for i in range(20)], 4)
        self.assertEqual(shots, [])

    def test_paper_budget_is_not_capped_at_six_seconds(self):
        times = np.arange(1200) / 4
        shots = select_keyshots(times, [(0, 399), (400, 799), (800, 1199)], 4)
        self.assertEqual([right - left + 1 for left, right, _ in shots], [60, 60, 60])

    def test_invalid_parameters(self):
        for kwargs in ({'delta_t': 0}, {'gamma': -1}, {'gamma': float('nan')}):
            with self.assertRaises(ValueError):
                hierarchical_regions([0], [[1]], **kwargs)
        with self.assertRaises(ValueError):
            select_keyshots([0], [(0, 0)], 4, summary_ratio=1.1)


if __name__ == '__main__':
    unittest.main()
