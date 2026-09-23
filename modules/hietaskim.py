"""HieTaSkim (Cardoso et al., SIBGRAPI 2024, DOI 10.1109/SIBGRAPI62404.2024.10716326).

Method/equations: author's thesis sections 4.1, 4.3.1, 5.3.2:
https://bib.pucminas.br/teses/Informatica_LeonardoVilelaCardoso_32039_TextoCompleto.pdf
ResNet descriptor layer/preprocessing: author's HieTaSumm-lib/HieTaSumm/Models.py.
"""

from functools import lru_cache
import heapq
import os
from pathlib import Path

import cv2
import higra as hg
import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _descriptor_model():
    # Keep ImageNet weights in the project's runtime cache.
    os.environ.setdefault('KERAS_HOME', str(Path(__file__).resolve().parents[1] / '.local-realtime' / 'hietaskim'))
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    # The author's ResNet50 descriptor is the 1000-D predictions output.
    return ResNet50(weights='imagenet', include_top=True), preprocess_input


def extract_descriptors(video_path, frame_indices, batch_size=16):
    """ImageNet ResNet50 descriptors; no handcrafted-feature fallback."""
    if not len(frame_indices):
        return np.empty((0, 1000), dtype=np.float32)
    model, preprocess = _descriptor_model()
    capture = cv2.VideoCapture(str(video_path))
    descriptors, batch = [], []
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open video for HieTaSkim descriptors')
        wanted = iter(frame_indices)
        target = next(wanted, None)
        index = 0
        while target is not None:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot decode HieTaSkim frame {target}')
            if index == target:
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch.append(np.asarray(rgb.resize((224, 224), Image.Resampling.NEAREST), dtype=np.float32))
                target = next(wanted, None)
                if len(batch) == batch_size or target is None:
                    descriptors.extend(np.asarray(model(preprocess(np.stack(batch)), training=False)))
                    batch.clear()
            index += 1
    finally:
        capture.release()
    return np.asarray(descriptors, dtype=np.float32)


def sample_indices(times, sample_fps=2.0):
    """Sample actual timestamps at the paper's 2 fps analysis rate."""
    indices = []
    next_time = float(times[0]) if len(times) else 0.0
    for index, timestamp in enumerate(times):
        if timestamp + 1e-9 >= next_time:
            indices.append(index)
            next_time = timestamp + 1 / sample_fps
    return np.asarray(indices, dtype=int)


def _adaptive_labels(n, sources, targets, weights, saliency, gamma, min_components):
    """Order cuts by Phi_H; compare original w(e) with local mean + gamma*std.

    Zero saliency is the watershed base partition, only refined for NC_min.
    Equal-priority edges are processed individually, in stable edge order.
    """
    adjacency = [[] for _ in range(n)]
    for edge, (left, right) in enumerate(zip(sources, targets)):
        adjacency[left].append((right, edge))
        adjacency[right].append((left, edge))
    removed = set()

    def connected(start):
        nodes, edges, stack = {start}, set(), [start]
        while stack:
            node = stack.pop()
            for neighbor, edge in adjacency[node]:
                if edge in removed:
                    continue
                edges.add(edge)
                if neighbor not in nodes:
                    nodes.add(neighbor)
                    stack.append(neighbor)
        return nodes, edges

    pending, finished, seen = [], [], set()

    def push(nodes, edges):
        if not edges:
            finished.append(nodes)
        else:
            edge = min(edges, key=lambda i: (-saliency[i], i))
            heapq.heappush(pending, (-saliency[edge], edge, nodes, edges))

    count = 0
    for node in range(n):
        if node not in seen:
            nodes, edges = connected(node)
            seen.update(nodes)
            push(nodes, edges)
            count += 1
    while pending:
        _, edge, nodes, edges = heapq.heappop(pending)
        local = weights[list(edges)]
        threshold = float(local.mean() + gamma * local.std())
        forced = count < min(n, min_components)
        if not forced and (saliency[edge] <= 0 or weights[edge] < threshold):
            finished.append(nodes)
            continue
        removed.add(edge)
        count += 1
        for start in (sources[edge], targets[edge]):
            child_nodes, child_edges = connected(start)
            push(child_nodes, child_edges)
    labels = np.empty(n, dtype=int)
    for label, nodes in enumerate(finished):
        labels[list(nodes)] = label
    return labels


def hierarchical_regions(times, features, delta_t=4.0, gamma=0.75, min_components=3, hierarchy='area', return_labels=False):
    """Cosine graph -> Kruskal MST -> watershed saliency -> adaptive cuts.

    delta_t is seconds (4 seconds equals delta_t=8 at the paper's 2 fps).
    Disconnected recordings stay a forest. Return labels for component-based
    keyshot selection, or contiguous runs for inspecting the partition.
    """
    times = np.asarray(times, dtype=float)
    features = np.asarray(features, dtype=float)
    methods = {'area': hg.watershed_hierarchy_by_area, 'dynamics': hg.watershed_hierarchy_by_dynamics,
               'volume': hg.watershed_hierarchy_by_volume, 'number_of_parents': hg.watershed_hierarchy_by_number_of_parents}
    if not np.isfinite(delta_t) or delta_t <= 0 or not np.isfinite(gamma) or gamma < 0:
        raise ValueError('delta_t must be positive and gamma must be nonnegative')
    if hierarchy not in methods or min_components < 1 or int(min_components) != min_components:
        raise ValueError('invalid watershed hierarchy or minimum component count')
    if len(times) != len(features):
        raise ValueError('times and features must have equal lengths')
    if not len(times):
        return np.empty(0, dtype=int) if return_labels else []
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0) or not np.all(np.isfinite(features)):
        raise ValueError('finite features and chronological timestamps are required')
    features = features.reshape(len(times), -1)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('cosine distance requires nonzero descriptors')
    features = features / norms
    all_sources, all_targets, all_weights, all_saliency = [], [], [], []
    starts = [0] + (np.flatnonzero(np.diff(times) > delta_t) + 1).tolist()
    for start, stop in zip(starts, starts[1:] + [len(times)]):
        if stop - start < 2:
            continue
        graph = hg.UndirectedGraph(stop - start)
        weights = []
        for left in range(start, stop - 1):
            end = min(stop, int(np.searchsorted(times, times[left] + delta_t, side='right')))
            right = np.arange(left + 1, end)
            graph.add_edges(np.full(len(right), left - start, dtype=int), right - start)
            weights.extend(np.clip(1 - features[right] @ features[left], 0, 2))
        weights = np.asarray(weights)
        # Kruskal with deterministic ties: nearest in time, then earliest edge.
        # Ranking preserves the primary weight order without epsilon distortion.
        sources, targets = graph.edge_list()
        order = np.lexsort((sources, times[targets + start] - times[sources + start], weights))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(len(order))
        mst = hg.minimum_spanning_tree(graph, ranks)
        mst_weights = weights[hg.CptMinimumSpanningTree.get_edge_map(mst)]
        tree, altitudes = methods[hierarchy](mst, mst_weights)
        saliency = hg.saliency(tree, altitudes, mst)
        sources, targets = mst.edge_list()
        all_sources.extend(sources + start)
        all_targets.extend(targets + start)
        all_weights.extend(mst_weights)
        all_saliency.extend(saliency)
    labels = _adaptive_labels(len(times), np.array(all_sources, dtype=int), np.array(all_targets, dtype=int),
                              np.array(all_weights), np.array(all_saliency), gamma, min_components)
    if return_labels:
        return labels
    starts = [0] + (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()
    return [(start, stop - 1) for start, stop in zip(starts, starts[1:] + [len(times)])]


def component_keyshot_regions(labels):
    """One keyshot per component, around its temporally central member.

    A component can be noncontiguous. Keep its central member's continuous run
    instead of inflating NC by counting all of that component's runs as shots.
    """
    labels = np.asarray(labels)
    regions = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        center = int(members[len(members) // 2])
        left = right = center
        while left > 0 and labels[left - 1] == label:
            left -= 1
        while right + 1 < len(labels) and labels[right + 1] == label:
            right += 1
        regions.append((left, right, center))
    return sorted(regions)


def select_keyshots(times, regions, fps, summary_ratio=0.15):
    """Central windows with S = floor(N*p/NC), without an arbitrary 6s cap.

    N is the original decoded frame count. A playback-time budget additionally
    keeps timestamp-expanded clips within p for irregular realtime capture.
    If S is zero, no shot fits; graph cuts are never replaced by uniform picks.
    """
    times = np.asarray(times, dtype=float)
    if not np.isfinite(summary_ratio) or not 0 < summary_ratio <= 1:
        raise ValueError('summary_ratio must be in (0, 1]')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive')
    if not len(times) or not regions:
        return []
    allowance = int(np.floor(len(times) * summary_ratio / len(regions) + 1e-9))
    playback_allowance = int(np.floor(((times[-1] - times[0]) * fps + 1) * summary_ratio / len(regions) + 1e-9))
    if min(allowance, playback_allowance) < 1:
        return []
    shots = []
    for region in regions:
        start, end = region[:2]
        midpoint = (times[start] + times[end]) / 2
        center = region[2] if len(region) == 3 else min(range(start, end + 1), key=lambda index: abs(times[index] - midpoint))
        if len(region) == 3:
            midpoint = times[center]
        left = right = center
        while right - left + 1 < allowance:
            options = []
            for a, b in ((left - 1, right), (left, right + 1)):
                if a >= start and b <= end and round((times[b] - times[a]) * fps + 1) <= playback_allowance:
                    options.append((abs((times[a] + times[b]) / 2 - midpoint), a, b))
            if not options:
                break
            _, left, right = min(options)
        shots.append((left, right, center))
    return shots
