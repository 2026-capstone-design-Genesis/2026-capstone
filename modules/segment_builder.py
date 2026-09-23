def build_segments(frames: list[dict], segment_sec: float, representative_frame: str = "middle") -> list[dict]:
    if not frames:
        return []

    segments_map: dict[int, list[dict]] = {}
    for frame in frames:
        seg_id = int(frame["timestamp"] // segment_sec)
        segments_map.setdefault(seg_id, []).append(frame)

    segments = []
    for seg_id, seg_frames in sorted(segments_map.items()):
        if representative_frame == "first":
            rep = seg_frames[0]
        else:
            rep = seg_frames[len(seg_frames) // 2]

        segments.append({
            "segment_id": seg_id,
            "start_time": float(seg_id * segment_sec),
            "end_time": float((seg_id + 1) * segment_sec),
            "frame_ids": [f["frame_id"] for f in seg_frames],
            "representative_frame_id": rep["frame_id"],
            "representative_frame_path": rep["frame_path"],
            "caption": None,
        })

    return segments
