from pathlib import Path
import json
import shutil


def write_json(path: str | Path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_selected_frames(summary: list[dict], output_dir: str):
    dst_dir = Path(output_dir) / "selected_frames"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in summary:
        src = Path(item["representative_frame_path"])
        if src.exists():
            dst = dst_dir / f"segment_{item['segment_id']:06d}_{src.name}"
            shutil.copy2(src, dst)
            item["selected_frame_copy"] = str(dst)
