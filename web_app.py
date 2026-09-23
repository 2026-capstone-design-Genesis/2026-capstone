from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import html
import io
import json
import os
import subprocess
import sys
import time
import zipfile

import streamlit as st
import yaml


def _inside_streamlit() -> bool:
    """이 파일이 Streamlit으로 실행되고 있는지 확인한다."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _inside_streamlit():
    command = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
               "--server.address=0.0.0.0", "--server.port=8501"]
    print("[RUN] " + " ".join(command))
    print("[INFO] 종료하려면 이 창에서 Ctrl+C를 누르세요.")
    raise SystemExit(subprocess.call(command, cwd=str(Path(__file__).resolve().parent)))


from modules.hf_cache import configure_hf_cache, default_hf_cache_dir


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "web_uploads"
OUTPUT_ROOT = ROOT / "outputs"
WATCH_DIR = ROOT / "watch_videos"
SETTINGS_FILE = ROOT / "app_settings.json"
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

MODE_KMEANS = "K-Means 클러스터링"
MODE_LLAVA = "LLaVA / Llama 분석"
MODE_CHECKPOINT = "LLMVS 체크포인트 분석"
MODE_CLUFRAME = "CluFrame + LLMVS 분석"
MODE_FAST = "빠른 변화량 분석"
LLM_MODES = {MODE_LLAVA, MODE_CHECKPOINT, MODE_CLUFRAME}
NAV_VIDEO, NAV_KEYFRAME, NAV_REPORT = "영상 선택", "키프레임 조회", "분석 요약서"

st.set_page_config(page_title="CluLLM 영상 분석 관제", page_icon="🎥", layout="wide", initial_sidebar_state="expanded")


# ---------- 기존 추출 파이프라인 및 결과 조회 기능 ----------

def ensure_dirs() -> None:
    for directory in (UPLOAD_DIR, OUTPUT_ROOT, WATCH_DIR):
        directory.mkdir(exist_ok=True)


def load_base_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_app_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}
    except Exception:
        return {}


def save_app_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_dir() -> str:
    return load_app_settings().get("hf_cache_dir") or os.environ.get("HF_HOME") or default_hf_cache_dir()


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)


def make_run_name(video_path: Path) -> str:
    return f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def zip_dir(source_dir: Path) -> Path:
    """기존 전체 ZIP 다운로드 기능을 유지한다."""
    zip_path = source_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))
    return zip_path


def rows_to_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO()
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
    return output.getvalue().encode("utf-8-sig")


def build_caption_rows(output_dir: Path) -> list[dict]:
    segments = read_json(output_dir / "segments.json") or []
    captions = read_json(output_dir / "captions.json") or []
    captions_by_id = {item.get("segment_id"): item for item in captions if isinstance(item, dict)}
    rows = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        caption = captions_by_id.get(segment.get("segment_id"), {})
        rows.append({
            "segment_id": segment.get("segment_id"),
            "start_time": segment.get("start_time"),
            "end_time": segment.get("end_time"),
            "caption": segment.get("caption") or caption.get("caption", ""),
            "representative_frame_path": segment.get("representative_frame_path") or caption.get("representative_frame_path", ""),
        })
    return rows or [item for item in captions if isinstance(item, dict)]


def build_llama_rows(output_dir: Path) -> list[dict]:
    segments = read_json(output_dir / "segments.json") or []
    scores = read_json(output_dir / "scores.json") or read_json(output_dir / "scores_raw.json") or []
    raw_scores = read_json(output_dir / "scores_raw.json") or []
    summary = read_json(output_dir / "summary.json") or []
    segments_by_id = {item.get("segment_id"): item for item in segments if isinstance(item, dict)}
    raw_by_id = {item.get("segment_id"): item for item in raw_scores if isinstance(item, dict)}
    selected = {item.get("segment_id") for item in summary if isinstance(item, dict)}
    rows = []
    for score in scores:
        if not isinstance(score, dict):
            continue
        segment_id = score.get("segment_id")
        segment, raw = segments_by_id.get(segment_id, {}), raw_by_id.get(segment_id, {})
        rows.append({
            "selected": "yes" if segment_id in selected else "",
            "segment_id": segment_id,
            "start_time": segment.get("start_time"),
            "end_time": segment.get("end_time"),
            "raw_score_0_10": score.get("raw_score_0_10", raw.get("raw_score_0_10", "")),
            "local_score": score.get("local_score", raw.get("local_score", "")),
            "final_score": score.get("final_score", raw.get("final_score", score.get("local_score", ""))),
            "caption": segment.get("caption", ""),
            "llm_answer": score.get("llm_answer", raw.get("llm_answer", "")),
            "query": score.get("query", raw.get("query", "")),
        })
    return rows


def show_model_reports(output_dir: Path) -> None:
    """기존 Caption/Llama 표와 원본 파일 다운로드를 제공한다."""
    caption_rows, llama_rows = build_caption_rows(output_dir), build_llama_rows(output_dir)
    if not caption_rows and not llama_rows:
        st.info("이 결과에는 Caption 또는 Llama 상세 데이터가 없습니다.")
        return
    caption_tab, llama_tab, raw_tab = st.tabs(["LLaVA 캡션", "Llama 중요도", "원본 파일"])
    with caption_tab:
        if caption_rows:
            st.dataframe(caption_rows, use_container_width=True, hide_index=True)
            st.download_button("캡션 CSV 다운로드", rows_to_csv_bytes(caption_rows), "captions_report.csv", "text/csv", key=f"caption_{output_dir.name}")
        else:
            st.info("캡션 결과가 없습니다.")
    with llama_tab:
        if llama_rows:
            st.dataframe([{key: value for key, value in row.items() if key != "query"} for row in llama_rows], use_container_width=True, hide_index=True)
            with st.expander("세그먼트별 Llama 프롬프트/응답 확인"):
                for row in llama_rows:
                    st.markdown(f"**세그먼트 {row.get('segment_id')}** · 최종 점수 {row.get('final_score', '-')}")
                    st.code(row.get("query") or "저장된 프롬프트가 없습니다.")
                    if row.get("llm_answer"):
                        st.caption("Llama 응답")
                        st.code(row["llm_answer"])
            st.download_button("Llama 점수 CSV 다운로드", rows_to_csv_bytes(llama_rows), "llama_scores_report.csv", "text/csv", key=f"llama_{output_dir.name}")
        else:
            st.info("Llama 점수 결과가 없습니다.")
    with raw_tab:
        for column, filename in zip(st.columns(3), ["captions.json", "scores_raw.json", "scores.json"]):
            file_path = output_dir / filename
            with column:
                if file_path.exists():
                    st.download_button(filename, file_path.read_bytes(), filename, "application/json", key=f"{filename}_{output_dir.name}")


def build_llmvs_config(video_path: Path, run_name: str, ui: dict) -> dict:
    config = load_base_config()
    config["input"]["video_path"] = str(video_path)
    config["project"]["run_name"] = run_name
    config["project"]["output_dir"] = str(OUTPUT_ROOT)
    config["sampling"]["fps"] = ui["sample_fps"]
    config["segment"]["segment_sec"] = ui["segment_sec"]
    config["selection"].update({
        "mode": ui["selection_mode"],
        "top_k": ui["top_k"],
        "summary_ratio": ui["summary_ratio"],
        "threshold": ui["threshold"],
    })
    config["caption"].update({"model_id": ui["llava_model_id"], "cache_dir": ui["hf_cache_dir"]})
    config["scoring"].update({
        "model_id": ui["llama_model_id"], "cache_dir": ui["hf_cache_dir"], "use_quantization_4bit": ui["use_4bit"],
        "window_size": ui["window_size"], "checkpoint_path": ui["checkpoint_path"],
    })
    config["global_context"].update({"mode": ui["global_mode"], "smoothing_window": ui["smoothing_window"]})
    config.setdefault("preprocessing", {})
    config["preprocessing"].update({
        "enabled": ui["cluframe_enabled"], "sample_fps": ui["sample_fps"],
        "max_candidates": ui["cluframe_max_candidates"], "feature_backend": ui["cluframe_feature_backend"],
        "resize_width": ui["cluframe_resize_width"], "min_total_motion": ui["cluframe_min_total_motion"],
    })
    return config


def build_preload_config(ui: dict, mode: str) -> dict:
    config = load_base_config()
    config.setdefault("caption", {})
    config.setdefault("scoring", {})
    config["caption"].update({"model_id": ui["llava_model_id"], "cache_dir": ui["hf_cache_dir"], "backend": "llava", "enabled": True})
    config["scoring"].update({"model_id": ui["llama_model_id"], "cache_dir": ui["hf_cache_dir"], "use_quantization_4bit": ui["use_4bit"], "backend": "llmvs_checkpoint" if mode == MODE_CHECKPOINT else "llama2"})
    return config


def progress_logger(container):
    logs: list[str] = []
    def callback(message: str, payload: dict | None = None):
        logs.append(message if payload is None else f"{message} {payload}")
        container.code("\n".join(logs[-80:]))
    return callback


def process_video(video_path: Path, mode: str, ui: dict, hf_token: str | None) -> Path:
    """기존의 5개 영상 처리 방식을 그대로 실행한다."""
    output_dir = OUTPUT_ROOT / make_run_name(video_path)
    callback = progress_logger(st.empty())
    configure_hf_cache(ui["hf_cache_dir"])
    if mode == MODE_KMEANS:
        from modules.kmeans_keyframe import extract_keyframes_kmeans
        extract_keyframes_kmeans(
            video_path=str(video_path),
            output_dir=str(output_dir),
            sample_fps=ui["sample_fps"],
            top_k=ui["top_k"],
            min_gap_sec=ui["min_gap_sec"],
            progress_callback=callback,
        )
    elif mode == MODE_FAST:
        from modules.simple_keyframe import extract_keyframes_fast
        extract_keyframes_fast(
            video_path=str(video_path),
            output_dir=str(output_dir),
            sample_fps=ui["sample_fps"],
            top_k=ui["top_k"],
            min_gap_sec=ui["min_gap_sec"],
            progress_callback=callback,
        )
    else:
        from pipeline import run_llmvs_pipeline
        config = build_llmvs_config(video_path, output_dir.name, ui)
        config["scoring"]["backend"] = "llmvs_checkpoint" if mode == MODE_CHECKPOINT else "llama2"
        if mode == MODE_CLUFRAME:
            config["preprocessing"]["enabled"] = True
        run_llmvs_pipeline(config, hf_token=hf_token, progress_callback=callback, model_manager=get_model_manager())
    return output_dir


def find_watch_videos() -> list[Path]:
    return sorted(path for path in WATCH_DIR.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS)


def is_processed(video: Path) -> bool:
    return (OUTPUT_ROOT / ".processed" / f"{video.name}.done").exists()


def mark_processed(video: Path, output_dir: Path) -> None:
    marker_dir = OUTPUT_ROOT / ".processed"
    marker_dir.mkdir(exist_ok=True)
    (marker_dir / f"{video.name}.done").write_text(str(output_dir), encoding="utf-8")


@st.cache_resource(show_spinner=False)
def get_cached_model_manager():
    from modules.model_manager import ModelManager
    return ModelManager()


def get_model_manager():
    return get_cached_model_manager()


def preloaded_key(ui: dict) -> tuple:
    return (ui["llava_model_id"], ui["llama_model_id"], ui["hf_cache_dir"], ui["use_4bit"])


def models_ready(ui: dict, mode: str) -> bool:
    if st.session_state.get("models_preloaded_key") == preloaded_key(ui):
        return True
    try:
        manager = get_model_manager()
        return getattr(manager, "preloaded_key", None) == preloaded_key(ui) or manager.is_ready_for_config(build_preload_config(ui, mode))
    except Exception:
        return False


# ---------- 관제 대시보드 데이터 ----------

def list_runs() -> list[Path]:
    return sorted((path for path in OUTPUT_ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")), key=lambda path: path.stat().st_mtime, reverse=True)


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def timestamp(seconds: float) -> str:
    total = max(0, int(round(safe_float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def existing_path(value) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    path = path if path.is_absolute() else ROOT / path
    return path if path.exists() else None


def load_dashboard(run_path: Path | None) -> dict:
    if not run_path or not run_path.exists():
        return {"run": None, "metadata": {}, "items": [], "duration": 0.0}
    metadata = read_json(run_path / "metadata.json") or {}
    source = read_json(run_path / "summary.json") or read_json(run_path / "segments.json") or []
    items = []
    for index, raw in enumerate(source):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["display_id"] = int(item.get("segment_id", index))
        item["start_time"] = safe_float(item.get("start_time", item.get("timestamp", 0)))
        item["end_time"] = max(item["start_time"], safe_float(item.get("end_time", item["start_time"])))
        item["timestamp"] = safe_float(item.get("timestamp", (item["start_time"] + item["end_time"]) / 2))
        item["score"] = safe_float(item.get("final_score", item.get("local_score", item.get("raw_score", 0))))
        items.append(item)
    duration = safe_float(metadata.get("duration_sec")) or max((item["end_time"] for item in items), default=0.0)
    return {"run": run_path, "metadata": metadata, "items": sorted(items, key=lambda item: item["start_time"]), "duration": duration}


def status_for(item: dict) -> tuple[str, str, str]:
    """실제 상태값이 없으면 중요도를 위험이라고 단정하지 않는다."""
    raw = str(item.get("status") or item.get("event_status") or "").strip().lower()
    values = {
        "danger": ("위험", "danger", "모델이 저장한 위험 상태"), "critical": ("위험", "danger", "모델이 저장한 위험 상태"), "위험": ("위험", "danger", "모델이 저장한 위험 상태"),
        "warning": ("주의", "warning", "모델이 저장한 주의 상태"), "caution": ("주의", "warning", "모델이 저장한 주의 상태"), "주의": ("주의", "warning", "모델이 저장한 주의 상태"),
        "normal": ("문제없음", "normal", "모델이 저장한 정상 상태"), "safe": ("문제없음", "normal", "모델이 저장한 정상 상태"), "문제없음": ("문제없음", "normal", "모델이 저장한 정상 상태"),
    }
    if raw in values:
        return values[raw]
    if item.get("score", 0) >= .75:
        return "검토 우선 높음", "danger", "중요도 점수 기반 · 위험 판정 아님"
    if item.get("score", 0) >= .40:
        return "검토 우선 보통", "warning", "중요도 점수 기반 · 위험 판정 아님"
    return "검토 우선 낮음", "normal", "중요도 점수 기반 · 위험 판정 아님"


def frame_path(item: dict) -> Path | None:
    return existing_path(item.get("selected_frame_copy") or item.get("representative_frame_path"))


# ---------- 테마와 화면 구성 ----------

def apply_theme(dark: bool) -> None:
    colors = ({"bg":"#10131d", "panel":"#191e2b", "alt":"#131824", "border":"#30374a", "text":"#f2f5fb", "muted":"#aeb8cb", "blue":"#5aa8ff", "violet":"#a78bfa", "violetsoft":"rgba(167,139,250,.16)", "track":"#2b3343", "shadow":"rgba(0,0,0,.3)"}
              if dark else {"bg":"#f3f6fb", "panel":"#fff", "alt":"#f8faff", "border":"#dbe3f0", "text":"#172033", "muted":"#62708a", "blue":"#1976e9", "violet":"#7257e8", "violetsoft":"#f0edff", "track":"#e6edf7", "shadow":"rgba(33,53,85,.10)"})
    variables = "".join(f"--{key}:{value};" for key, value in colors.items())
    st.markdown(f"""
    <style>
    :root{{{variables}}}.stApp{{background:var(--bg);color:var(--text)}}
    .main .block-container{{max-width:1720px;padding-top:1.05rem;padding-bottom:2.5rem}}[data-testid="stHeader"]{{background:transparent}}
    [data-testid="stSidebar"]{{background:var(--alt);border-right:1px solid var(--border)}}[data-testid="stSidebar"] *{{color:var(--text)}}
    h1,h2,h3,p,label,.stCaption,.stMarkdown{{color:var(--text)}}[data-testid="stWidgetLabel"] p{{color:var(--muted);font-size:.82rem}}
    div[data-testid="stMetric"]{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:.7rem .85rem}}div[data-testid="stMetricLabel"]{{color:var(--muted)}}div[data-testid="stMetricValue"]{{color:var(--text);font-size:1.15rem}}
    div[data-testid="stExpander"]{{background:var(--panel);border:1px solid var(--border);border-radius:12px}}.stTextInput input,.stNumberInput input,.stSelectbox [data-baseweb="select"]>div{{background:var(--panel)!important;color:var(--text)!important;border-color:var(--border)!important}}
    .stButton>button,.stDownloadButton>button{{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:10px;min-height:2.3rem}}.stButton>button:hover,.stDownloadButton>button:hover{{border-color:var(--violet);color:var(--violet)}}.stButton>button[kind="primary"]{{background:var(--violet);color:white;border-color:var(--violet)}}
    [data-testid="stFileUploader"]{{background:var(--panel);border:1px dashed var(--border);border-radius:12px;padding:.25rem}}[data-testid="stTabs"] button[aria-selected="true"]{{color:var(--violet);border-bottom-color:var(--violet)}}
    .dashboard-header{{background:linear-gradient(120deg,var(--panel) 0%,var(--panel) 66%,var(--violetsoft) 100%);border:1px solid var(--border);border-radius:18px;padding:1.1rem 1.35rem;margin-bottom:1rem;box-shadow:0 8px 24px var(--shadow)}}.dashboard-header h1{{margin:0;font-size:1.55rem;letter-spacing:-.03em}}.dashboard-header p{{color:var(--muted);margin:.35rem 0 0;font-size:.91rem}}
    .sidebar-brand{{padding:.15rem .2rem .95rem}}.eyebrow,.section-kicker{{color:var(--violet);font-size:.69rem;font-weight:800;letter-spacing:.11em}}.sidebar-brand h2{{margin:.18rem 0 0;font-size:1.12rem}}.sidebar-brand p,.panel-subtitle{{color:var(--muted);font-size:.78rem;margin:.28rem 0 0}}.panel-heading{{font-size:1.02rem;font-weight:750;color:var(--text);margin:.14rem 0 .22rem}}
    .panel-shell,.timeline-shell{{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:1rem;box-shadow:0 5px 17px var(--shadow)}}.video-meta,.timeline-top{{display:flex;align-items:start;justify-content:space-between;gap:.75rem;margin-bottom:.8rem}}.video-name,.timeline-title{{font-size:1.02rem;font-weight:750;color:var(--text);overflow-wrap:anywhere}}.connection{{display:inline-flex;align-items:center;gap:.35rem;padding:.26rem .5rem;border-radius:99px;background:var(--violetsoft);color:var(--violet);font-size:.72rem;font-weight:700;white-space:nowrap}}.dot{{width:6px;height:6px;border-radius:50%;background:var(--blue)}}
    .video-placeholder{{min-height:315px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;background:#080b12;color:#e8eaf2;border:1px solid var(--border);border-radius:12px;padding:1.5rem}}.camera{{width:48px;height:36px;border:2px solid var(--violet);border-radius:8px;margin-bottom:.8rem;position:relative}}.camera:after{{content:'';position:absolute;right:-13px;top:8px;border-left:10px solid var(--violet);border-top:7px solid transparent;border-bottom:7px solid transparent}}.video-placeholder span{{color:#afb8c8;font-size:.82rem;margin-top:.35rem}}
    .selection-note{{background:var(--violetsoft);border:1px solid rgba(114,87,232,.22);border-radius:10px;color:var(--text);padding:.58rem .7rem;margin-top:.72rem;font-size:.8rem}}.keyframe-item{{border:1px solid var(--border);background:var(--panel);border-radius:12px;padding:.58rem;margin-bottom:.35rem}}.keyframe-item.selected{{border-color:var(--violet);box-shadow:0 0 0 2px var(--violetsoft)}}.caption,.time{{color:var(--muted);font-size:.72rem;line-height:1.35;margin:.24rem 0}}.badge{{display:inline-block;font-size:.67rem;font-weight:750;border-radius:99px;padding:.18rem .42rem}}.badge.danger{{background:rgba(239,68,68,.16);color:#ef4444}}.badge.warning{{background:rgba(245,158,11,.17);color:#d97706}}.badge.normal{{background:rgba(16,185,129,.16);color:#059669}}
    .timeline-shell{{margin-top:.85rem}}.timeline-value{{font-size:.76rem;color:var(--muted)}}.timeline{{position:relative;height:68px;margin:.7rem .35rem .1rem}}.timeline-track{{position:absolute;top:26px;left:0;right:0;height:8px;border-radius:999px;background:var(--track)}}.timeline-selected{{position:absolute;top:21px;height:18px;border-radius:999px;background:var(--violetsoft);border:1px solid var(--violet)}}.timeline-marker{{position:absolute;top:16px;width:10px;height:10px;border-radius:50%;transform:translateX(-50%);border:2px solid var(--panel);box-shadow:0 0 0 1px var(--border)}}.timeline-marker.danger{{background:#ef4444}}.timeline-marker.warning{{background:#f59e0b}}.timeline-marker.normal{{background:#10b981}}.timeline-marker.active{{top:12px;width:18px;height:18px;border:3px solid var(--panel);box-shadow:0 0 0 2px var(--violet);background:var(--violet)}}.timeline-tick{{position:absolute;top:43px;font-size:.66rem;color:var(--muted);transform:translateX(-50%);white-space:nowrap}}.legend{{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:.28rem;color:var(--muted);font-size:.7rem}}.legend span{{display:inline-flex;align-items:center;gap:.26rem}}.legend i{{display:inline-block;width:7px;height:7px;border-radius:50%}}
    .empty-keyframes{{padding:2.25rem .7rem;text-align:center;color:var(--muted);border:1px dashed var(--border);border-radius:12px}}.result-info{{background:var(--alt);border:1px solid var(--border);border-radius:12px;padding:.75rem .85rem;color:var(--muted);font-size:.82rem}}
    @media(max-width:900px){{.main .block-container{{padding-left:.8rem;padding-right:.8rem}}.video-placeholder{{min-height:220px}}}}
    </style>""", unsafe_allow_html=True)


def init_state() -> None:
    for key, value in {"theme_dark": False, "active_run": None, "active_video": None, "selected_keyframe": None, "uploader_reset": 0}.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_selection() -> None:
    st.session_state["uploader_reset"] += 1
    for key in ("active_run", "active_video", "selected_keyframe", "models_preloaded_key", "models_preloaded_status", "uploaded_signature"):
        st.session_state.pop(key, None)
    st.rerun()


def render_header() -> None:
    left, right = st.columns([5.8, 1.1], vertical_alignment="center")
    with left:
        st.markdown("""<div class="dashboard-header"><h1>CluLLM 영상 분석 관제</h1><p>업로드 영상의 키프레임을 빠르게 탐색하고, 시간대별 분석 결과를 한 화면에서 확인합니다.</p></div>""", unsafe_allow_html=True)
    with right:
        st.toggle("다크 모드", key="theme_dark", help="밝은 모드와 다크 모드를 전환합니다.")


def sidebar(runs: list[Path]) -> tuple[str, str, dict, str | None]:
    """필수 3개 내비게이션과 기존 실행 설정을 한 곳에 유지한다."""
    with st.sidebar:
        st.markdown("""<div class="sidebar-brand"><div class="eyebrow">VIDEO INTELLIGENCE</div><h2>CluLLM 관제실</h2><p>키프레임 추출 · 결과 탐색 · 분석 요약</p></div>""", unsafe_allow_html=True)
        navigation = st.radio("작업 메뉴", [NAV_VIDEO, NAV_KEYFRAME, NAV_REPORT], label_visibility="collapsed")
        st.divider()
        run_names = [run.name for run in runs]
        active_name = Path(st.session_state["active_run"]).name if st.session_state.get("active_run") else None
        options = ["새 영상 또는 결과 미선택"] + run_names
        choice = st.selectbox("표시할 분석 결과", options, index=options.index(active_name) if active_name in options else 0, help="저장된 분석 결과를 선택하면 키프레임과 원본 영상을 대시보드에 연결합니다.")
        if choice == options[0]:
            if st.session_state.get("active_run"):
                st.session_state["active_run"], st.session_state["selected_keyframe"] = None, None
                st.rerun()
        else:
            selected_run = next(run for run in runs if run.name == choice)
            if str(selected_run) != st.session_state.get("active_run"):
                st.session_state["active_run"], st.session_state["selected_keyframe"] = str(selected_run), None
                source = existing_path((read_json(selected_run / "metadata.json") or {}).get("video_path"))
                if source:
                    st.session_state["active_video"] = str(source)
                st.rerun()
        if st.button("화면 선택 초기화", use_container_width=True):
            reset_selection()
        st.divider()
        with st.expander("키프레임 추출 설정", expanded=navigation == NAV_VIDEO):
            mode = st.radio("추출 방식", [MODE_KMEANS, MODE_LLAVA, MODE_CHECKPOINT, MODE_CLUFRAME, MODE_FAST], help="기존 추출 파이프라인을 그대로 실행합니다.")
            sample_fps = st.number_input("샘플링 FPS", .1, 10., 1., .1)
            top_k = st.number_input("선택 키프레임 수", 1, 200, 10)
            min_gap = st.number_input("최소 키프레임 간격(초)", .0, 60., 2., .5)
            settings = load_app_settings()
            hf_cache = st.text_input("Hugging Face 캐시 위치", value=settings.get("hf_cache_dir") or cache_dir())
            c1, c2 = st.columns(2)
            with c1:
                if st.button("캐시 저장", use_container_width=True):
                    settings["hf_cache_dir"] = configure_hf_cache(hf_cache)
                    save_app_settings(settings)
                    st.success("저장했습니다.")
            with c2:
                if st.button("기본값", use_container_width=True):
                    settings["hf_cache_dir"] = configure_hf_cache(default_hf_cache_dir())
                    save_app_settings(settings)
                    st.rerun()
            with st.expander("고급 LLMVS 설정"):
                hf_token = st.text_input("Hugging Face 토큰", type="password", help="입력값은 저장하지 않습니다.")
                segment_sec = st.number_input("세그먼트 길이(초)", 1., 60., 5., 1.)
                window_size = st.number_input("Llama 점수 창 크기", 1, 31, 7, 2)
                selection_mode = st.selectbox("선택 방식", ["top_k", "top_ratio", "threshold"])
                summary_ratio = st.slider("요약 비율", .01, 1., .15, .01)
                threshold = st.slider("점수 임계값", .0, 1., .65, .01)
                global_mode = st.selectbox("전역 문맥 처리", ["temporal_smoothing", "none"])
                smoothing_window = st.number_input("스무딩 창 크기", 1, 15, 3, 2)
                use_4bit = st.checkbox("4-bit 양자화 사용", value=True)
                llava_id = st.text_input("LLaVA 모델 ID", "llava-hf/llava-1.5-7b-hf")
                llama_id = st.text_input("Llama 모델 ID", "meta-llama/Llama-2-13b-chat-hf")
                checkpoint = st.text_input("LLMVS 체크포인트 경로", "Summaries/summe_head2_layer3/summe/summe_split4/best_rho_model/epoch=31-val_sRho=0.343.ckpt")
                candidate_count = st.number_input("CluFrame 후보 수", 1, 300, max(12, int(top_k) * 3))
                feature_backend = st.selectbox("CluFrame 특징 추출기", ["lightweight", "efficientnet_b0"])
                resize_width = st.number_input("CluFrame 리사이즈 폭", 160, 1280, 320, 20)
                motion = st.number_input("CluFrame 움직임 임계값", 100., 20000., 1500., 100.)
            # 고급 설정은 접혀 있어도 각 위젯이 실행되므로 현재 값을 그대로 사용한다.
            ui = {"sample_fps": sample_fps, "top_k": int(top_k), "min_gap_sec": min_gap,
                  "segment_sec": segment_sec, "window_size": int(window_size), "selection_mode": selection_mode,
                  "summary_ratio": summary_ratio, "threshold": threshold, "global_mode": global_mode,
                  "smoothing_window": int(smoothing_window), "use_4bit": use_4bit, "llava_model_id": llava_id,
                  "llama_model_id": llama_id, "hf_cache_dir": configure_hf_cache(hf_cache), "checkpoint_path": checkpoint,
                  "cluframe_enabled": mode == MODE_CLUFRAME, "cluframe_max_candidates": int(candidate_count),
                  "cluframe_feature_backend": feature_backend, "cluframe_resize_width": int(resize_width),
                  "cluframe_min_total_motion": float(motion)}
        if mode in LLM_MODES:
            with st.expander("모델 사전 로딩"):
                if models_ready(ui, mode):
                    st.success(st.session_state.get("models_preloaded_status") or "현재 모델이 준비되어 있습니다.")
                else:
                    st.warning("분석 시작 전 현재 설정의 모델을 사전 로딩해야 합니다.")
                if st.button("현재 설정으로 모델 사전 로딩", type="primary", use_container_width=True):
                    status = st.empty()
                    status.info("LLaVA/Llama 모델을 메모리에 불러오는 중입니다.")
                    try:
                        result = get_model_manager().preload_for_config(build_preload_config(ui, mode), hf_token=hf_token or None)
                        message = f"사전 로딩 완료 · LLaVA: {result.get('llava')} / Llama: {result.get('llama')}"
                        st.session_state["models_preloaded_key"], st.session_state["models_preloaded_status"] = preloaded_key(ui), message
                        status.success(message)
                    except Exception as error:
                        status.error(f"모델 사전 로딩 실패: {error}")
        else:
            st.caption("현재 방식은 LLaVA/Llama 사전 로딩 없이 실행됩니다.")
    return navigation, mode, ui, hf_token or None


def video_selection(ui: dict, mode: str, hf_token: str | None) -> None:
    st.markdown('<div class="section-kicker">VIDEO SOURCE</div><div class="panel-heading">분석할 영상 선택</div><div class="panel-subtitle">업로드 후 바로 키프레임 추출을 시작하거나, 기존 결과를 좌측에서 선택하세요.</div>', unsafe_allow_html=True)
    left, right = st.columns([2.2, 1.2], gap="large")
    with left:
        uploaded = st.file_uploader("CCTV 또는 휴대폰 영상 업로드", type=sorted(extension.lstrip(".") for extension in SUPPORTED_VIDEO_EXTS), key=f"video_uploader_{st.session_state['uploader_reset']}", help="현재는 업로드 파일을 연결하며, 향후 실시간 휴대폰 스트림을 같은 영상 영역에 연결할 수 있습니다.")
        if uploaded:
            target = UPLOAD_DIR / safe_name(uploaded.name)
            signature = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("uploaded_signature") != signature or not target.exists():
                target.write_bytes(uploaded.getvalue())
                st.session_state["uploaded_signature"] = signature
            st.session_state["active_video"], st.session_state["active_run"], st.session_state["selected_keyframe"] = str(target), None, None
            st.success(f"연결됨: {uploaded.name}")
    with right:
        active = existing_path(st.session_state.get("active_video"))
        st.markdown(f'<div class="result-info"><b>현재 영상</b><br>{html.escape(active.name) if active else "선택된 영상이 없습니다."}<br><small>{"업로드 영상 · 분석 대기" if active else "좌측 결과를 선택하거나 영상을 업로드하세요."}</small></div>', unsafe_allow_html=True)
    active = existing_path(st.session_state.get("active_video"))
    if st.button("키프레임 추출 시작", type="primary", disabled=active is None):
        if mode in LLM_MODES and not models_ready(ui, mode):
            st.error("LLaVA/Llama 방식은 좌측의 ‘모델 사전 로딩’을 먼저 완료해야 합니다.")
        else:
            with st.spinner("키프레임과 분석 결과를 생성하는 중입니다."):
                output_dir = process_video(active, mode, ui, hf_token)
            st.session_state["active_run"], st.session_state["selected_keyframe"] = str(output_dir), None
            st.success("분석이 완료되었습니다. 키프레임 조회 화면으로 연결합니다.")
            st.rerun()
    with st.expander("감시 폴더의 영상 처리"):
        pending = [video for video in find_watch_videos() if not is_processed(video)]
        st.caption(f"감시 폴더: {WATCH_DIR}")
        st.write(f"처리 대기 영상 {len(pending)}개")
        if pending:
            st.dataframe([{"파일명": video.name, "크기(MB)": round(video.stat().st_size / 1024 / 1024, 2)} for video in pending], use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1: scan_once = st.button("대기 영상 한 번 처리", key="watch_scan")
        with c2: auto_scan = st.toggle("자동 스캔", key="watch_auto")
        interval = st.number_input("자동 스캔 간격(초)", 5, 300, 30, 5, key="watch_interval")
        if scan_once or auto_scan:
            if mode in LLM_MODES and not models_ready(ui, mode):
                st.error("LLaVA/Llama 방식은 모델 사전 로딩 후 사용할 수 있습니다.")
            elif not pending:
                st.info("처리할 대기 영상이 없습니다.")
            else:
                for video in pending:
                    try:
                        output_dir = process_video(video, mode, ui, hf_token)
                        mark_processed(video, output_dir)
                        st.session_state["active_run"], st.session_state["active_video"] = str(output_dir), str(video)
                        st.success(f"완료: {video.name}")
                    except Exception as error:
                        st.error(f"{video.name} 처리 실패: {type(error).__name__}: {error}")
                        break
            if auto_scan:
                time.sleep(int(interval))
                st.rerun()


def keyframe_list(items: list[dict], selected: int | None, run_key: str) -> None:
    st.markdown('<div class="section-kicker">KEYFRAMES</div><div class="panel-heading">키프레임 목록</div><div class="panel-subtitle">항목을 선택하면 중앙 영상과 타임라인의 현재 구간이 바뀝니다.</div>', unsafe_allow_html=True)
    if not items:
        st.markdown('<div class="empty-keyframes">아직 표시할 키프레임이 없습니다.<br>영상을 업로드하고 추출을 시작하거나, 저장된 결과를 선택하세요.</div>', unsafe_allow_html=True)
        return
    st.caption("저장된 위험 상태가 없으면 색상은 키프레임 중요도에 따른 검토 우선도입니다.")
    with st.container(height=570, border=False):
        for item in items:
            item_id = item["display_id"]
            label, style, _ = status_for(item)
            picture, details = st.columns([.85, 1.65], gap="small")
            with picture:
                image = frame_path(item)
                if image: st.image(str(image), use_container_width=True)
                else: st.markdown("<div class='empty-keyframes' style='padding:.65rem .2rem'>이미지 없음</div>", unsafe_allow_html=True)
            with details:
                st.markdown(f'<div class="keyframe-item {"selected" if item_id == selected else ""}"><span class="badge {style}">{html.escape(label)}</span><div class="caption">{html.escape(str(item.get("caption") or "설명 데이터 없음"))}</div><div class="time">구간 {timestamp(item["start_time"])} – {timestamp(item["end_time"])}</div></div>', unsafe_allow_html=True)
                if st.button(f"키프레임 {item_id + 1} 보기", key=f"keyframe_{run_key}_{item_id}", use_container_width=True):
                    st.session_state["selected_keyframe"] = item_id
                    st.rerun()


def video_panel(data: dict, selected: dict | None) -> None:
    video = existing_path(st.session_state.get("active_video")) or existing_path(data["metadata"].get("video_path"))
    name = video.name if video else (data["run"].name if data["run"] else "연결된 영상 없음")
    connection = "분석 결과 연결됨" if data["run"] else ("업로드 영상 연결됨" if video else "영상 선택 대기")
    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="video-meta"><div><div class="section-kicker">LIVE / UPLOAD VIDEO</div><div class="video-name">{html.escape(name)}</div></div><div class="connection"><span class="dot"></span>{html.escape(connection)}</div></div>', unsafe_allow_html=True)
    if video:
        st.video(str(video), start_time=max(0, int(selected["timestamp"])) if selected else 0)
    elif selected and frame_path(selected):
        st.image(str(frame_path(selected)), caption="원본 영상 경로를 찾지 못해 선택된 키프레임을 표시합니다.", use_container_width=True)
    else:
        st.markdown('<div class="video-placeholder"><div class="camera"></div><strong>영상 소스를 연결해 주세요</strong><span>업로드 영상과 향후 휴대폰 실시간 영상이 이 영역에 표시됩니다.</span></div>', unsafe_allow_html=True)
    if selected:
        label, _, explanation = status_for(selected)
        st.markdown(f'<div class="selection-note"><b>선택 구간 · {timestamp(selected["start_time"])}</b> · {html.escape(label)}<br>{html.escape(str(selected.get("caption") or "설명 데이터 없음"))}<br><span style="color:var(--muted)">{html.escape(explanation)}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def timeline_html(items: list[dict], selected: dict | None, visible: float) -> str:
    visible = max(visible, 1)
    parts = ["<div class='timeline-track'></div>"]
    if selected:
        left = max(0, min(100, selected["start_time"] / visible * 100))
        width = max(1.2, min(100 - left, (selected["end_time"] - selected["start_time"]) / visible * 100))
        parts.append(f"<div class='timeline-selected' style='left:{left:.3f}%;width:{width:.3f}%'></div>")
    for item in items:
        if item["timestamp"] > visible: continue
        _, style, _ = status_for(item)
        position = max(0, min(100, item["timestamp"] / visible * 100))
        current = " active" if selected and item["display_id"] == selected["display_id"] else ""
        title = html.escape(f"키프레임 {item['display_id'] + 1} · {timestamp(item['timestamp'])}")
        parts.append(f"<span title='{title}' class='timeline-marker {style}{current}' style='left:{position:.3f}%'></span>")
    for index in range(6):
        position, tick = index / 5 * 100, visible * index / 5
        parts.append(f"<span class='timeline-tick' style='left:{position:.3f}%'>{timestamp(tick)}</span>")
    return "<div class='timeline'>" + "".join(parts) + "</div>"


def timeline(data: dict, selected: dict | None, run_key: str) -> None:
    duration, items = data["duration"], data["items"]
    st.markdown('<div class="timeline-shell">', unsafe_allow_html=True)
    c1, c2 = st.columns([3.3, 2], vertical_alignment="center")
    with c1:
        now = timestamp(selected["timestamp"]) if selected else "--:--"
        st.markdown(f'<div class="timeline-top"><div class="timeline-title">시간대별 키프레임</div><div class="timeline-value">현재 선택 {now} · 전체 {timestamp(duration)}</div></div>', unsafe_allow_html=True)
    with c2:
        range_name = st.radio("표시 범위", ["전체", "1분", "5분", "10분", "30분"], horizontal=True, label_visibility="collapsed", key=f"time_range_{run_key}", help="현재는 표시 범위이며, 향후 해당 단위 분석 제어로 확장할 수 있습니다.")
    unit = {"전체": duration, "1분": 60, "5분": 300, "10분": 600, "30분": 1800}[range_name]
    visible = min(duration, unit) if duration else unit
    if items:
        st.markdown(timeline_html(items, selected, visible), unsafe_allow_html=True)
        st.markdown('<div class="legend"><span><i style="background:#ef4444"></i>검토 우선 높음 / 위험</span><span><i style="background:#f59e0b"></i>검토 우선 보통 / 주의</span><span><i style="background:#10b981"></i>검토 우선 낮음 / 문제없음</span><span>· 실제 상태값이 없으면 중요도 점수를 표시합니다.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-keyframes">표시할 분석 구간이 없습니다.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def report(data: dict) -> None:
    run = data["run"]
    if not run:
        st.info("분석 요약서를 보려면 좌측에서 저장된 분석 결과를 선택하세요.")
        return
    metadata = data["metadata"]
    st.markdown('<div class="section-kicker">ANALYSIS REPORT</div><div class="panel-heading">분석 요약서</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("선택 키프레임", f"{len(data['items'])}개")
    b.metric("영상 길이", timestamp(data["duration"]))
    c.metric("원본 FPS", metadata.get("fps", "-"))
    d.metric("해상도", f"{metadata.get('width')} × {metadata.get('height')}" if metadata.get("width") and metadata.get("height") else "-")
    plot = run / "score_plot.png"
    if plot.exists():
        try:
            st.image(str(plot), caption="세그먼트/키프레임 점수", use_container_width=True)
        except Exception:
            # 이전 실행 결과에 손상된 그래프가 있어도 요약서의 다른 기능은 계속 제공한다.
            st.warning("점수 그래프 파일을 열 수 없어 그래프 표시는 건너뜁니다.")
    with st.expander("Caption / Llama 상세 결과"):
        show_model_reports(run)
    with st.expander("결과 다운로드"):
        c1, c2 = st.columns(2)
        with c1:
            archive = zip_dir(run)
            st.download_button("전체 결과 ZIP 다운로드", archive.read_bytes(), archive.name, "application/zip", use_container_width=True)
        with c2:
            html_report = run / "report.html"
            if html_report.exists(): st.download_button("HTML 리포트 다운로드", html_report.read_bytes(), "report.html", "text/html", use_container_width=True)
            else: st.caption("이 분석 결과에는 HTML 리포트가 없습니다.")


def dashboard(navigation: str, mode: str, ui: dict, hf_token: str | None) -> None:
    if navigation == NAV_VIDEO:
        video_selection(ui, mode, hf_token)
        st.divider()
    data = load_dashboard(existing_path(st.session_state.get("active_run")))
    item_ids = {item["display_id"] for item in data["items"]}
    if item_ids and st.session_state.get("selected_keyframe") not in item_ids:
        st.session_state["selected_keyframe"] = data["items"][0]["display_id"]
    selected = next((item for item in data["items"] if item["display_id"] == st.session_state.get("selected_keyframe")), None)
    run_key = data["run"].name if data["run"] else "no_run"
    list_column, main_column = st.columns([1.03, 2.47], gap="large")
    with list_column: keyframe_list(data["items"], st.session_state.get("selected_keyframe"), run_key)
    with main_column:
        video_panel(data, selected)
        timeline(data, selected, run_key)
    if navigation == NAV_REPORT:
        st.divider()
        report(data)
    elif navigation == NAV_KEYFRAME and not data["items"]:
        st.info("좌측 ‘표시할 분석 결과’에서 저장된 결과를 선택하면 키프레임 목록이 채워집니다.")


def main() -> None:
    ensure_dirs()
    configure_hf_cache(cache_dir())
    init_state()
    apply_theme(st.session_state["theme_dark"])
    render_header()
    runs = list_runs()
    if st.session_state["active_run"] is None and not st.session_state["active_video"] and runs:
        st.session_state["active_run"] = str(runs[0])
        source = existing_path((read_json(runs[0] / "metadata.json") or {}).get("video_path"))
        if source: st.session_state["active_video"] = str(source)
    navigation, mode, ui, hf_token = sidebar(runs)
    dashboard(navigation, mode, ui, hf_token)


if __name__ == "__main__":
    main()
