from pathlib import Path
import json
from datetime import datetime
from time import perf_counter
from typing import Callable, Optional

from modules.video_loader import load_video_metadata
from modules.frame_sampler import sample_frames
from modules.cluframe_preprocessor import run_cluframe_preprocessing
from modules.segment_builder import build_segments
from modules.caption_generator import generate_segment_captions
from modules.llm_scorer import score_segments_with_llama2
from modules.global_context import apply_global_context
from modules.summary_selector import select_summary
from modules.result_writer import write_json, save_selected_frames
from modules.visualizer import save_score_plot, save_html_report
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.model_manager import ModelManager


ProgressCallback = Optional[Callable[[str, dict | None], None]]


def format_duration(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _notify(callback: ProgressCallback, message: str, payload: dict | None = None) -> None:
    if callback:
        callback(message, payload)
    print(message)


def run_llmvs_pipeline(cfg: dict, hf_token: str | None = None, progress_callback: ProgressCallback = None, model_manager: "ModelManager | None" = None) -> dict:
    """Run the original LLMVS-style pipeline from an already-loaded config dict.

    This is separated from the CLI so Streamlit/FastAPI/etc. can call the pipeline
    without using terminal-only getpass input.
    """
    start_time = datetime.now()
    start_perf = perf_counter()

    video_path = cfg["input"]["video_path"]
    run_name = cfg["project"].get("run_name", "run")
    output_root = Path(cfg["project"].get("output_dir", "./outputs"))
    output_dir = output_root / run_name
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_info = {
        "started_at": start_time.isoformat(timespec="seconds"),
        "ended_at": None,
        "elapsed_seconds": None,
        "elapsed_hms": None,
        "status": "running",
    }
    write_json(output_dir / "runtime.json", runtime_info)

    _notify(progress_callback, "[START] 파이프라인 시작", {"output_dir": str(output_dir)})

    if model_manager is not None:
        _notify(progress_callback, "[PRELOAD] 시작 시점에 LLaVA/Llama 모델 확인/로딩 중")
        model_manager.preload_for_config(cfg, hf_token=hf_token)

    try:
        _notify(progress_callback, "[1/8] 비디오 메타데이터 읽는 중")
        metadata = load_video_metadata(video_path)
        metadata["config"] = cfg
        metadata["runtime"] = {"started_at": runtime_info["started_at"]}
        write_json(output_dir / "metadata.json", metadata)

        preprocessing_cfg = cfg.get("preprocessing", {}) or {}
        if preprocessing_cfg.get("enabled", False):
            _notify(progress_callback, "[2/8] CluFrame 중복 제거 전처리 중")
            frames, preprocessing_report = run_cluframe_preprocessing(
                video_path=video_path,
                output_frame_dir=str(frames_dir),
                preprocessing_cfg=preprocessing_cfg,
                sampling_cfg=cfg.get("sampling", {}),
                progress_callback=progress_callback,
            )
            metadata["preprocessing"] = preprocessing_report
            write_json(output_dir / "metadata.json", metadata)
        else:
            _notify(progress_callback, "[2/8] 프레임 샘플링 중")
            frames = sample_frames(
                video_path=video_path,
                sample_fps=float(cfg["sampling"].get("fps", 1)),
                output_frame_dir=str(frames_dir),
                image_ext=cfg["sampling"].get("image_ext", ".jpg"),
            )
        write_json(output_dir / "frames.json", frames)

        _notify(progress_callback, "[3/8] 세그먼트 구성 중")
        segments = build_segments(
            frames,
            segment_sec=float(cfg["segment"].get("segment_sec", 5)),
            representative_frame=cfg["segment"].get("representative_frame", "middle"),
        )

        scoring_cfg = dict(cfg["scoring"])
        backend = scoring_cfg.get("backend", "llama2")

        if backend == "llmvs_checkpoint":
            # 기존 코드 보존:
            # _notify(progress_callback, "[4/8] 대표 프레임 캡션 생성 중")
            # segments = generate_segment_captions(segments, cfg["caption"], str(output_dir), hf_token=hf_token)

            _notify(progress_callback, "[4/8] checkpoint 모드: 대표 프레임 caption 생성 중")
            segments = generate_segment_captions(
                segments,
                cfg["caption"],
                str(output_dir),
                hf_token=hf_token,
                model_manager=model_manager,
            )
            write_json(output_dir / "segments.json", segments)
        else:
            _notify(progress_callback, "[4/8] 대표 프레임 캡션 생성 중")
            segments = generate_segment_captions(
                segments,
                cfg["caption"],
                str(output_dir),
                hf_token=hf_token,
                model_manager=model_manager,
            )
            write_json(output_dir / "segments.json", segments)

        _notify(progress_callback, "[5/8] 중요도 점수 계산 중")

        if backend == "llmvs_checkpoint":
            _notify(progress_callback, "[5/8] Llama embedding 생성 중")

            embedding_cfg = dict(scoring_cfg)
            embedding_cfg["backend"] = "llama2"
            embedding_cfg["extract_embeddings"] = True

            # 여기서 llm_embeddings.npy가 생성됨
            _, embeddings = score_segments_with_llama2(
                segments,
                embedding_cfg,
                str(output_dir),
                hf_token=hf_token,
                model_manager=model_manager,
            )

            _notify(progress_callback, "[6/8] LLMVS checkpoint로 중요도 점수 계산 중")

            from modules.llmvs_checkpoint_scorer import score_segments_with_llmvs_checkpoint

            scores, embeddings = score_segments_with_llmvs_checkpoint(
                segments,
                scoring_cfg,
                str(output_dir),
            )

        else:
            scores, embeddings = score_segments_with_llama2(
                segments,
                scoring_cfg,
                str(output_dir),
                hf_token=hf_token,
                model_manager=model_manager,
            )

        _notify(progress_callback, "[6/8] Global context 적용 중")
        scores = apply_global_context(
            scores,
            embeddings,
            cfg.get("global_context", {})
        )
        write_json(output_dir / "scores.json", scores)

        _notify(progress_callback, "[7/8] 요약 키프레임 선택 중")
        summary = select_summary(segments, scores, cfg["selection"])
        if cfg.get("output", {}).get("save_selected_frames", True):
            save_selected_frames(summary, str(output_dir))
        write_json(output_dir / "summary.json", summary)

        _notify(progress_callback, "[8/8] 리포트 저장 중")
        if cfg.get("output", {}).get("save_graph", True):
            save_score_plot(scores, summary, str(output_dir))
        if cfg.get("output", {}).get("save_html_report", True):
            save_html_report(metadata, segments, scores, summary, str(output_dir))

        runtime_info["status"] = "success"
        result = {
            "status": "success",
            "output_dir": str(output_dir),
            "num_frames": len(frames),
            "num_segments": len(segments),
            "num_selected": len(summary),
            "runtime": runtime_info,
        }
        _notify(progress_callback, "[DONE] 완료", result)
        return result

    except Exception as exc:
        runtime_info["status"] = "failed"
        runtime_info["error"] = f"{type(exc).__name__}: {exc}"
        _notify(progress_callback, "[ERROR] 실패", {"error": runtime_info["error"]})
        raise

    finally:
        end_time = datetime.now()
        elapsed = perf_counter() - start_perf
        runtime_info["ended_at"] = end_time.isoformat(timespec="seconds")
        runtime_info["elapsed_seconds"] = round(elapsed, 3)
        runtime_info["elapsed_hms"] = format_duration(elapsed)
        write_json(output_dir / "runtime.json", runtime_info)
