from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import torch

from networks.model import LLMVS


def _build_llmvs_config(cfg: dict):
    """
    LLMVS 모델 생성에 필요한 최소 config만 만든다.
    TVSum/SumMe dataloader, 평가용 설정은 여기서는 필요 없음.
    """
    return SimpleNamespace(
        model=cfg.get("llmvs_model", "summe_head2_layer3"),
        dataset=cfg.get("llmvs_dataset", "summe"),
        split_idx=int(cfg.get("split_idx", 0)),
        epochs=1,
        reduced_dim=int(cfg.get("reduced_dim", 2048)),
        num_heads=int(cfg.get("num_heads", 2)),
        num_layers=int(cfg.get("num_layers", 3)),
        lr=float(cfg.get("lr", 7e-5)),
    )


def _load_llmvs_checkpoint(cfg: dict, device: str):
    """
    PyTorch Lightning checkpoint를 LLMVS 모델로 로드한다.
    """
    ckpt_path = cfg["checkpoint_path"]
    config = _build_llmvs_config(cfg)

    model = LLMVS.load_from_checkpoint(
        ckpt_path,
        config=config,
        map_location=device,
    )

    model.to(device)
    model.eval()
    return model


def _prepare_llmvs_input(embeddings: np.ndarray, device: str) -> torch.Tensor:
    """
    웹앱에서 생성한 llm_embeddings.npy를 LLMVS forward 입력 형태로 변환한다.

    일반적으로 현재 웹앱 embedding shape:
        [num_segments, 5120]

    LLMVS forward가 기대하는 shape:
        [num_segments, something, 5120]

    원본 LLMVS 학습에서는 user_prompt embedding + generation embedding이 있어서
        [num_segments, 2, 5120]
    형태가 될 수 있다.

    현재 웹앱에서는 segment embedding 하나만 있으므로
        [num_segments, 1, 5120]
    로 넣는다.
    """
    x = torch.tensor(embeddings, dtype=torch.float32, device=device)

    if x.ndim == 2:
        # [N, 5120] -> [N, 1, 5120]
        x = x.unsqueeze(1)

    elif x.ndim == 3:
        # 이미 [N, C, 5120] 형태면 그대로 사용
        pass

    else:
        raise ValueError(
            f"지원하지 않는 embedding shape입니다: {tuple(x.shape)}. "
            "예상 shape은 [num_segments, 5120] 또는 [num_segments, C, 5120]입니다."
        )

    if x.shape[-1] != 5120:
        raise ValueError(
            f"LLMVS checkpoint는 마지막 차원 5120을 기대합니다. "
            f"현재 embedding shape: {tuple(x.shape)}"
        )

    return x


def score_segments_with_llmvs_checkpoint(
    segments: list[dict],
    cfg: dict,
    output_dir: str,
) -> tuple[list[dict], np.ndarray | None]:
    """
    학습 완료된 LLMVS checkpoint를 현재 영상 요약 프로그램의 scoring method로 사용한다.

    전제:
    - output_dir/llm_embeddings.npy 가 이미 존재해야 한다.
    - 이 embedding은 현재 영상의 segment별 Llama embedding이어야 한다.
    - 이 함수는 training/evaluation을 하지 않고, checkpoint forward만 수행한다.
    """

    output_dir = Path(output_dir)
    cache_path = output_dir / "scores_raw_checkpoint.json"
    emb_path = output_dir / "llm_embeddings.npy"

    if cfg.get("cache", True) and cache_path.exists():
        scores = json.loads(cache_path.read_text(encoding="utf-8"))
        embeddings = np.load(emb_path) if emb_path.exists() else None
        return scores, embeddings

    if not emb_path.exists():
        raise FileNotFoundError(
            "llm_embeddings.npy가 없습니다. "
            "LLMVS checkpoint 방식은 현재 영상의 segment embedding이 필요합니다. "
            "pipeline.py에서 checkpoint scoring 전에 Llama embedding 생성 단계를 먼저 실행해야 합니다."
        )

    checkpoint_path = cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("cfg['checkpoint_path']가 설정되어 있지 않습니다.")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 파일을 찾을 수 없습니다: {checkpoint_path}")

    embeddings = np.load(emb_path).astype(np.float32)

    if len(embeddings) == 0:
        raise ValueError("llm_embeddings.npy가 비어 있습니다.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _load_llmvs_checkpoint(cfg, device=device)

    x = _prepare_llmvs_input(embeddings, device=device)

    # 현재 forward에서는 mask를 실제로 쓰지는 않지만, 함수 시그니처상 필요함.
    mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(x, mask=mask)

    if isinstance(out, (tuple, list)):
        out = out[0]

    pred = out.squeeze(-1).detach().float().cpu().numpy()

    # segment 수와 score 수 맞추기
    pred = pred[: len(segments)]

    # checkpoint 출력은 이미 Sigmoid라 0~1 범위지만,
    # 영상 하나 내부에서 top-k를 안정적으로 뽑기 위해 한 번 더 normalize.
    min_v = float(pred.min()) if len(pred) else 0.0
    max_v = float(pred.max()) if len(pred) else 1.0
    denom = max(max_v - min_v, 1e-8)
    pred_norm = (pred - min_v) / denom

    scores = []
    for seg, raw_score, norm_score in zip(segments, pred, pred_norm):
        scores.append(
            {
                "segment_id": seg["segment_id"],
                "local_score": float(norm_score),
                "final_score": float(norm_score),
                "raw_score": float(raw_score),
                "backend": "llmvs_checkpoint",
            }
        )

    cache_path.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return scores, embeddings