# modules/global_context.py

import numpy as np
import torch
import torch.nn as nn


class GlobalContextAggregator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 2, num_layers: int = 3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: [B, T]
        """
        h = self.input_proj(x)
        h = self.encoder(h)
        scores = self.score_head(h).squeeze(-1)
        return scores


def apply_global_context(scores: list[dict], embeddings: np.ndarray, cfg: dict) -> list[dict]:
    """
    scores: Llama가 만든 segment별 local score
    embeddings: segment embedding, shape [T, D]
    """

    if len(scores) == 0:
        return scores

    mode = cfg.get("mode", "transformer")

    if mode == "none":
        for s in scores:
            s["final_score"] = float(s.get("score", 0.0))
        return scores

    if mode == "temporal_smoothing":
        local = np.array([float(s.get("score", 0.0)) for s in scores])
        window = int(cfg.get("smoothing_window", 3))
        half = window // 2

        final = []
        for i in range(len(local)):
            start = max(0, i - half)
            end = min(len(local), i + half + 1)
            final.append(float(local[start:end].mean()))

        for s, fs in zip(scores, final):
            s["final_score"] = fs
        return scores

    # Transformer global context
    hidden_dim = int(cfg.get("hidden_dim", 256))
    num_heads = int(cfg.get("num_heads", 2))
    num_layers = int(cfg.get("num_layers", 3))
    alpha = float(cfg.get("alpha", 0.5))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = np.asarray(embeddings, dtype=np.float32)
    x = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)

    model = GlobalContextAggregator(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    model.eval()

    with torch.no_grad():
        global_scores = model(x).squeeze(0).cpu().numpy()

    local_scores = np.array([float(s.get("score", 0.0)) for s in scores])

    # 학습된 weight가 없는 상태라 transformer score만 믿으면 위험함.
    # 그래서 Llama local score와 global context score를 fusion.
    final_scores = alpha * local_scores + (1.0 - alpha) * global_scores

    for s, gs, fs in zip(scores, global_scores, final_scores):
        s["global_context_score"] = float(gs)
        s["final_score"] = float(fs)

    return scores