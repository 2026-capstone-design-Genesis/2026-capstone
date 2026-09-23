def select_summary(segments: list[dict], scores: list[dict], cfg: dict) -> list[dict]:
    mode = cfg.get("mode", "top_k")
    score_by_id = {s["segment_id"]: s for s in scores}
    enriched = []
    for seg in segments:
        sc = score_by_id.get(seg["segment_id"], {})
        enriched.append({**seg, **sc})

    if mode == "top_k":
        k = int(cfg.get("top_k", 5))
        selected = sorted(enriched, key=lambda x: x.get("final_score", 0), reverse=True)[:k]
    elif mode == "threshold":
        th = float(cfg.get("threshold", 0.65))
        selected = [x for x in enriched if x.get("final_score", 0) >= th]
    elif mode == "top_ratio":
        ratio = float(cfg.get("summary_ratio", 0.15))
        k = max(1, int(round(len(enriched) * ratio)))
        selected = sorted(enriched, key=lambda x: x.get("final_score", 0), reverse=True)[:k]
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    return sorted(selected, key=lambda x: x["start_time"])
