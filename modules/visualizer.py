from pathlib import Path
import html
import matplotlib.pyplot as plt


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def save_score_plot(scores: list[dict], summary: list[dict], output_dir: str):
    if not scores:
        return None
    xs = [s["segment_id"] for s in scores]
    ys = [s.get("final_score", s.get("local_score", 0)) for s in scores]
    selected_ids = {s["segment_id"] for s in summary}

    plt.figure(figsize=(12, 4))
    plt.plot(xs, ys, marker="o")
    for x, y in zip(xs, ys):
        if x in selected_ids:
            plt.axvspan(x - 0.4, x + 0.4, alpha=0.2)
    plt.xlabel("Segment ID")
    plt.ylabel("Importance score")
    plt.title("Segment Importance Scores")
    plt.tight_layout()
    path = Path(output_dir) / "score_plot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def save_html_report(metadata: dict, segments: list[dict], scores: list[dict], summary: list[dict], output_dir: str):
    score_by_id = {s["segment_id"]: s for s in scores}
    summary_ids = {s["segment_id"] for s in summary}
    rows = []
    caption_rows = []
    llama_rows = []

    for seg in segments:
        sc = score_by_id.get(seg["segment_id"], {})
        final_score = sc.get("final_score", sc.get("local_score", 0))
        selected_badge = "✅" if seg["segment_id"] in summary_ids else ""
        frame_path = seg.get("representative_frame_path", "")
        caption = seg.get("caption", "")
        llm_answer = sc.get("llm_answer", "")
        query = sc.get("query", "")

        rows.append(f"""
        <tr>
          <td>{selected_badge}</td>
          <td>{seg['segment_id']}</td>
          <td>{seg['start_time']:.1f} - {seg['end_time']:.1f}</td>
          <td>{_esc(sc.get('raw_score_0_10', ''))}</td>
          <td>{float(final_score):.3f}</td>
          <td>{_esc(caption)}</td>
          <td>{_esc(llm_answer)}</td>
          <td><img src="{_esc(frame_path)}" width="180"></td>
        </tr>
        """)

        caption_rows.append(f"""
        <tr>
          <td>{seg['segment_id']}</td>
          <td>{seg['start_time']:.1f} - {seg['end_time']:.1f}</td>
          <td>{_esc(caption)}</td>
          <td><img src="{_esc(frame_path)}" width="180"></td>
        </tr>
        """)

        llama_rows.append(f"""
        <tr>
          <td>{selected_badge}</td>
          <td>{seg['segment_id']}</td>
          <td>{_esc(sc.get('raw_score_0_10', ''))}</td>
          <td>{float(final_score):.3f}</td>
          <td>{_esc(llm_answer)}</td>
          <td><details><summary>prompt/query 보기</summary><pre>{_esc(query)}</pre></details></td>
        </tr>
        """)

    selected = "".join([
        f"<li>Segment {s['segment_id']} ({s['start_time']:.1f}s-{s['end_time']:.1f}s), score={float(s.get('final_score', 0)):.3f}<br>{_esc(s.get('caption', ''))}</li>"
        for s in summary
    ])

    html_doc = f"""
    <html><head><meta charset='utf-8'><title>LLMVS Test Report</title>
    <style>
      body{{font-family:Arial,sans-serif;margin:24px;line-height:1.45;color:#111827}}
      table{{border-collapse:collapse;width:100%;margin:12px 0 32px 0}}
      td,th{{border:1px solid #ddd;padding:8px;vertical-align:top}}
      th{{background:#f3f4f6}}
      pre{{white-space:pre-wrap;background:#f8fafc;padding:10px;border-radius:8px;max-height:360px;overflow:auto}}
      .section{{margin-top:34px}}
      .note{{background:#eef2ff;border:1px solid #c7d2fe;border-radius:12px;padding:12px}}
    </style>
    </head><body>
    <h1>LLMVS Test Report</h1>
    <p class="note">이 리포트는 선택된 키프레임뿐 아니라 LLaVA caption 결과와 Llama 중요도 평가 결과를 함께 확인하기 위한 상세 보고서입니다.</p>

    <div class="section">
      <h2>Metadata</h2>
      <pre>{_esc(metadata)}</pre>
    </div>

    <div class="section">
      <h2>Selected Summary Segments</h2>
      <ol>{selected}</ol>
    </div>

    <div class="section">
      <h2>Score Plot</h2>
      <img src="score_plot.png" width="900">
    </div>

    <div class="section">
      <h2>Caption Report</h2>
      <table>
        <tr><th>ID</th><th>Time</th><th>LLaVA Caption</th><th>Frame</th></tr>
        {''.join(caption_rows)}
      </table>
    </div>

    <div class="section">
      <h2>Llama Importance Report</h2>
      <table>
        <tr><th>Selected</th><th>ID</th><th>Raw 0-10</th><th>Final</th><th>Llama Answer</th><th>Prompt / Query</th></tr>
        {''.join(llama_rows)}
      </table>
    </div>

    <div class="section">
      <h2>All Segments</h2>
      <table>
        <tr><th>Selected</th><th>ID</th><th>Time</th><th>Raw 0-10</th><th>Final</th><th>Caption</th><th>Llama Answer</th><th>Frame</th></tr>
        {''.join(rows)}
      </table>
    </div>
    </body></html>
    """
    path = Path(output_dir) / "report.html"
    path.write_text(html_doc, encoding="utf-8")
    return str(path)
