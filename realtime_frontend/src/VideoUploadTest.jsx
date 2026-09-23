import React, { useEffect, useState } from 'react';
import { Film, LoaderCircle, Upload, Download } from 'lucide-react';

const time = value => `${Math.floor((value || 0) / 60)}:${String(Math.floor((value || 0) % 60)).padStart(2, '0')}`;
const savedJob = 'clullm-upload-test';

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || '요청을 처리하지 못했습니다.');
  return data;
}

export default function VideoUploadTest() {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(() => {
    const id = sessionStorage.getItem(savedJob);
    return id ? { id, status: 'analyzing' } : null;
  });
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const busy = uploading || job?.status === 'analyzing';
  const result = job?.result;
  const rows = result?.observations || [];
  const selected = rows.find(row => row.id === selectedId) || rows[0];

  useEffect(() => {
    if (!job?.id || !['analyzing', 'done'].includes(job.status)) return;
    let disposed = false, timer;
    async function poll() {
      try {
        const response = await fetch(`/api/upload-tests/${job.id}`);
        if (response.status === 404) {
          sessionStorage.removeItem(savedJob);
          if (!disposed) setJob(null);
        }
        const data = await readResponse(response);
        if (!disposed) { setJob(data); setError(''); }
        if (!disposed && ['analyzing', 'done'].includes(data.status)) timer = setTimeout(poll, data.status === 'done' ? 5000 : 2000);
      } catch (problem) {
        if (!disposed) { setError(problem.message); timer = setTimeout(poll, 4000); }
      }
    }
    poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [job?.id, job?.status]);

  async function analyze(event) {
    event.preventDefault();
    if (!file || busy) return;
    if (file.size > 1024 ** 3) { setError('영상은 최대 1GB까지 업로드할 수 있습니다.'); return; }
    setUploading(true); setError(''); setSelectedId(null);
    try {
      const data = await readResponse(await fetch(`/api/upload-tests?filename=${encodeURIComponent(file.name)}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file,
      }));
      sessionStorage.setItem(savedJob, data.id);
      setJob(data);
    } catch (problem) { setError(problem.message); }
    finally { setUploading(false); }
  }

  return <section className="upload-test-screen" aria-label="영상 업로드 테스트">
    <div className="page-heading"><div className="eyebrow">VIDEO TEST</div><h1>영상 업로드 테스트</h1><p>영상 전체에 HieTaSkim을 실행하고, 선택된 클립의 AI 설명과 전체 요약을 생성합니다.</p></div>
    <form className="upload-test-form" onSubmit={analyze}>
      <label htmlFor="test-video"><Film size={20} /><b>테스트 영상</b><span>최대 1GB · MP4 / MOV / AVI / MKV / WebM</span></label>
      <input id="test-video" type="file" accept=".mp4,.mov,.avi,.mkv,.webm,.m4v" disabled={busy} onChange={event => setFile(event.target.files[0] || null)} />
      <button className="button primary" type="submit" disabled={!file || busy}>{busy ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />}{uploading ? '업로드 중…' : job?.status === 'analyzing' ? '분석 중…' : '분석 실행'}</button>
    </form>
    {busy && <p className="upload-test-status" role="status">{uploading ? '영상을 서버로 업로드하고 있습니다.' : `${job.filename || '업로드 영상'} 분석 중 · 긴 영상은 수 분 걸릴 수 있습니다. 완료되면 결과가 자동으로 표시됩니다.`}</p>}
    {(error || job?.error) && <p className="message error" role="alert">{error || job.error}</p>}
    {result && <>
      <div className="upload-test-result-heading"><h2>{job.filename} 분석 결과</h2><a className="button subtle compact" href={`/api/history/${job.run}/batch_0001.json`} download="upload-test-result.json"><Download size={15} /> 결과 JSON</a></div>
      <div className="analysis-metrics upload-test-metrics">
        <div><span>영상 길이</span><b>{time(result.batch.end)}</b></div>
        <div><span>Component</span><b>{result.batch.skimming?.region_count ?? '—'}</b></div>
        <div><span>키프레임 / 클립 후보</span><b>{rows.length}</b></div>
        <div><span>저화질 키프레임</span><b>{rows.filter(row => row.quality_status === 'rejected').length}</b></div>
      </div>
      <p className="upload-test-settings">시간 범위 {result.batch.skimming?.delta_t_seconds}초 · γ {result.batch.skimming?.gamma} · 요약 비율 {Math.round((result.batch.skimming?.summary_ratio || 0) * 100)}% · {result.batch.skimming?.hierarchy} watershed</p>
      {result.batch.skimming?.clips_before_dedup != null && <p className="upload-test-settings">선택 클립 {result.batch.skimming.clips_before_dedup}개 → 유사 클립 {result.batch.skimming.duplicates_removed}개 제거 → {result.batch.skimming.clips_after_dedup}개 유지</p>}
      <div className="upload-test-media">
        <section><h3>요약 영상</h3>{result.batch.summary_video_url ? <video aria-label="테스트 요약 영상" src={result.batch.summary_video_url} controls preload="metadata" /> : <p>길이 예산에 맞는 keyshot이 없어 요약 영상이 생성되지 않았습니다.</p>}</section>
        <section><h3>선택 클립 {selected ? `· ${time(selected.timestamp)}` : ''}</h3>{selected ? <>{selected.clip_url ? <video aria-label="테스트 선택 클립" src={selected.clip_url} poster={selected.image_url} controls preload="metadata" /> : <img src={selected.image_url} alt="선택 키프레임" />}<p>{time(selected.start_time)}–{time(selected.end_time)} · 움직임 {selected.motion_percent}% · {selected.quality_status === 'rejected' ? '저화질' : '화질 통과'}</p></> : <p>선택된 키프레임이 없습니다.</p>}</section>
      </div>
      <h3>키프레임 <span>{rows.length}개</span></h3>
      <div className="upload-test-frames">{rows.map((row, index) => <button key={row.id} className={selected?.id === row.id ? 'selected' : ''} aria-pressed={selected?.id === row.id} onClick={() => setSelectedId(row.id)}><img src={row.image_url} loading="lazy" alt={`테스트 키프레임 ${index + 1}`} /><b>#{index + 1} · {time(row.timestamp)}</b><span>{row.quality_status === 'rejected' ? '저화질' : '화질 통과'} · 움직임 {row.motion_percent}%</span></button>)}</div>
    </>}
  </section>;
}
