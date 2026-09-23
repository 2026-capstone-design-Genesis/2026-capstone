import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { QRCodeSVG } from 'qrcode.react';
import { Aperture, ArrowLeft, ArrowRight, Camera, Check, ChevronRight, Clock3, Copy, Download, Expand, Film, Focus, Info, Layers3, Link2, LoaderCircle, Moon, Play, QrCode, Radio, RotateCcw, ScanLine, Settings2, ShieldCheck, Smartphone, Square, Sun, Video, Wifi, WifiOff, X } from 'lucide-react';
import './styles.css';
import VideoUploadTest from './VideoUploadTest';

const INTERVALS = [{ value: 30, label: '30초' }, { value: 60, label: '1분' }, { value: 300, label: '5분' }, { value: 600, label: '10분' }, { value: 1800, label: '30분' }, { value: 3600, label: '1시간' }];
const labelFor = value => INTERVALS.find(item => item.value === value)?.label || '5분';
const formatTime = value => {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60].map(item => String(item).padStart(2, '0')).join(':');
};
const socketUrl = path => `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${path}`;
const qualityReason = reasons => (reasons || []).map(reason => ({
  too_dark: '화면이 너무 어두움', too_bright: '화면이 너무 밝음',
  low_information: '화면 정보가 부족함', blurred: '화면이 흐림', empty_frame: '빈 프레임',
})[reason] || reason).join(' · ');

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '요청을 처리하지 못했습니다. 다시 시도해 주세요.');
  return data;
}

function useCameraSession(sessionId) {
  const [snapshot, setSnapshot] = useState(null);
  const [frame, setFrame] = useState(null);
  const [online, setOnline] = useState(false);
  useEffect(() => {
    setSnapshot(null); setFrame(null); setOnline(false);
    if (!sessionId) return;
    let disposed = false, socket, timer, retry = 0;
    const connect = () => {
      if (disposed) return;
      socket = new WebSocket(socketUrl(`/ws/viewer/${sessionId}`));
      socket.binaryType = 'blob';
      socket.onopen = () => { if (!disposed) { setOnline(true); retry = 0; } };
      socket.onmessage = event => {
        if (disposed) return;
        if (typeof event.data === 'string') {
          try { setSnapshot(JSON.parse(event.data)); } catch { /* 다음 정상 상태 메시지를 기다린다. */ }
        } else setFrame(URL.createObjectURL(event.data));
      };
      socket.onclose = () => {
        if (!disposed) {
          setOnline(false);
          timer = setTimeout(connect, Math.min(1000 * 2 ** retry++, 10000));
        }
      };
      socket.onerror = () => socket.close();
    };
    connect();
    return () => { disposed = true; clearTimeout(timer); socket?.close(); };
  }, [sessionId]);
  useEffect(() => () => { if (frame) URL.revokeObjectURL(frame); }, [frame]);
  return { snapshot, frame, online };
}

function EmptyView({ title = '연결된 카메라가 없습니다', description = 'QR 코드를 스캔해 휴대폰 카메라를 연결하세요.' }) {
  return <div className="empty-view"><div className="focus-symbol"><Focus size={62} strokeWidth={1} /><Video size={24} /></div><h3>{title}</h3><p>{description}</p></div>;
}

function CameraTile({ camera, selected, onSelect, onUpdate, onStop }) {
  const { snapshot, frame, online } = useCameraSession(camera.id);
  const current = snapshot || camera;
  const live = online && current.status === 'live';
  useEffect(() => { if (snapshot) onUpdate(camera.id, snapshot); }, [camera.id, snapshot, onUpdate]);
  const fullscreen = async event => {
    event.stopPropagation();
    const panel = event.currentTarget.closest('.monitor-panel');
    if (panel?.requestFullscreen) await panel.requestFullscreen();
  };
  const state = current.status === 'ended' ? 'ended' : live ? 'live' : online ? (current.received_frames ? 'reconnecting' : 'waiting') : 'reconnecting';
  return <section className={`monitor-panel camera-tile ${selected ? 'selected-camera' : ''}`} onClick={() => onSelect(camera.id)}>
    <div className="monitor-header"><div className="camera-title"><Video size={18} /><h1>{current.label || `카메라 ${String(current.slot || 1).padStart(2, '0')}`} <span>{String(current.slot || 1).padStart(2, '0')}</span></h1><span className={`live-pill ${live ? 'on' : ''}`}>{state}</span></div><div className="monitor-tools"><button className="icon-button" onClick={fullscreen} aria-label={`${current.label || '카메라'} 영상 전체 화면`}><Expand size={17} /></button></div></div>
    <div className="video-stage">{frame && current.status !== 'ended' ? <img className={`live-image ${!live ? 'stale' : ''}`} src={frame} alt="휴대폰 실시간 카메라 영상" /> : <EmptyView title={current.status === 'ended' ? '카메라 연동이 종료되었습니다' : '프레임 대기 중'} description={current.status === 'ended' ? '이 카메라는 종료되었습니다.' : '휴대폰에서 QR을 스캔해 카메라를 연결하세요.'} />}
      {frame && <div className="video-overlay-top"><span><Focus size={14} />{current.label}</span><span>{formatTime(current.elapsed_seconds)}</span></div>}
    </div>
    <div className="monitor-footer"><div><i className={`status-dot ${live ? 'live' : ''}`} /><span>{state}</span><span className="footer-separator" /><span>수신 시간 <b>{formatTime(current.elapsed_seconds)}</b></span></div><button className="text-button stop" onClick={event => { event.stopPropagation(); onStop(camera.id); }} disabled={current.status === 'ended'}><Square size={12} /> 연동 종료</button></div>
  </section>;
}

function KeyframeDrawer({ keyframes, selectedId, onSelect, onClose }) {
  const closeRef = useRef(null);
  const drag = useRef(null);
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    closeRef.current?.focus();
    const escape = event => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, [onClose]);
  const endDrag = event => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (drag.current) drag.current.active = false;
    setDragging(false);
  };
  return <section id="keyframe-drawer" className="keyframe-drawer" aria-label="추출된 키프레임 패널">
    <div className="drawer-heading"><div><h2><Layers3 size={18} /> 추출된 키프레임 <span>{keyframes.length}</span></h2><p>위아래로 드래그하거나 스크롤해 탐색하세요.</p></div><button ref={closeRef} className="icon-button" onClick={onClose} aria-label="키프레임 패널 닫기"><X size={18} /></button></div>
    <div className={`keyframe-list drawer-list ${dragging ? 'dragging' : ''}`} tabIndex={0} aria-label="키프레임 목록" onPointerDown={event => {
      // 터치는 브라우저의 기본 스와이프를 사용하고 마우스만 직접 처리한다.
      drag.current = event.pointerType === 'mouse' && event.button === 0 ? { active: true, moved: false, y: event.clientY, top: event.currentTarget.scrollTop } : null;
    }} onPointerMove={event => {
      const current = drag.current;
      if (!current?.active) return;
      const delta = event.clientY - current.y;
      if (!current.moved && Math.abs(delta) < 6) return;
      if (!current.moved) { current.moved = true; setDragging(true); event.currentTarget.setPointerCapture(event.pointerId); }
      event.preventDefault();
      event.currentTarget.scrollTop = current.top - delta;
    }} onPointerUp={endDrag} onPointerCancel={endDrag} onLostPointerCapture={endDrag} onClickCapture={event => {
      // 드래그를 마친 손떼기 동작이 썸네일 클릭으로 처리되지 않게 한다.
      if (drag.current?.moved && event.detail > 0) { event.preventDefault(); event.stopPropagation(); }
    }}>
      {keyframes.length ? [...keyframes].reverse().map((item, index) => <button className={`keyframe-card ${selectedId === item.id ? 'selected' : ''}`} aria-pressed={selectedId === item.id} key={item.id} onClick={() => onSelect(item.id)}><img src={item.image_url} alt={`키프레임 ${keyframes.length - index}`} loading="lazy" draggable={false} /><div><span className="keyframe-time">{formatTime(item.timestamp)}</span><strong>{item.title}</strong><span className="unreviewed">미분류</span></div><ChevronRight size={16} /></button>) : <div className="empty-list"><Film size={32} strokeWidth={1.3} /><b>첫 분석을 기다리고 있어요</b><p>분석이 완료되면 추출된 키프레임이<br />이곳에 최신순으로 표시됩니다.</p></div>}
    </div><div className="drawer-footnote">썸네일을 클릭하면 영상 영역에서 크게 확인할 수 있습니다.{keyframes.length >= 200 && ' 최근 200개를 표시합니다.'}</div>
  </section>;
}

function AnalysisDrawer(props) {
  const [historyRows, setHistoryRows] = useState(null);
  const [checkedHistory, setCheckedHistory] = useState([]);
  const [deleteNotice, setDeleteNotice] = useState('');
  const [archive, setArchive] = useState(null);
  const [archiveSelected, setArchiveSelected] = useState(null);
  const [historyError, setHistoryError] = useState('');
  const [busy, setBusy] = useState(false);
  const latest = props.batches.at(-1);
  const rows = archive ? archive.observations : props.keyframes.filter(item => !latest || item.batch_id === latest.id);
  const queueCount = archive ? rows.filter(item => ['pending', 'running', 'waiting'].includes(item.caption_status)).length : props.captionQueueCount;
  const batch = archive?.batch || latest;
  const report = { ...(archive?.report || props.report),
    recorded_seconds: batch ? batch.end - batch.start : 0,
    source_frames: batch?.source_frames || 0, motion_frames: batch?.motion_frames || 0,
    latest_summary_video_url: batch?.summary_video_url,
    latest_recording_url: batch?.recording_url };
  async function openHistory() {
    setBusy(true); setHistoryError('');
    try { setHistoryRows(await api('/history')); setCheckedHistory([]); }
    catch (error) { setHistoryError(error.message); }
    finally { setBusy(false); }
  }
  async function openRecord(row) {
    setBusy(true); setHistoryError('');
    try {
      setArchive(await api(`/history/${encodeURIComponent(row.run)}/${encodeURIComponent(row.file)}`));
      setArchiveSelected(null); setHistoryRows(null);
    } catch (error) { setHistoryError(error.message); }
    finally { setBusy(false); }
  }
  async function deleteCheckedHistory() {
    setBusy(true); setHistoryError('');
    try {
      const items = historyRows.filter(row => checkedHistory.includes(`${row.run}/${row.file}`));
      const response = await fetch('/api/history/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ items: items.map(({ run, file }) => ({ run, file })) }) });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error([404, 405].includes(response.status) ? '삭제 기능이 실행 중인 서버에 반영되지 않았거나 기록이 변경되었습니다. 서버 재시작 후 보관함을 다시 열어 주세요.' : result.detail || '삭제하지 못했습니다.');
      setDeleteNotice(`분석 ${result.deleted}개와 파일 ${result.removed_files || 0}개를 삭제했습니다.`);
      setArchive(null); setArchiveSelected(null); setCheckedHistory([]);
      setHistoryRows(await api('/history'));
    } catch (error) { setHistoryError(error.message); }
    finally { setBusy(false); }
  }
  const controls = <div className="analysis-media-actions">
    <button className="button subtle compact" disabled={busy} onClick={() => { setArchive(null); setHistoryRows(null); }}>현재 분석</button>
    <button className="button subtle compact" disabled={busy} onClick={openHistory}>분석 보관함</button>
    {archive && <span>{archive.batch.completed_at?.replace('T', ' ').slice(0, 19)}</span>}
    {historyError && <p role="alert">{historyError}</p>}
    {deleteNotice && <p role="status">{deleteNotice}</p>}
    {historyRows && <div style={{width: '100%'}}><h3>분석 보관함</h3>
      {!historyRows.length && <p>저장된 분석이 없습니다.</p>}
      {!!historyRows.length && <div className="history-delete-controls"><label><input type="checkbox" aria-label="분석 전체 선택" disabled={busy} checked={checkedHistory.length === historyRows.length} onChange={event => setCheckedHistory(event.target.checked ? historyRows.map(row => `${row.run}/${row.file}`) : [])} /> 전체 선택</label><button className="button subtle compact" disabled={busy || !checkedHistory.length} onClick={deleteCheckedHistory}>선택 삭제 ({checkedHistory.length})</button><small>선택한 분석의 클립·키프레임·요약 영상도 삭제합니다. 원본 영상은 유지합니다.</small></div>}
      {historyRows.map(row => <div className="history-select-row" key={`${row.run}/${row.file}`}><input type="checkbox" aria-label={`${row.run} ${row.file} 선택`} disabled={busy} checked={checkedHistory.includes(`${row.run}/${row.file}`)} onChange={event => setCheckedHistory(current => event.target.checked ? [...current, `${row.run}/${row.file}`] : current.filter(key => key !== `${row.run}/${row.file}`))} /><button className="button subtle full" disabled={busy} onClick={() => openRecord(row)}>
        {row.run} · {formatTime(row.start)}–{formatTime(row.end)} · {row.count}개
      </button></div>)}
    </div>}
  </div>;
  return <AnalysisContents {...props} keyframes={historyRows ? [] : rows} report={historyRows ? {} : report} batches={historyRows ? [] : batch ? [batch] : []}
    captionQueueCount={queueCount}
    captionWorker={props.captionWorker}
    selectedId={archive ? archiveSelected : props.selectedId}
    onSelect={archive ? setArchiveSelected : props.onSelect} analyzing={!archive && props.analyzing} controls={controls} />;
}

function AnalysisContents({ keyframes, report, batches, analyzing, captionQueueCount = 0, captionWorker, selectedId, onSelect, onClose, controls }) {
  const closeRef = useRef(null);
  const selected = keyframes.find(item => item.id === selectedId) || keyframes.at(-1);
  const latestBatch = batches.at(-1);
  useEffect(() => {
    closeRef.current?.focus();
    const escape = event => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, [onClose]);

  return <section id="keyframe-drawer" className="keyframe-drawer analysis-drawer" aria-label="영상 분석 결과 패널">
    <div className="drawer-heading"><div><h2><ScanLine size={18} /> 영상 분석 <span>{keyframes.length}</span></h2><p>저장 영상, 요약 영상, 행동 관찰과 쇼츠를 한곳에서 확인합니다.</p></div><button ref={closeRef} className="icon-button" onClick={onClose} aria-label="영상 분석 패널 닫기"><X size={18} /></button></div>
    <div className="analysis-drawer-scroll">
      {controls}
      {captionQueueCount > 0 && <p className="analysis-queue-status" role="status">GPU 대기열 총 {captionQueueCount}개 · 현재 회차 대기 {keyframes.filter(item => item.caption_status === 'pending').length} · 생성 중 {keyframes.filter(item => item.caption_status === 'running').length} · 미래 프레임 대기 {keyframes.filter(item => item.caption_status === 'waiting').length}</p>}
      {captionQueueCount > 0 && captionWorker && !captionWorker.online && <p className="analysis-queue-error" role="alert">LiveCC GPU worker가 실행되지 않아 설명 작업이 대기 중입니다.</p>}
      {analyzing && <div className="analysis-running"><LoaderCircle size={17} className="spin" /><div><b>저장 영상을 분석하고 있습니다</b><span>Frame Differencing, 키프레임, AI 캡션을 순서대로 처리합니다.</span></div></div>}

      <section className="analysis-overview">
        <div className="analysis-section-title"><div><span className="eyebrow">AI VIDEO SUMMARY</span><h3>AI 영상 요약</h3></div>{report?.ai_model && <span className="model-label">{report.ai_model}</span>}</div>
        <p className="ai-summary-text">{report?.ai_summary || '첫 영상 분석이 완료되면 전체 흐름을 요약해 드립니다.'}</p>
        <div className="analysis-metrics">
          <div><span>저장 영상</span><b>{formatTime(report?.recorded_seconds)}</b></div>
          <div><span>원본 프레임</span><b>{Number(report?.source_frames || 0).toLocaleString()}</b></div>
          <div><span>움직임 프레임</span><b>{Number(report?.motion_frames || 0).toLocaleString()}</b></div>
          <div><span>행동 관찰</span><b>{keyframes.length}</b></div>
        </div>
      </section>

      <section className="analysis-media-section">
        <div className="analysis-section-title"><div><span className="eyebrow">FRAME DIFFERENCING</span><h3>움직임 요약 영상</h3></div>{latestBatch && <span className="analysis-time-range">{formatTime(latestBatch.start)}–{formatTime(latestBatch.end)}</span>}</div>
        {report?.latest_summary_video_url ? <video className="analysis-video" src={report.latest_summary_video_url} controls preload="metadata" /> : <div className="analysis-media-empty"><Film size={28} /><span>분석 완료 후 요약 영상이 표시됩니다.</span></div>}
        <div className="analysis-media-actions">
          {report?.latest_recording_url && <a className="button subtle compact" href={report.latest_recording_url} target="_blank" rel="noreferrer"><Video size={14} /> 저장 영상</a>}
          {report?.latest_summary_video_url && <a className="button subtle compact" href={report.latest_summary_video_url} target="_blank" rel="noreferrer"><Play size={14} /> 요약 영상</a>}
        </div>
      </section>

      {selected && <section className="analysis-selected">
        {selected.caption_status && <p role="status">{({pending: '추론 대기', running: '설명 생성 중', waiting: '미래 프레임 대기', ok: '설명 생성 완료'})[selected.caption_status] || `설명 상태: ${selected.caption_status}`}</p>}
        {selected.quality_status === 'rejected' && <p className="quality-warning" role="status">입력 품질 제외 · {qualityReason(selected.quality_reasons)}</p>}
        {selected.caption_details?.clip_export_error && <p role="alert">설명은 생성됐지만 2초 쇼츠를 만들지 못했습니다.</p>}
        {selected.caption_details?.clip_start_sec != null && <p>설명 관찰 구간 {formatTime(selected.caption_details.clip_start_sec)}–{formatTime(selected.caption_details.clip_end_sec)}</p>}
        {selected.action_clip_url && <a className="button subtle compact" href={selected.action_clip_url} target="_blank" rel="noreferrer">전체 행동 구간 {formatTime(selected.start_time)}–{formatTime(selected.end_time)}</a>}
        <div className="analysis-section-title"><div><span className="eyebrow">SHORT CLIP</span><h3>{formatTime(selected.timestamp)} 행동 구간</h3></div></div>
        {selected.clip_url ? <video className="analysis-video shorts-video" src={selected.clip_url} poster={selected.image_url} controls preload="metadata" /> : <img className="analysis-video shorts-video" src={selected.image_url} alt={`${formatTime(selected.timestamp)} 대표 키프레임`} />}
        <p>{selected.caption}</p>
      </section>}

      <section className="observation-section">
        <div className="analysis-section-title"><div><span className="eyebrow">TIMELINE OBSERVATIONS</span><h3>시간대별 행동 관찰</h3></div></div>
        <div className="observation-list">
          {keyframes.length ? keyframes.map(item => <button key={item.id} className={`observation-row ${selected?.id === item.id ? 'selected' : ''} ${item.quality_status === 'rejected' ? 'quality-rejected' : ''}`} onClick={() => onSelect(item.id)}><img src={item.image_url} alt="" loading="lazy" /><div><span>{formatTime(item.timestamp)} · 변화량 {item.motion_percent}%{item.quality_status === 'rejected' ? ' · 입력 품질 낮음' : ''}</span><b>{item.caption || '행동 설명 생성 중'}</b></div><Play size={15} /></button>) : <div className="empty-list"><Film size={32} strokeWidth={1.3} /><b>첫 영상 분석을 기다리고 있습니다</b><p>분석이 완료되면 시간순 행동 관찰과<br />대표 쇼츠가 이곳에 표시됩니다.</p></div>}
        </div>
      </section>
    </div>
    <div className="drawer-footnote">이 화면은 행동 관찰 결과만 제공하며 이벤트 체크나 위험 태그를 생성하지 않습니다.</div>
  </section>;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('clullm-theme') || 'dark');
  const [step, setStep] = useState(1);
  const [config, setConfig] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [session, setSession] = useState(null);
  const [interval, setIntervalValue] = useState(300);
  const [selectedId, setSelectedId] = useState(null);
  const [keyframesOpen, setKeyframesOpen] = useState(false);
  const keyframesButtonRef = useRef(null);
  const closeKeyframes = React.useCallback(() => { setKeyframesOpen(false); keyframesButtonRef.current?.focus(); }, []);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [loading, setLoading] = useState(true);
  // Each CameraTile owns one viewer socket and cleans it up independently.
  const current = session;
  // The former single-camera panel is visually suppressed; keep these inert values
  // until that legacy JSX is removed so switching to the grid cannot crash.
  const frame = null;
  const online = false;
  const keyframes = current?.keyframes || [];
  const selected = keyframes.find(item => item.id === selectedId);
  const isLive = current?.status === 'live';
  const ended = current?.status === 'ended';
  const viewRef = useRef(null);
  const cctvGridRef = useRef(null);
  const appShellRef = useRef(null);
  const [cctvFullscreen, setCctvFullscreen] = useState(false);
  const [cctvControlsOpen, setCctvControlsOpen] = useState(false);
  const [cctvTimelineOpen, setCctvTimelineOpen] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('clullm-theme', theme);
  }, [theme]);

  const load = async () => {
    setLoading(true); setError('');
    try {
      const parameters = new URLSearchParams(location.hash.slice(1));
      if (parameters.has('admin')) {
        const token = parameters.get('admin');
        history.replaceState(null, '', location.pathname);
        await api('/login', { method: 'POST', body: JSON.stringify({ token }) });
      }
      const nextConfig = await api('/config');
      setConfig(nextConfig);
      const active = await Promise.all(nextConfig.sessions.map(item => api(`/sessions/${item.id}`)));
      setCameras(active);
      const recent = active.at(-1);
      if (recent) {
        setSelectedCameraId(recent.id); setSession(recent); setIntervalValue(recent.interval_seconds);
        if (recent.received_frames > 0) setStep(2);
      }
    } catch (problem) { setError(problem.message); }
    finally { setLoading(false); }
  };
  useEffect(() => { load(); }, []);
  useEffect(() => {
    if (current?.interval_seconds) setIntervalValue(current.interval_seconds);
  }, [current?.interval_seconds]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 5000);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    const syncFullscreen = () => {
      if (document.fullscreenElement !== appShellRef.current) {
        setCctvFullscreen(false); setCctvControlsOpen(false); setCctvTimelineOpen(false);
      }
    };
    document.addEventListener('fullscreenchange', syncFullscreen);
    return () => document.removeEventListener('fullscreenchange', syncFullscreen);
  }, []);

  const action = async task => {
    setPending(true); setError('');
    try { await task(); } catch (problem) { setError(problem.message); }
    finally { setPending(false); }
  };
  const createSession = () => action(async () => {
    const data = await api('/sessions', { method: 'POST', body: JSON.stringify({ interval_seconds: interval }) });
    setCameras(previous => [...previous, data]); setSelectedCameraId(data.id); setSession(data); setSelectedId(null);
  });
  const changeInterval = value => {
    if (value === interval) return;
    if (!session || ended) { setIntervalValue(value); return; }
    action(async () => {
      await api(`/sessions/${session.id}/interval`, { method: 'PATCH', body: JSON.stringify({ interval_seconds: value }) });
      setIntervalValue(value);
      setNotice(`${labelFor(value)} 주기로 변경했습니다. 모아 둔 장면은 다음 분석에 포함됩니다.`);
    });
  };
  const stopSession = () => action(async () => {
    const data = await api(`/sessions/${session.id}/stop`, { method: 'POST' });
    setCameras(previous => previous.filter(item => item.id !== data.id));
    setSession(previous => previous?.id === data.id ? data : previous);
    setNotice(data.error || '카메라 연동을 종료했습니다. 남은 구간의 키프레임을 저장했습니다.');
  });
  const copyLink = () => action(async () => {
    await navigator.clipboard.writeText(session.phone_url);
    setNotice('휴대폰 연결 주소를 복사했습니다.');
  });
  const selectCamera = id => {
    const camera = cameras.find(item => item.id === id);
    if (camera) { setSelectedCameraId(id); setSession(camera); setIntervalValue(camera.interval_seconds); setSelectedId(null); }
  };
  const updateCamera = React.useCallback((id, snapshot) => {
    setCameras(previous => previous.map(item => item.id === id ? { ...item, ...snapshot } : item));
    if (id === selectedCameraId) setSession(previous => previous?.id === id ? { ...previous, ...snapshot } : previous);
  }, [selectedCameraId]);
  const stopCamera = id => { selectCamera(id); action(async () => {
    const data = await api(`/sessions/${id}/stop`, { method: 'POST' });
    setCameras(previous => previous.filter(item => item.id !== id));
    setSession(previous => previous?.id === id ? data : previous);
    setNotice('카메라 연동을 종료했습니다.');
  }); };
  const fullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (viewRef.current?.requestFullscreen) await viewRef.current.requestFullscreen();
      else setNotice('이 브라우저는 전체 화면 전환을 지원하지 않습니다.');
    } catch { setNotice('브라우저에서 전체 화면 전환을 허용해 주세요.'); }
  };
  const fullscreenCctv = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (appShellRef.current?.requestFullscreen) { await appShellRef.current.requestFullscreen(); setCctvFullscreen(true); }
      else setNotice('이 브라우저는 전체 화면 전환을 지원하지 않습니다.');
    } catch { setNotice('브라우저에서 전체 화면 전환을 허용해 주세요.'); }
  };

  return <div ref={appShellRef} className={`app-shell ${cctvFullscreen ? 'cctv-focus-mode' : ''}`}>
    {cctvFullscreen && <><div className="cctv-left-hotspot" onMouseEnter={() => setCctvControlsOpen(true)} /><div className="cctv-bottom-hotspot" onMouseEnter={() => setCctvTimelineOpen(true)} /></>}
    <header className="app-header">
      <a className="brand" href="/" aria-label="CluLLM 관제 홈"><span className="brand-icon"><Aperture size={25} /></span><strong>CluLLM<span>VISION</span></strong></a>
      <div className="header-divider" /><span className="header-title">실시간 영상 관제</span>
      <div className="header-right"><span className="workspace-label"><ShieldCheck size={15} /> 로컬 관제 공간</span><button className="icon-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label={theme === 'dark' ? '밝은 모드로 전환' : '다크 모드로 전환'}>{theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}</button><div className="avatar">관제</div></div>
    </header>

    <div className="workspace">
      <nav className={`nav-rail ${cctvControlsOpen ? 'cctv-controls-open' : ''}`} aria-label="주요 메뉴" onMouseLeave={() => cctvFullscreen && setCctvControlsOpen(false)}>
        <button className={step === 1 ? 'rail-item active' : 'rail-item'} onClick={() => setStep(1)}><QrCode size={22} /><span>카메라 연동</span></button>
        <button className={step === 2 ? 'rail-item active' : 'rail-item'} onClick={() => setStep(2)}><Video size={22} /><span>영상 확인</span></button>
        <div className="rail-bottom"><span className="rail-line" /><Aperture size={20} /><span>CluLLM</span></div>
      </nav>

      <aside className={`side-panel ${cctvControlsOpen ? 'cctv-controls-open' : ''}`} onMouseEnter={() => cctvFullscreen && setCctvControlsOpen(true)} onMouseLeave={() => cctvFullscreen && setCctvControlsOpen(false)}>
        <div className="side-controls">
        <div className="side-heading"><span className="eyebrow">CONTROL PANEL</span><h2>카메라 & 분석</h2><p>연결부터 장면 탐색까지</p></div>
        <section className="source-card"><div className="source-icon"><Smartphone size={22} /></div><div><b>휴대폰 카메라</b><span>{isLive ? '실시간 수신 중' : ended ? '연동 종료' : current?.received_frames ? '수신 대기 중' : '연결 대기 중'}</span></div><i className={`status-dot ${isLive ? 'live' : ''}`} /></section>
        <section className="interval-section"><h3><Clock3 size={16} /> 영상 분석 주기</h3><p>선택한 주기마다 저장 구간을 요약하고 분석합니다.</p><div className="interval-grid">{INTERVALS.map(item => <button aria-pressed={interval === item.value} disabled={pending || (!config && !loading)} key={item.value} className={interval === item.value ? 'interval active' : 'interval'} onClick={() => changeInterval(item.value)}>{item.label}{interval === item.value && <Check size={13} />}</button>)}</div><div className="schedule-box"><div><span>다음 분석까지</span><b>{current && !ended ? formatTime(current.next_analysis_in) : '—'}</b></div><div className="progress-track"><span style={{ width: `${current && !ended ? Math.max(0, Math.min(100, (1 - current.next_analysis_in / interval) * 100)) : 0}%` }} /></div><small>{current?.analysis_status === 'waiting' ? '공용 분석 큐에서 대기 중…' : current?.analysis_status === 'running' ? '저장 영상 분석 중…' : current?.analysis_status === 'error' ? '분석 오류: 다시 시도할 수 있습니다.' : isLive ? '영상 수신 시간을 기준으로 분석합니다.' : '영상이 수신되면 시간이 흐릅니다.'}</small></div><button className="button subtle full" disabled={pending || !current?.pending_candidates || current?.analyzing} onClick={() => action(async () => { await api(`/sessions/${session.id}/analyze`, { method: 'POST' }); setNotice('현재까지 저장한 영상을 분석합니다.'); })}>{current?.analyzing ? <LoaderCircle size={15} className="spin" /> : <ScanLine size={15} />} 지금 분석</button></section>
        </div>
        <button ref={keyframesButtonRef} className={`keyframes-toggle ${keyframesOpen ? 'active' : ''}`} aria-expanded={keyframesOpen} aria-controls="keyframe-drawer" onClick={() => setKeyframesOpen(value => !value)}><ScanLine size={19} /><span><b>영상 분석</b><small>{keyframesOpen ? '분석 결과 접기' : '요약 · 행동 관찰 · 쇼츠'}</small></span><span className="keyframes-count">{current?.total_keyframes || 0}</span><ChevronRight size={17} /></button>
        <div className="side-footnote"><Info size={14} /><span>저장 영상 기반 요약 · 행동 분석</span></div>
      </aside>

      <main className="main-area">
        {keyframesOpen && <AnalysisDrawer keyframes={keyframes} report={current?.analysis_report} batches={current?.batches || []} analyzing={current?.analyzing} captionQueueCount={current?.caption_queue_count || 0} captionWorker={current?.caption_worker} selectedId={selectedId} onClose={closeKeyframes} onSelect={id => { setSelectedId(id); setStep(2); }} />}
        <div className="steps-bar"><div className="step-buttons"><button className={step === 1 ? 'step active' : 'step complete'} onClick={() => setStep(1)}><span>{isLive ? <Check size={13} /> : '01'}</span>카메라 연동</button><ChevronRight size={16} /><button className={step === 2 ? 'step active' : 'step'} onClick={() => setStep(2)}><span>02</span>영상 확인</button><ChevronRight size={16} /><button className={step === 3 ? 'step active' : 'step'} onClick={() => { setStep(3); setKeyframesOpen(false); }}><span>03</span>업로드 테스트</button></div><span className="connection-label"><i className={`status-dot ${isLive ? 'live' : ''}`} />{isLive ? '카메라 연결됨' : ended ? '연동 종료됨' : '카메라 연결 대기'}</span></div>

        {error && <div className="message error" role="alert"><Info size={17} /><span>{error}</span><button onClick={() => setError('')} aria-label="알림 닫기"><X size={15} /></button></div>}
        {notice && <div className="message notice" role="status"><Check size={17} /><span>{notice}</span></div>}

        {config && <div className="upload-test-container" hidden={step !== 3}><VideoUploadTest /></div>}
        {!config ? <div className="startup-state"><div className="round-icon">{loading ? <LoaderCircle className="spin" /> : <ShieldCheck />}</div><h2>{loading ? '관제 화면을 준비하고 있습니다' : '관제 접속 주소로 열어 주세요'}</h2><p>실행기에 표시된 ‘관제 화면’ 주소를 이 브라우저에서 열면 카메라를 연결할 수 있습니다.</p>{!loading && <button className="button primary" onClick={load}><RotateCcw size={16} /> 다시 확인</button>}</div> : step === 3 ? null : step === 1 ?
          <div className="pairing-screen"><div className="page-heading"><div className="eyebrow">CONNECT YOUR CAMERA</div><h1>휴대폰이 관제 카메라가 됩니다.</h1><p>카메라별 QR을 스캔하고 카메라를 허용하세요. 최대 4대를 동시에 관제할 수 있습니다.</p></div>
            <section className="multi-camera-cards" aria-label="카메라 연결 목록">{cameras.map(camera => <article className={`camera-pair-card ${selectedCameraId === camera.id ? 'selected-camera' : ''}`} key={camera.id} onClick={() => selectCamera(camera.id)}><div><b>{camera.label}</b><small>카메라 {String(camera.slot).padStart(2, '0')} · {camera.status === 'live' ? '연결됨' : 'QR 스캔 대기'}</small></div><div className="small-qr"><QRCodeSVG value={camera.phone_url} size={96} marginSize={2} level="M" title={`${camera.label} QR 코드`} /></div><div className="camera-card-actions"><button className="button subtle compact" onClick={event => { event.stopPropagation(); navigator.clipboard.writeText(camera.phone_url).then(() => setNotice('휴대폰 연결 주소를 복사했습니다.')); }}><Copy size={13} /> 링크 복사</button><button className="button subtle compact" onClick={event => { event.stopPropagation(); stopCamera(camera.id); }}><Square size={13} /> 종료</button></div></article>)}<button className="button primary add-camera" disabled={pending || cameras.length >= 4} onClick={createSession}>{pending ? <LoaderCircle className="spin" size={17} /> : <Link2 size={17} />}{cameras.length >= 4 ? '최대 4대 연결됨' : '카메라 추가'}</button></section>
            <div className="pairing-grid"><section className="pairing-card"><div className="card-title"><span className="round-icon"><QrCode size={20} /></span><div><h2>휴대폰 카메라 연결</h2><p>같은 Wi-Fi에 연결된 휴대폰으로 스캔해 주세요.</p></div></div>
              <div className="qr-stage">{session && !ended ? <div className="qr-code"><QRCodeSVG value={session.phone_url} size={206} marginSize={3} level="M" title="휴대폰 카메라 연결 QR 코드" /></div> : <div className="qr-placeholder"><QrCode size={90} strokeWidth={.85} /><span>연결 코드를 생성해 주세요</span></div>}<div className={`qr-status ${isLive ? 'connected' : ''}`}>{isLive ? <><Check size={15} /> 휴대폰 연결 완료</> : <><span className="pulse-dot" /> {session && !ended ? 'QR 스캔 대기 중' : '새 카메라 연결 준비'}</>}</div></div>
              {session && !ended ? <div className="pair-actions"><button className="button subtle" onClick={copyLink} disabled={pending}><Copy size={15} /> 연결 주소 복사</button><button className="button primary" onClick={() => setStep(2)} disabled={!current?.received_frames}>영상 확인하기 <ArrowRight size={16} /></button></div> : <button className="button primary full" onClick={createSession} disabled={pending}>{pending ? <LoaderCircle className="spin" size={17} /> : <Link2 size={17} />} QR 연결 코드 생성</button>}
              <p className="qr-expiry"><ShieldCheck size={13} /> 연결 코드는 24시간 유효하며, 수신 영상은 관제 PC에 저장됩니다.</p>
            </section>
            <section className="guide-card"><div className="eyebrow">HOW TO CONNECT</div><h2>세 단계면 연결 완료</h2><div className="guide-step"><span>1</span><div><h3>같은 Wi-Fi에 연결</h3><p>PC와 휴대폰을 같은 네트워크에 연결하세요.</p></div><Wifi size={20} /></div><div className="guide-step"><span>2</span><div><h3>QR 코드 스캔</h3><p>기본 카메라로 QR 코드를 스캔해 연결 페이지를 여세요.</p></div><QrCode size={20} /></div><div className="guide-step"><span>3</span><div><h3>카메라 허용 후 전송 시작</h3><p>휴대폰에서 ‘카메라 연결’을 누르고 이 화면의 ‘영상 확인’으로 이동하세요.</p></div><Camera size={20} /></div><div className="privacy-note"><ShieldCheck size={20} /><div><b>전송은 직접 시작하고 종료합니다.</b><p>음성은 전송하지 않으며, 수신 영상과 분석 결과는 관제 PC에 저장합니다. 휴대폰 화면은 켜 두세요.</p></div></div><details className="https-guide"><summary>휴대폰에서 카메라가 열리지 않나요?</summary><p>카메라는 신뢰할 수 있는 HTTPS 연결에서 사용할 수 있습니다. 로컬 인증서를 사용 중이면 실행기에 표시된 ‘휴대폰 최초 설정’ 주소에서 인증서를 먼저 등록하세요. 자세한 절차는 프로젝트의 REALTIME_GUIDE.md에 있습니다.</p><p>현재 연결 주소: <code>{config.public_origin}</code></p></details></section></div>
            <div className="flow-strip"><div><Smartphone size={21} /><span>휴대폰 카메라</span></div><span className="flow-dash" /><div><Radio size={21} /><span>실시간 영상 저장</span></div><span className="flow-dash" /><div><ScanLine size={21} /><span>{labelFor(interval)}마다 영상 요약 · AI 분석</span></div></div>
          </div> : <div className={`monitor-screen ${cctvTimelineOpen ? 'cctv-timeline-open' : ''}`} onMouseMove={event => { if (cctvFullscreen && event.clientY < window.innerHeight - 260) setCctvTimelineOpen(false); }}>
            <button className="button subtle cctv-fullscreen-button" onClick={fullscreenCctv}><Expand size={15} /> 관제 전체 화면</button>
            <div ref={cctvGridRef} className={`camera-grid count-${Math.max(1, cameras.length)}`}>{cameras.length ? cameras.map(camera => <CameraTile key={camera.id} camera={camera} selected={camera.id === selectedCameraId} onSelect={selectCamera} onUpdate={updateCamera} onStop={stopCamera} />) : <EmptyView />}</div>
            <section className="monitor-panel" ref={viewRef}><div className="monitor-header"><div className="camera-title"><Video size={18} /><h1>휴대폰 카메라 <span>01</span></h1><span className={`live-pill ${isLive && !selected ? 'on' : ''}`}>{selected ? '키프레임' : isLive ? '실시간' : ended ? '종료됨' : '대기 중'}</span></div><div className="monitor-tools">{selected && <button className="text-button" onClick={() => setSelectedId(null)}><Radio size={14} /> 실시간으로</button>}<button className="icon-button" onClick={fullscreen} aria-label="영상 전체 화면"><Expand size={17} /></button></div></div>
              <div className="video-stage">{selected ? selected.clip_url ? <video className="live-image" src={selected.clip_url} poster={selected.image_url} controls autoPlay muted /> : <img className="live-image" src={selected.image_url} alt={`${formatTime(selected.timestamp)} 선택 키프레임`} /> : frame && !ended ? <img className={`live-image ${!isLive ? 'stale' : ''}`} src={frame} alt={isLive ? '휴대폰 실시간 카메라 영상' : '마지막으로 수신한 영상'} /> : <EmptyView title={ended ? '카메라 연동이 종료되었습니다' : undefined} description={ended ? '왼쪽 영상 분석에서 분석된 장면을 다시 확인할 수 있습니다.' : undefined} />}
                {(selected || (frame && !ended)) && <><div className="video-overlay-top"><span><Focus size={14} />{selected ? '선택 행동 구간' : '휴대폰 카메라 01'}</span><span>{selected ? formatTime(selected.timestamp) : formatTime(current?.elapsed_seconds)}</span></div><div className="video-overlay-bottom"><span><Aperture size={14} />{selected ? selected.caption : '장면 변화 분석 활성화'}</span><span>{selected ? `${selected.motion_percent}% 변화` : `장면 변화 ${current?.motion_percent || 0}%`}</span></div></>}
                {!selected && frame && !ended && !isLive && <div className="reconnect-overlay"><WifiOff size={22} /><b>{online ? '휴대폰 영상 수신이 멈췄습니다' : '관제 서버에 다시 연결 중입니다'}</b><span>마지막 수신 화면 · 분석 시간 일시 정지</span></div>}
              </div>
              <div className="monitor-footer"><div><i className={`status-dot ${isLive ? 'live' : ''}`} /><span>{isLive ? '영상 수신 중' : ended ? '연동 종료' : '연결 대기'}</span><span className="footer-separator" /> <span>수신 시간 <b>{formatTime(current?.elapsed_seconds)}</b></span></div><button className="text-button stop" disabled={pending || !session || ended} onClick={stopSession}>{pending ? <LoaderCircle className="spin" size={13} /> : <Square size={12} />} 연동 종료</button></div>
            </section>
            <section className="timeline-panel"><div className="timeline-heading"><div><h2><Layers3 size={16} /> 키프레임 타임라인</h2><p>{selected ? `${formatTime(selected.timestamp)} · ${selected.title}` : '분석이 끝난 장면을 선택해 확인하세요.'}</p></div><span className="timeline-unit"><Clock3 size={13} /> {labelFor(interval)} 주기</span></div><div className="timeline-ruler">{[0, 1, 2, 3, 4].map(index => <span key={index}>{formatTime((current?.elapsed_seconds || interval) * index / 4)}</span>)}</div><div className="timeline-track" role="group" aria-label="키프레임 시간축"><div className="timeline-baseline" />{keyframes.map(item => <button key={item.id} className={`timeline-marker ${item.id === selectedId ? 'selected' : ''}`} style={{ left: `${Math.min(99, Math.max(1, item.timestamp / Math.max(1, current.elapsed_seconds) * 100))}%` }} onClick={() => setSelectedId(item.id)} aria-label={`${formatTime(item.timestamp)} 키프레임 선택`} title={`${formatTime(item.timestamp)} · ${item.title}`} />)}<div className="timeline-cursor" style={{ left: `${selected ? Math.min(99, selected.timestamp / Math.max(1, current?.elapsed_seconds) * 100) : 99}%` }} /></div><div className="timeline-legend"><span><i /> 추출된 키프레임</span><span><i className="selected" /> 현재 선택</span><span className="timeline-empty-note">{keyframes.length ? '색상은 선택 상태를 나타냅니다.' : '첫 분석 완료 후 키프레임이 표시됩니다.'}</span></div></section>
            <div className="analysis-footer"><div className="analysis-summary"><span className="analysis-icon"><ScanLine size={18} /></span><div><b>{current?.analyzing ? '영상 분석 중' : '주기별 영상 분석'}<span>{current?.batches?.length || 0}회 완료</span></b><p>{current?.error || (ended ? '저장 영상과 분석 결과는 outputs 폴더에 저장되었습니다.' : `${labelFor(interval)}마다 요약 영상과 행동 관찰을 생성합니다.`)}</p></div></div>{current?.total_keyframes > 0 && <a className="button subtle compact" href={`/api/sessions/${session.id}/report`} download><Download size={14} /> 분석 결과</a>}</div>
          </div>}
      </main>
    </div>
  </div>;
}

function PhoneCamera() {
  const [state, setState] = useState('idle');
  const [message, setMessage] = useState('카메라를 연결하면 관제 PC로 영상이 전송됩니다.');
  const [sent, setSent] = useState(0);
  const [interval, setIntervalValue] = useState(300);
  const videoRef = useRef(null);
  const captureRef = useRef({ stream: null, socket: null, timer: null, active: false, wake: null, sequence: 0 });
  const [credentials] = useState(() => {
    const values = new URLSearchParams(location.hash.slice(1));
    if (values.has('session') && values.has('token')) {
      const item = { session: values.get('session'), token: values.get('token') };
      sessionStorage.setItem('clullm-phone', JSON.stringify(item));
      history.replaceState(null, '', '/phone');
      return item;
    }
    try { return JSON.parse(sessionStorage.getItem('clullm-phone')); } catch { return null; }
  });

  const release = () => {
    const capture = captureRef.current;
    capture.active = false;
    capture.sequence++;
    clearTimeout(capture.timer);
    capture.socket?.close(); capture.socket = null;
    capture.stream?.getTracks().forEach(track => track.stop()); capture.stream = null;
    capture.wake?.release().catch(() => {}); capture.wake = null;
  };
  const halt = text => { release(); setState('paused'); setMessage(text); };
  useEffect(() => {
    document.documentElement.dataset.theme = 'dark';
    const hidden = () => { if (document.hidden && captureRef.current.active) halt('화면이 꺼지거나 다른 앱으로 이동해 전송을 일시 정지했습니다. 다시 연결해 주세요.'); };
    document.addEventListener('visibilitychange', hidden);
    window.addEventListener('pagehide', release);
    return () => { release(); document.removeEventListener('visibilitychange', hidden); window.removeEventListener('pagehide', release); };
  }, []);

  const start = async () => {
    if (!credentials) { setMessage('관제 PC에 표시된 QR 코드를 먼저 스캔해 주세요.'); return; }
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setMessage('카메라를 사용하려면 신뢰할 수 있는 HTTPS 연결이 필요합니다. PC 실행기에 표시된 최초 설정 안내를 완료한 뒤 QR 코드를 다시 스캔해 주세요.'); return;
    }
    release();
    const capture = captureRef.current, sequence = capture.sequence;
    capture.active = true;
    setState('connecting'); setMessage('카메라 권한을 허용해 주세요.');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false });
      if (!capture.active || sequence !== capture.sequence) { stream.getTracks().forEach(track => track.stop()); return; }
      capture.stream = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      if (!capture.active || sequence !== capture.sequence) return;
      const socket = new WebSocket(socketUrl(`/ws/phone/${credentials.session}`));
      capture.socket = socket;
      setMessage('관제 PC에 연결하고 있습니다.');
      capture.timer = setTimeout(() => {
        if (capture.active && sequence === capture.sequence) halt('PC 연결 응답이 없습니다. 네트워크와 서버 상태를 확인한 후 다시 연결해 주세요.');
      }, 12000);
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d', { alpha: false });
      const takeFrame = () => {
        if (!capture.active || socket.readyState !== WebSocket.OPEN) return;
        const video = videoRef.current;
        if (video.readyState < 2 || !video.videoWidth) { capture.timer = setTimeout(takeFrame, 250); return; }
        const scale = Math.min(1, 720 / Math.max(video.videoWidth, video.videoHeight));
        canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
          if (blob && capture.active && socket.readyState === WebSocket.OPEN && sequence === capture.sequence) {
            socket.send(blob);
            capture.timer = setTimeout(() => {
              if (capture.active && sequence === capture.sequence) halt('영상 수신 확인이 중단되었습니다. 다시 연결해 주세요.');
            }, 10000);
          }
        }, 'image/jpeg', .62);
      };
      socket.onopen = () => socket.send(JSON.stringify({ token: credentials.token }));
      socket.onmessage = event => {
        if (!capture.active || sequence !== capture.sequence) return;
        let data;
        try { data = JSON.parse(event.data); } catch { return; }
        if (data.type === 'ready') {
          clearTimeout(capture.timer);
          setState('live'); setMessage('관제 PC로 영상이 전송되고 있습니다. 이 화면을 켜 두세요.');
          if (navigator.wakeLock) navigator.wakeLock.request('screen').then(lock => { if (capture.active && sequence === capture.sequence) capture.wake = lock; else lock.release(); }).catch(() => {});
          takeFrame();
        }
        if (data.type === 'ack') { clearTimeout(capture.timer); setSent(value => value + 1); capture.timer = setTimeout(takeFrame, 55); }
        if (data.interval_seconds) setIntervalValue(data.interval_seconds);
      };
      socket.onclose = event => {
        if (capture.active && sequence === capture.sequence) halt(event.reason || 'PC와 연결이 끊겼습니다. 네트워크를 확인하고 다시 연결해 주세요.');
      };
      socket.onerror = () => {
        if (capture.active && sequence === capture.sequence) halt('관제 PC에 연결할 수 없습니다. 같은 Wi-Fi인지, PC 서버가 켜져 있는지 확인해 주세요.');
      };
    } catch (problem) {
      if (sequence !== capture.sequence) return;
      const explanations = { NotAllowedError: '카메라 권한이 차단되었습니다. 브라우저 설정에서 카메라를 허용하고 다시 연결해 주세요.', NotFoundError: '사용 가능한 카메라를 찾지 못했습니다.', NotReadableError: '다른 앱이 카메라를 사용 중일 수 있습니다. 해당 앱을 종료하고 다시 연결해 주세요.' };
      halt(explanations[problem.name] || '카메라를 시작하지 못했습니다. 브라우저와 카메라 권한을 확인해 주세요.');
    }
  };
  return <main className="phone-page"><header className="phone-header"><span className="brand-icon"><Aperture size={24} /></span><strong>CluLLM <span>카메라 연결</span></strong></header><div className="phone-intro"><span className="eyebrow">MOBILE CAMERA</span><h1>이 휴대폰을<br />관제 카메라로 연결합니다.</h1><p>후면 카메라를 사용합니다. 음성은 전송하지 않습니다.</p></div><div className="phone-view"><video ref={videoRef} muted autoPlay playsInline aria-label="휴대폰 카메라 미리보기" />{state !== 'live' && state !== 'connecting' && <div className="phone-view-placeholder"><Camera size={40} strokeWidth={1.2} /><span>카메라 연결 대기</span></div>}{state === 'live' && <span className="phone-live"><i className="status-dot live" /> 전송 중</span>}</div><div className={`phone-message ${state === 'live' ? 'connected' : ''}`} role="status">{state === 'live' ? <Radio size={19} /> : <Info size={19} />}<p>{message}</p></div><div className="phone-stats"><span>분석 주기 <b>{labelFor(interval)}</b></span><span>수신 확인 <b>{sent.toLocaleString()}장</b></span></div>{state === 'live' || state === 'connecting' ? <button className="button phone-stop full" onClick={() => halt('전송을 중지했습니다. 다시 연결하면 기존 구간에 이어서 수집합니다.')}><Square size={16} /> 전송 중지</button> : <button className="button primary full" disabled={!credentials} onClick={start}><Camera size={18} />{state === 'paused' ? '카메라 다시 연결' : '카메라 연결'}</button>}<p className="phone-footnote"><ShieldCheck size={14} /> 전송 중에는 휴대폰 화면을 켜 두세요.</p></main>;
}

createRoot(document.getElementById('root')).render(location.pathname === '/phone' ? <PhoneCamera /> : <App />);

