"""Offline clip QA application, not LiveCC streaming reproduction."""
from pathlib import Path
import importlib.metadata
import functools
import json
import re
import threading
import time
import hashlib

import numpy as np

MODEL_ID = 'chenjoya/LiveCC-7B-Instruct'
REVISION = 'dbe7415c5d2024c3e2d7f898ecbc58fd9fc8da2f'
PROMPT = ('제공된 영상 구간의 프레임을 시간순으로 살펴보고, 직접 확인되는 핵심 행동이나 상태 변화를 한국어 한 문장으로 설명하세요. '
          '주체와 핵심 행동을 포함하고, 필요한 경우 대상이나 관찰된 변화를 덧붙이세요. '
          '인물의 신원, 의도, 감정, 보이지 않는 원인이나 이후 결과를 추측하지 마세요. '
          '뚜렷한 행동이 없으면 관찰되는 상태를 설명하세요. 설명을 위한 서문, 목록, 타임스탬프 없이 한 문장만 출력하세요. '
          '불명확한 화면은 확인 가능한 수준으로 표현하고, 진행 중인 행동을 완료된 결과로 표현하지 마세요. '
          '프레임의 순서만 사용하며 동작 속도나 정확한 소요 시간은 추정하지 마세요.')


def select_indices(times, keyframe, mode='centered', seconds=2., count=4, ended=True, available_until=None):
    if mode not in ('centered', 'past_only') or seconds <= 0 or count < 1 or not np.isfinite(keyframe):
        raise ValueError('Invalid clip settings')
    times = np.asarray(times, dtype=float)
    if not len(times):
        return {'status': 'no_frames'}, []
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError('Timestamps must be finite and ordered')
    wanted_end = keyframe + seconds / 2 if mode == 'centered' else keyframe
    wanted_start = keyframe - seconds / 2 if mode == 'centered' else keyframe - seconds
    available = times[-1] if available_until is None else available_until
    if keyframe < times[0] or keyframe > available:
        return {'status': 'invalid_input'}, []
    if not ended and wanted_end > available:
        return {'status': 'waiting', 'required_until_sec': wanted_end}, []
    start, end = max(times[0], wanted_start), min(available, wanted_end)
    valid = np.flatnonzero((times >= start) & (times <= end))
    valid = np.array(list(dict((float(times[i]), int(i)) for i in valid).values()), dtype=int)
    if not len(valid):
        return {'status': 'no_frames'}, []
    nearest = int(valid[np.argmin(abs(times[valid] - keyframe))])
    # Reserve the keyframe slot first, then maximize temporal coverage.
    targets = np.linspace(start, end, count)
    chosen = {nearest}
    for target in sorted(targets, key=lambda x: abs(x - times[nearest]), reverse=True):
        if len(chosen) >= min(count, len(valid)):
            break
        candidates = [int(i) for i in valid if int(i) not in chosen]
        chosen.add(min(candidates, key=lambda i: abs(times[i] - target)))
    indices = sorted(chosen)
    return {'status': 'sampled', 'clip_start_sec': float(start), 'clip_end_sec': float(end),
            'clip_duration_sec': float(end - start), 'frame_times_sec': times[indices].tolist(),
            'observed_frame_count': len(indices), 'source_frame_indices': indices}, indices


def decode_clip(path, keyframe, **settings):
    import av
    times = []
    pts = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Missing frame PTS')
            times.append(float(frame.pts * stream.time_base))
            pts.append(frame.pts)
    sidecar = Path(path).with_suffix('.timestamps.json')
    if sidecar.exists():
        recorded = json.loads(sidecar.read_text(encoding='utf-8'))
        if len(recorded) != len(times):
            raise ValueError('Frame timestamp sidecar mismatch')
        times = recorded
        duration = times[-1] if times else 0
    settings.setdefault('available_until', duration)
    result, indices = select_indices(times, keyframe, **settings)
    frames = []
    wanted = set(indices)
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                frames.append(frame.to_ndarray(format='rgb24'))
    result['decoded_pts'] = [pts[i] for i in indices]
    return result, frames


def decode_session(paths, keyframe, **settings):
    """Join finalized files by capture timestamps, without filling gaps."""
    import av
    entries = []
    for path in paths:
        path = Path(path)
        sidecar = path.with_suffix('.timestamps.json')
        if not sidecar.exists():
            continue
        capture_times = json.loads(sidecar.read_text(encoding='utf-8'))
        with av.open(str(path)) as container:
            decoded_count = 0
            for index, frame in enumerate(container.decode(video=0)):
                if frame.pts is None or index >= len(capture_times):
                    raise ValueError('Invalid capture timestamp mapping')
                entries.append({'file_id': path.name, 'path': str(path), 'frame_index': index,
                    'pts': frame.pts, 'time_base': str(frame.time_base),
                    'file_time_sec': float(frame.pts * frame.time_base),
                    'session_time_sec': capture_times[index]})
                decoded_count += 1
            if decoded_count != len(capture_times):
                raise ValueError('Capture timestamp count mismatch')
    entries.sort(key=lambda item: item['session_time_sec'])
    result, indices = select_indices([item['session_time_sec'] for item in entries], keyframe, **settings)
    selected = [entries[index] for index in indices]
    decoded = {}
    for path in set(item['path'] for item in selected):
        wanted = {item['frame_index'] for item in selected if item['path'] == path}
        with av.open(path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    decoded[(path, index)] = frame.to_ndarray(format='rgb24')
                if wanted and index >= max(wanted):
                    break
    result['frame_mapping'] = selected
    return result, [decoded[(item['path'], item['frame_index'])] for item in selected]


def valid_caption(text):
    return bool(re.search('[가-힣]', text)) and '\n' not in text and not re.match(r'^(?:[-*]|\d+[.)])', text) and len(re.split(r'[.!?。！？]+\s*', text.rstrip('.!?。！？'))) == 1


class LiveCCCaptioner:
    def __init__(self, model_path, max_pixels=100 * 28 * 28, profile='bf16'):
        self.path = Path(model_path)
        self.max_pixels = max_pixels
        self.model = self.processor = None
        self.lock = threading.Lock()
        if profile not in ('bf16', 'nf4'):
            raise ValueError('Unknown model profile')
        self.profile = profile

    def load(self):
        if self.model is not None:
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if self.profile == 'bf16' and torch.cuda.get_device_properties(0).total_memory < 18 * 1024**3:
            raise RuntimeError('BF16 GPU profile requires at least 18 GiB; this is a preflight guard, not a measured minimum')
        if self.profile == 'nf4':
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('BF16 compute unsupported; no automatic FP16 fallback')
            if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
                raise RuntimeError('NF4 preflight needs 6 GiB free; inference still requires measurement')
        if not self.path.is_dir():
            raise RuntimeError('Pinned model snapshot is not installed')
        revision_file = self.path / 'model_revision.txt'
        if self.path.name != REVISION and (not revision_file.exists() or revision_file.read_text().strip() != REVISION):
            raise RuntimeError('Model snapshot revision is not verified')
        if self.profile == 'bf16':
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
            apply_liger_kernel_to_qwen2_vl()
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        self.processor = AutoProcessor.from_pretrained(self.path, use_fast=False, local_files_only=True, max_pixels=self.max_pixels)
        options = {}
        if self.profile == 'nf4':
            options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.path, torch_dtype=torch.bfloat16,
            device_map={'': 0}, low_cpu_mem_usage=True, use_safetensors=True,
            attn_implementation='sdpa' if self.profile == 'nf4' else 'flash_attention_2', local_files_only=True, **options)
        if any(p.device.type != 'cuda' for p in self.model.parameters()):
            self.model = None
            raise RuntimeError('Unexpected CPU/disk model placement')
        if self.profile == 'bf16':
            from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation
            self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
        self.model.eval()

    def generate(self, frames, prompt):
        import torch
        conversation = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        started = time.perf_counter()
        video = np.stack(frames)
        self.last_input_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in video]
        inputs = self.processor(text=[text], videos=[video], return_tensors='pt').to(self.model.device)
        self.last_grid = inputs['video_grid_thw'].tolist()
        patch = self.processor.image_processor.patch_size
        self.last_dimensions = [[row[1] * patch, row[2] * patch] for row in self.last_grid]
        if any(h * w > self.max_pixels for h, w in self.last_dimensions):
            raise ValueError('Processor exceeded requested pixel budget')
        self.preprocess_seconds = time.perf_counter() - started
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=160, do_sample=False, repetition_penalty=1.05)
        return self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    def caption(self, path, keyframe, **settings):
        result = {'model_id': MODEL_ID, 'model_revision': REVISION, 'keyframe_time_sec': keyframe,
                  'caption': '', 'settings': settings, 'max_pixels': self.max_pixels,
                  'profile': self.profile, 'liger_enabled': self.profile == 'bf16',
                  'timing_limitation': 'Frame order supplied; nonuniform intervals are not encoded in model positions.'}
        started = time.perf_counter()
        try:
            sampled, frames = (decode_session if isinstance(path, list) else decode_clip)(path, keyframe, **settings)
            result.update(sampled)
        except ImportError as error:
            return {**result, 'status': 'dependency_missing', 'error': str(error)}
        except Exception as error:
            return {**result, 'status': 'decode_error', 'error': str(error)}
        result['decode_seconds'] = time.perf_counter() - started
        if result['status'] != 'sampled':
            return result
        observed_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        result['observed_rgb_sha256'] = observed_hashes
        with self.lock:
            import torch
            try:
                load_started = time.perf_counter()
                self.load()
            except torch.cuda.OutOfMemoryError as error:
                return {**result, 'status': 'oom', 'stage': 'model_load', 'error': str(error)}
            except Exception as error:
                return {**result, 'status': 'model_load_error', 'error': str(error)}
            result['load_seconds'] = time.perf_counter() - load_started
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            try:
                # Qwen2-VL groups frames in pairs. Padding is not an observation.
                padding = len(frames) % 2
                inputs = frames + frames[-1:] if padding else frames
                result['padding_frame_count'] = padding
                result['model_input_frame_indices'] = result['source_frame_indices'] + (result['source_frame_indices'][-1:] if padding else [])
                raw = self.generate(inputs, PROMPT)
                if self.last_input_hashes[:len(observed_hashes)] != observed_hashes:
                    raise RuntimeError('Selected RGB frames do not match the model input')
                result['raw_caption'] = raw
                if not valid_caption(raw):
                    raw = self.generate(inputs, PROMPT + '\n출력 형식을 수정하세요. 한국어 한 문장만 출력하세요.')
                    result['retry_caption'] = raw
                result.update(caption=raw, status='ok' if valid_caption(raw) else 'format_error')
            except torch.cuda.OutOfMemoryError as error:
                result.update(status='oom', error=str(error))
            except Exception as error:
                result.update(status='inference_error', error=str(error))
            result['inference_seconds'] = time.perf_counter() - started
            result['peak_vram_bytes'] = torch.cuda.max_memory_allocated()
            result['video_grid_thw'] = getattr(self, 'last_grid', None)
            result['actual_dimensions'] = getattr(self, 'last_dimensions', None)
            result['model_input_rgb_sha256'] = getattr(self, 'last_input_hashes', [])
            result['preprocess_seconds'] = getattr(self, 'preprocess_seconds', None)
            result['peak_reserved_vram_bytes'] = torch.cuda.max_memory_reserved()
            result['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'av', 'bitsandbytes' if self.profile == 'nf4' else 'liger-kernel')}
        return result
