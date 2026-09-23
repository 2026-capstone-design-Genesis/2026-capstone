# LLMVS-style Video Summary Test Module

이 프로젝트는 LLMVS 논문 구조를 테스트 가능한 형태로 축소한 모듈형 파이프라인입니다.

사용 모델:

- Caption generator: `llava-hf/llava-1.5-7b-hf`
- Local importance scorer: `meta-llama/Llama-2-13b-chat-hf`

논문 완전 재현은 아니고, 다음 흐름을 구현합니다.

```text
Video
→ 1fps frame sampling
→ 5 sec segment building
→ representative frame captioning with LLaVA-1.5-7B
→ local window scoring with Llama-2-13B-chat
→ optional Llama hidden embedding extraction
→ temporal smoothing
→ top-k summary segment selection
→ JSON / plot / HTML report export
```

## 1. 설치

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U hf_xet
pip install -r requirements.txt
```

## 2. Hugging Face Token

이 버전은 토큰을 코드나 config에 저장하지 않습니다. 프로그램 실행 시 터미널에서 직접 입력받습니다. 입력값은 `getpass`로 처리되어 화면에 표시되지 않고, 파일로 저장되지 않습니다.

```bash
python main.py --config config.yaml
```

실행 후 다음 문구가 나오면 Hugging Face token을 입력하세요. 공개 모델만 테스트하거나 이미 캐시된 모델을 쓸 경우 Enter로 넘어갈 수 있습니다.

```text
Hugging Face token 입력(없으면 Enter, 입력값은 저장되지 않음):
```

입력한 토큰은 모델 다운로드/로딩 함수에만 인자로 전달되며, `config.yaml`, `metadata.json`, `scores.json`, cache 파일에 기록되지 않습니다.

## 3. 입력 영상 배치

```text
input/sample.mp4
```

또는 `config.yaml`의 `input.video_path`를 수정합니다.

## 4. 실행

```bash
python main.py --config config.yaml
```

## 5. 결과물

```text
outputs/{run_name}/metadata.json
outputs/{run_name}/frames.json
outputs/{run_name}/segments.json
outputs/{run_name}/captions.json
outputs/{run_name}/scores_raw.json
outputs/{run_name}/scores.json
outputs/{run_name}/summary.json
outputs/{run_name}/llm_embeddings.npy
outputs/{run_name}/score_plot.png
outputs/{run_name}/report.html
outputs/{run_name}/selected_frames/
```

## 6. 주요 설정

```yaml
sampling:
  fps: 1

segment:
  segment_sec: 5

scoring:
  window_size: 7
  use_quantization_4bit: true
  extract_embeddings: true

selection:
  mode: "top_k"
  top_k: 5
```

## 7. 하드웨어 주의

LLaVA-1.5-7B와 Llama-2-13B-chat을 같이 쓰므로 GPU VRAM 사용량이 큽니다.
기본 설정은 Llama-2를 4bit quantization으로 로드합니다.
VRAM이 부족하면 다음을 시도하세요.

- 짧은 영상으로 먼저 테스트
- `sampling.fps`를 낮추기
- `segment.segment_sec`를 늘리기
- `scoring.extract_embeddings`를 `false`로 변경
- Llama-2-13B 대신 7B 또는 다른 instruct model로 교체

## 8. 모듈 구조

```text
modules/
├── video_loader.py
├── frame_sampler.py
├── segment_builder.py
├── caption_generator.py
├── llm_scorer.py
├── global_context.py
├── summary_selector.py
├── result_writer.py
└── visualizer.py
```

## 9. 한계

- LLMVS 논문처럼 self-attention global aggregator를 학습하지 않습니다.
- SumMe/TVSum benchmark 평가 코드는 포함하지 않습니다.
- Llama hidden embedding은 저장하지만, 현재 MVP에서는 학습형 aggregator 입력으로만 보존합니다.
- LLaVA caption hallucination이 있을 수 있으므로 report.html에서 caption과 대표 프레임을 같이 확인해야 합니다.

## 실행 시간 기록

프로그램은 실행 시작 시각과 종료 시각, 총 소요 시간을 자동으로 기록합니다.

터미널 출력 예시:

```text
[START] 시작 시간: 2026-05-29T18:12:03
[END] 종료 시간: 2026-05-29T18:28:41
[TIME] 총 소요 시간: 00:16:38.124 (998.124초)
```

또한 다음 파일에 실행 시간 정보가 저장됩니다. Hugging Face token 값은 저장하지 않습니다.

```text
outputs/{run_name}/runtime.json
```

저장 예시:

```json
{
  "started_at": "2026-05-29T18:12:03",
  "ended_at": "2026-05-29T18:28:41",
  "elapsed_seconds": 998.124,
  "elapsed_hms": "00:16:38.124",
  "status": "success"
}
```

---

## Web UI 사용법

이 버전에는 `web_app.py`가 추가되어 브라우저에서 영상을 업로드하거나 폴더를 감시하면서 키프레임을 추출할 수 있습니다.

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 웹 UI 실행

```bash
streamlit run web_app.py
```

브라우저가 열리면 다음 두 가지 방식 중 하나를 사용할 수 있습니다.

### 방식 A. 파일 업로드 실행

1. `파일 업로드 실행` 탭에서 영상 파일을 업로드합니다.
2. 사이드바에서 모드를 선택합니다.
   - `빠른 CV 키프레임 모드`: LLaVA/Llama 없이 프레임 변화량 기반으로 빠르게 키프레임 추출
   - `Full LLMVS 모드`: 기존 LLaVA caption + Llama importance scoring 파이프라인 실행
3. `키프레임 추출 시작` 버튼을 누릅니다.
4. 결과 이미지, score plot, ZIP 다운로드 버튼이 표시됩니다.

### 방식 B. 폴더 감시 실행

1. 프로젝트 폴더 안의 `watch_videos/` 폴더에 영상 파일을 넣습니다.
2. 웹 UI의 `폴더 감시 실행` 탭으로 이동합니다.
3. `대기 영상 한 번 처리`를 누르면 아직 처리하지 않은 파일만 처리합니다.
4. `켜두고 자동 스캔`을 켜면 브라우저 세션이 열려 있는 동안 주기적으로 새 영상을 확인합니다.

### 출력 위치

모든 결과는 `outputs/<영상이름_날짜시간>/` 아래에 저장됩니다.

주요 결과물:

- `frames/`: 샘플링된 프레임
- `selected_frames/`: 최종 선택된 키프레임
- `summary.json`: 선택된 키프레임 정보
- `scores.json`: 프레임/세그먼트 점수
- `score_plot.png`: 중요도 점수 그래프
- `report.html`: Full LLMVS 모드 실행 시 HTML 리포트

### Hugging Face token

Full LLMVS 모드에서 gated model을 사용할 경우 사이드바의 `Hugging Face token` 입력칸에 토큰을 넣으세요. 이 값은 파일에 저장하지 않고 실행 중 메모리로만 전달됩니다.

### 권장 사용

GPU가 없거나 빠르게 UI 동작만 확인하려면 먼저 `빠른 CV 키프레임 모드`로 테스트하세요. LLaVA/Llama 기반 Full LLMVS 모드는 모델 다운로드와 로딩 시간이 길고 GPU 메모리를 많이 사용합니다.

## 10. 웹 UI 요약 방식 선택

`web_app.py` 사이드바에서 다음 방식 중 하나를 선택할 수 있습니다.

1. **K-Means clustering 키프레임**
   - 샘플링된 프레임에서 색상/에지/썸네일 기반 특징을 추출합니다.
   - K-Means로 비슷한 장면을 묶고 각 클러스터 중심에 가장 가까운 프레임을 대표 키프레임으로 저장합니다.
   - LLaVA/Llama 모델을 로딩하지 않으므로 빠르게 테스트할 수 있습니다.

2. **LLaVA/Llama 모듈 키프레임**
   - LLaVA로 대표 프레임 caption을 생성합니다.
   - Llama 계열 모델로 window 기반 중요도 점수를 계산합니다.
   - 점수 기준으로 top-k, top-ratio, threshold 선택을 수행합니다.

3. **빠른 변화량 키프레임**
   - 모델 없이 프레임 간 히스토그램 변화량이 큰 장면을 선택하는 간단한 테스트 모드입니다.

---

## Hugging Face 모델 캐시 고정

웹 UI 사이드바의 **모델 캐시 설정**에서 `Hugging Face cache 위치`를 지정하고 **캐시 위치 저장/적용**을 누르면, 이후 실행에서도 같은 폴더를 계속 사용합니다.

Windows 권장 예시:

```text
D:\huggingface_cache
```

저장 후 앱은 내부적으로 다음 경로를 고정합니다.

```text
HF_HOME=D:\huggingface_cache
HF_HUB_CACHE=D:\huggingface_cache\hub
HF_DATASETS_CACHE=D:\huggingface_cache\datasets
```

효과:

- LLaVA/Llama 모델 파일을 한 위치에 저장합니다.
- 새 영상을 분석해도 기존 모델 캐시를 재사용합니다.
- 매번 30GB 모델을 다시 다운로드하는 상황을 줄입니다.
- 결과 파일(`captions.json`, `scores.json`, `selected_frames/`)은 영상별로 새로 생성됩니다.

주의:

- 캐시 고정은 다운로드 반복을 막는 용도입니다.
- 앱을 새로 켤 때 모델을 RAM/VRAM에 올리는 **로딩 시간**은 여전히 발생할 수 있습니다.
- `app_settings.json`에 캐시 경로만 저장되며 Hugging Face token은 저장하지 않습니다.
