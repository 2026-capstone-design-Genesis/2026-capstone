# CluFrame + LLMVS 통합 파이프라인 요약

## 목표

기존 `llmvs_web_ui_fixed_cache`의 LLMVS 방식은 샘플링된 프레임/세그먼트에 대해 LLaVA caption과 Llama/LLMVS 중요도 판단을 수행한다. 이번 수정에서는 `private` 프로젝트의 key frame extraction 흐름을 앞단 전처리로 붙여서, LLM이 모든 샘플 프레임을 보지 않고 **중복이 제거된 후보 프레임**만 평가하도록 구성했다.

## 전체 흐름

1. 입력 영상 로드
2. **CluFrame 전처리**
   - 일정 FPS로 프레임 샘플링
   - Frame Differencing으로 정적/중복 프레임 제거
   - 남은 motion 후보 프레임에서 visual feature 추출
   - K-Means clustering으로 비슷한 장면을 묶음
   - 각 cluster에서 centroid에 가장 가까운 대표 후보 프레임 선택
   - 선택된 후보를 시간순으로 정렬하여 LLMVS 입력 프레임으로 변환
3. LLMVS segment 구성
4. LLaVA로 대표 프레임 caption 생성
5. Llama 또는 LLMVS checkpoint로 segment/frame 중요도 계산
6. Global context smoothing 적용
7. top-k/ratio/threshold 기준으로 최종 keyframe 선택
8. 결과 저장

## 핵심 구조

```text
Video
  ↓
CluFrame preprocessing
  ├─ Frame Differencing: 정적/중복 프레임 제거
  ├─ Feature extraction: lightweight OpenCV descriptor 또는 EfficientNetB0
  └─ K-Means representative selection
  ↓
Reduced candidate frames
  ↓
LLMVS
  ├─ LLaVA caption
  ├─ Llama / LLMVS checkpoint importance score
  ├─ Global context
  └─ summary selection
```

## 변경된 파일

### `modules/cluframe_preprocessor.py`

새로 추가한 핵심 모듈이다. `private`의 다음 로직을 LLMVS 프로젝트에 맞게 통합했다.

- `frame_differencing.py`의 움직임 프레임 탐지 아이디어
- `feature_extractor.py`의 TC feature 추출 구조를 옵션화
- `clustering.py`의 K-Means clustering
- `keyframe_extractor.py`의 cluster centroid 최근접 대표 프레임 선택

기본값은 `lightweight` feature backend다. 이 모드는 OpenCV histogram/edge/thumbnail feature를 사용해서 별도 CNN 모델 다운로드 없이 빠르게 후보를 줄인다. `efficientnet_b0`로 바꾸면 private 쪽 TC module 방향에 더 가깝게 ImageNet EfficientNetB0 특징을 쓸 수 있다.

### `pipeline.py`

기존 `[2/8] 프레임 샘플링` 단계 앞뒤를 수정했다.

- `cfg["preprocessing"]["enabled"] == true`이면 `sample_frames()` 대신 `run_cluframe_preprocessing()` 실행
- 전처리 결과를 기존 LLMVS와 동일한 `frames` dict 형식으로 반환
- `frames.json`, `metadata.json`, `cluframe_preprocessing.json`, `cluframe_features.npy`, `cluframe_candidates/` 저장
- 이후 caption/scoring/selection 단계는 기존 LLMVS 흐름 그대로 사용

### `web_app.py`

UI에 새 실행 모드를 추가했다.

- `CluFrame + LLMVS 키프레임`
- CluFrame 전처리 설정 추가
  - LLMVS에 넘길 최대 후보 프레임 수
  - feature backend: `lightweight` / `efficientnet_b0`
  - FD resize width
  - FD motion threshold

이 모드를 선택하면 `cfg["preprocessing"]["enabled"] = True`로 설정되고, LLaVA/Llama 기반 중요도 판단이 이어진다.

### `config.yaml`

기존 코드에서 참조하지만 zip 안에 없던 기본 설정 파일을 추가했다. CLI와 Web UI 모두 같은 config 구조를 사용할 수 있다.

### `PIPELINE_SUMMARY.md`

이번 통합 방식과 변경 파일을 설명하는 문서다.

## 결과 파일에서 확인할 수 있는 것

실행 결과 폴더에는 기존 LLMVS 산출물에 더해 다음 파일이 추가된다.

- `cluframe_preprocessing.json`: 샘플링 수, motion 후보 수, 최종 후보 수, 선택 timestamp 등
- `cluframe_features.npy`: 후보 프레임 feature matrix
- `cluframe_candidates/`: LLMVS에 실제로 넘어간 후보 프레임 사본
- `frames.json`: CluFrame으로 줄어든 후보 프레임 목록
- `segments.json`: 후보 프레임 기반 segment와 caption
- `scores.json`: LLMVS 중요도 점수
- `summary.json`: 최종 선택 keyframe

## 사용 방법

### Web UI

```bash
streamlit run web_app.py
```

사이드바에서 `CluFrame + LLMVS 키프레임`을 선택한 뒤 영상을 업로드하면 된다.

### CLI

`config.yaml`에서 다음처럼 설정한다.

```yaml
preprocessing:
  enabled: true
  sample_fps: 1.0
  max_candidates: 30
  feature_backend: lightweight
```

그 다음 실행한다.

```bash
python main.py --config config.yaml
```

## 주의점

- `lightweight` backend는 빠르고 안정적인 기본값이지만, 학습된 CNN 의미 특징은 아니다.
- `efficientnet_b0` backend는 private의 TC module 방향에 더 가깝지만 `torchvision`과 모델 다운로드가 필요할 수 있다.
- CluFrame 전처리는 LLMVS의 중요도 판단을 대체하는 것이 아니라, LLMVS가 볼 후보를 줄이는 전처리 단계다.
