# QR 휴대폰 카메라 · React 전체 화면 관제

`start_realtime_app.bat`으로 실행합니다. 처음에는 아래 설치와 인증서 설정을 한 번 진행하세요. 기존 Streamlit 앱과 추출 모델 환경은 그대로 사용할 수 있습니다.

## 설치

PowerShell에서 프로젝트 폴더를 열고 실행합니다. 기존 Python 패키지에 영향을 주지 않도록 실시간 앱은 별도 환경을 사용합니다.

```powershell
cd D:\Capstone_CluLLM
python -m venv .venv-realtime
.\.venv-realtime\Scripts\python.exe -m pip install -r requirements-realtime.txt
npm.cmd --prefix realtime_frontend install
npm.cmd --prefix realtime_frontend run build
```

## 실제 휴대폰 연결

1. PC와 휴대폰을 같은 Wi-Fi에 연결합니다.
2. `start_realtime_app.bat`을 실행합니다. 또는 `.\.venv-realtime\Scripts\python.exe run_realtime_app.py`를 실행합니다.
3. 실행기에 표시되는 **관제 화면 주소**를 PC 브라우저에서 엽니다. 주소 끝의 접속 키는 관제 권한이므로 공유하지 마세요.
4. 최초 실행 시 생성되는 `.local-realtime/tls/ca.pem`은 PC 브라우저에서도 신뢰할 수 있어야 합니다. Windows의 본인 계정 인증서 관리에서 **신뢰할 수 있는 루트 인증 기관**에 이 CA를 등록하세요. 실행기는 시스템의 인증서나 방화벽 설정을 자동으로 바꾸지 않습니다.
5. 휴대폰은 실행기의 **휴대폰 최초 설정** 주소에 접속하여 해당 PC의 인증서를 설치합니다. iPhone은 Safari로 프로파일을 받은 뒤 **설정 → 일반 → VPN 및 기기 관리**에서 설치하고, **일반 → 정보 → 인증서 신뢰 설정**에서 CluLLM CA 신뢰를 켭니다. Android는 기기의 보안 설정에서 CA 인증서를 설치합니다. 사용 후 같은 메뉴에서 삭제할 수 있습니다.
6. 관제 화면에서 분석 주기를 선택하고 **카메라 추가**를 누릅니다. 카메라마다 서로 다른 QR이 생성됩니다.
7. 각 휴대폰으로 해당 카메라 카드의 QR을 스캔한 뒤 **카메라 연결**을 눌러 권한을 허용합니다. 한 QR에는 한 휴대폰만 연결할 수 있습니다.
8. 최대 4대까지 반복하고, PC의 **② 영상 확인**에서 자동 분할된 수신 영상과 선택한 카메라의 키프레임을 확인합니다. 다섯 번째 카메라는 추가할 수 없습니다.

휴대폰 브라우저는 신뢰할 수 있는 HTTPS 연결에서 카메라를 사용할 수 있습니다. 인증서를 설치할 수 없는 관리 대상 휴대폰은 관리자가 준비한 공인 HTTPS 주소를 사용해야 합니다. 인증서 경고 우회나 브라우저 보안 해제에 의존하지 않습니다.

방화벽이 막는 경우 Windows 방화벽에서 해당 Python 서버의 **개인 네트워크** 수신을 허용해야 합니다. 기본 포트는 HTTPS `8765`, 최초 인증서 안내 `8766`입니다. 회사 Wi-Fi의 기기 간 통신 차단, 게스트 Wi-Fi, VPN 환경에서는 같은 Wi-Fi라도 연결이 안 될 수 있습니다.

PC에 여러 네트워크가 있으면 실제 Wi-Fi IP를 지정합니다.

```powershell
.\.venv-realtime\Scripts\python.exe run_realtime_app.py --lan-ip 192.168.0.10
```

이미 신뢰되는 인증서를 갖고 있다면 `--cert 서버인증서.pem --key 개인키.pem`을 사용합니다. 직접 구성한 HTTPS 프록시에는 `--public-origin https://camera.example.com`을 지정할 수 있습니다. 프록시는 `/api`, `/ws`와 React 파일을 같은 출처에서 제공하고 WebSocket을 지원해야 합니다. 실행기는 외부 터널이나 공개 배포를 생성하지 않습니다.

## 업로드 분석 테스트

HieTaSkim이 선택한 클립에는 유사 클립 제거 후처리가 적용됩니다(실시간 분석도 동일). 같은 분석 회차의 선택 클립끼리 시작·중간·끝의 기존 시각 특징을 비교하고, 세 지점 모두 평균제곱오차가 `0.015` 미만이면 화질 통과 여부와 화질 점수를 우선해 하나를 남깁니다. 남은 클립은 시간순으로 저장하고 요약 영상을 구성합니다. 이는 논문의 계층 분할과 별도인 앱 후처리이며, 제거 후 요약 길이는 설정 비율보다 짧아질 수 있습니다. 테스트 결과에 제거 전후 개수를 표시합니다.

관제 화면 상단의 **③ 업로드 테스트**에서 영상 파일을 선택하고 **분석 실행**을 누릅니다. 카메라 연결 없이 영상 전체를 동일한 HieTaSkim 로직으로 한 번 분석한 뒤, 선택된 클립의 AI 설명과 전체 요약을 생성합니다. `.env`의 `OPENAI_API_KEY`를 사용합니다. MP4, MOV, AVI, MKV, WebM, M4V 파일을 최대 1GB까지 받습니다.

결과에서 component 수, 키프레임 수, 저화질 키프레임 수, 요약 영상과 개별 클립을 확인하고 JSON을 내려받을 수 있습니다. 결과는 `outputs/upload_test_*`에 저장됩니다. 한 번에 한 영상만 분석하며 서버를 재시작하면 테스트 탭의 작업 상태는 초기화됩니다. 저장된 결과 파일은 유지됩니다.

## 화면과 분석 동작

- 전체 브라우저 영역을 채우는 React 화면입니다. 좌측의 **① 카메라 연동 → ② 영상 확인** 흐름을 사용합니다.
- 왼쪽에서 **30초 / 1분 / 5분 / 10분 / 30분 / 1시간** 중 주기를 선택합니다. 초기값은 5분입니다.
- 왼쪽 아래 **키프레임** 버튼을 누르면 오른쪽으로 목록 패널이 펼쳐집니다. 마우스 드래그·휠 또는 터치 스와이프로 탐색하고, 썸네일을 클릭해 장면을 크게 확인합니다. 닫기 버튼이나 Esc 키로 패널을 접습니다.
- 주기는 실제 프레임이 수신되는 누적 시간을 기준으로 계산합니다. 3초 이상 수신이 끊기면 수집 시간은 멈추며, 재연결하면 이어집니다.
- 주기를 바꾸면 모아 둔 장면을 보존하고, 변경 시점부터 새 주기로 다음 분석을 예약합니다.
- **지금 분석**은 기다리지 않고 현재까지의 장면을 추출합니다. **연동 종료**는 휴대폰 전송을 닫고 남은 구간까지 추출합니다.
- 휴대폰은 카메라당 최대 약 4fps, 긴 변 720px 이하 JPEG 전송을 권장합니다. 수신 확인 후 다음 프레임을 보내는 방식이라 느린 네트워크에서 대기 영상이 쌓이지 않습니다. 일반 영상 통화처럼 30fps를 재생하는 구성은 아닙니다.
- 2~4대 연결 시에는 같은 5GHz Wi-Fi 또는 유선 PC 연결을 권장합니다. 프레임 검증·디코딩·녹화는 카메라 수신 전용 스레드 풀에서 처리하고, 카메라별 영상 요약은 별도 프로세스에서 병렬 실행합니다. 요약 문장 생성도 별도 스레드에서 실행되므로 영상 미리보기와 수신은 분석 중에도 계속됩니다. GPU 메모리가 부족하면 `CLULLM_ANALYSIS_PROCESSES=1` 또는 `2`로 동시 분석 프로세스 수를 낮추세요(기본 최대 4).
- 프레임 변화량은 수신 즉시 계산됩니다. 저장 영상의 주기별 분석은 아래 HieTaSkim 방식으로 클립과 대표 프레임을 선택합니다. 수신 중 후보 버퍼는 한 세션에서 최대 360개만 보관합니다.
- **이벤트 체크 및 위험 태그 기능은 이 실시간 경로에 연결하지 않았습니다.** VLM은 시간대별 행동을 서술하고 LLM은 전체 영상을 요약하지만, 변화량 수치를 위험도로 사용하지 않습니다.
- 휴대폰 화면을 켜 두세요. 화면 잠금 또는 앱 전환 시 전송을 중지하며, 다시 연결 버튼으로 이어갑니다. 음성은 수집하지 않습니다.
- 서버가 정상 종료되면 남은 장면도 저장합니다. 강제 종료·전원 차단 시 아직 메모리에 있는 구간은 저장되지 않을 수 있습니다.

## 결과와 기존 기능

완료 결과는 `outputs/phone_날짜_세션/`에 저장합니다. `recordings/`에는 실시간 수신 영상, `analysis/summary_videos/`에는 HieTaSkim keyshot을 연결한 요약 영상, `analysis/keyframes/`에는 대표 키프레임, `analysis/clips/`에는 연속 클립이 생성됩니다. `analysis.json`에는 시간대별 행동 관찰과 AI 영상 요약이 기록됩니다. 관제 화면의 **영상 분석** 패널에서 생성된 결과를 함께 확인할 수 있습니다.

저장 영상의 클립 생성은 **2fps 샘플링 → ImageNet ResNet50 특징 → 시간 제한 cosine 거리 그래프 → Kruskal MST → watershed saliency 재가중 → 적응형 cut → 중앙 keyshot** 순서입니다. [저자 논문의 상세 설명](https://bib.pucminas.br/teses/Informatica_LeonardoVilelaCardoso_32039_TextoCompleto.pdf)의 4.1, 4.3.1, 5.3.2절과 [공개 모델 코드](https://github.com/IMScience-PPGINF-PucMinas/HieTaSumm-lib/blob/main/HieTaSumm/Models.py)를 기준으로 구현했습니다. 공개 라이브러리의 별도 cut-number 휴리스틱 대신 논문의 식 `w(e) >= mean(w) + gamma * std(w)`를 현재 component에 적용합니다. edge 처리 순서는 재가중된 saliency 기준입니다.

프로젝트 `.env` 또는 환경변수로 `HIETASKIM_DELTA_T=4.0`(초, 논문의 2fps 기준 8프레임), `HIETASKIM_GAMMA=0.75`, `HIETASKIM_MIN_COMPONENTS=3`, `HIETASKIM_SUMMARY_RATIO=0.15`, `HIETASKIM_HIERARCHY=area`를 조절합니다. hierarchy는 `area`, `dynamics`, `volume`, `number_of_parents`를 지원합니다. 실행 환경에 `requirements-realtime.txt`를 설치해야 하며, 첫 분석 때 ResNet50 가중치를 내려받아 `.local-realtime/hietaskim/`에 캐시합니다. KERAS_HOME이 이미 설정되어 있으면 해당 경로를 사용합니다.

각 keyshot은 원본 프레임 수 `N`에 대해 `floor(N*p/NC)` 이내로 선택합니다. 기존 6초 제한과 분석 배치 간 클립 병합은 적용하지 않습니다. 논문에서 정하지 않은 퇴화 입력은 다음처럼 처리합니다: 동일 가중치 MST는 시간상 가까운 edge 우선, saliency 0인 watershed 바닥은 최소 component 수 충족 후 분할 중단, 비연속 component는 시간순 중앙 프레임이 속한 연속 구간에서 keyshot 하나를 선택합니다. 실시간 수신 간격이 불규칙하면 재생 길이에도 p 제한을 적용합니다. 영상이 너무 짧아 keyshot당 한 프레임도 배정할 수 없으면 요약 영상은 생성하지 않습니다. 기존 저화질 진단 결과는 유지하되 요약에 추가하지 않습니다.

기존 업로드 영상 분석, LLaVA/Llama, LLMVS 체크포인트, CluFrame, 감시 폴더, 과거 결과 조회는 기존 명령 `python run_web_app.py`로 계속 사용할 수 있습니다. 실시간 앱의 주기별 추출과 기존 고비용 LLM 분석은 별도 경로입니다. 서버 재시작 후 과거 실시간 결과도 기존 Streamlit의 결과 선택에서 조회할 수 있습니다.

QR은 24시간 유효하고 세션 종료 시 무효화됩니다. 관제 화면에는 최대 네 대의 독립 카메라 세션을 연결할 수 있으며, 각 QR과 세션은 카메라별로 다릅니다. 한 카메라를 종료하면 그 슬롯은 새 QR에 바로 재사용됩니다. `.local-realtime/`의 개인키는 공유하거나 소스 저장소에 올리지 마세요.

## PC에서만 화면 확인

```powershell
.\.venv-realtime\Scripts\python.exe run_realtime_app.py --dev-http
```

실행기에 표시된 `http://localhost:8765/#admin=...` 주소를 사용합니다. 이 모드는 PC 화면 개발용이며 실제 휴대폰에는 기본 HTTPS 실행을 사용해야 합니다.

```powershell
.\.venv-realtime\Scripts\python.exe -m unittest discover -s tests -p "test_realtime*.py"
npm.cmd --prefix realtime_frontend run build
```

