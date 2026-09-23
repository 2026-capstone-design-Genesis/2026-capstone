@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv-realtime-local\Scripts\python.exe" (
  ".venv-realtime-local\Scripts\python.exe" run_realtime_app.py
) else if exist ".venv-realtime\Scripts\python.exe" (
  ".venv-realtime\Scripts\python.exe" run_realtime_app.py
) else (
  echo REALTIME_GUIDE.md 안내에 따라 실시간 앱 환경을 먼저 설치해 주세요.
)
pause
