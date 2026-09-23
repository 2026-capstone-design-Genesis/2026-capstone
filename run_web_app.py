from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WEB_APP = ROOT / "web_app.py"


def main() -> int:
    if not WEB_APP.exists():
        print(f"[ERROR] web_app.py를 찾을 수 없습니다: {WEB_APP}")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(WEB_APP),
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("[RUN] " + " ".join(cmd))
    print("[INFO] 종료하려면 이 창에서 Ctrl+C를 누르세요.")
    code = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if code != 0:
        print(f"[ERROR] Streamlit이 종료되었습니다. exit code={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
