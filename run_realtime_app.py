"""React 실시간 관제 앱 실행기. 기본값은 같은 Wi-Fi용 로컬 HTTPS다."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
from pathlib import Path
import plistlib
import secrets
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parent


class LocalSetupHTTPServer(ThreadingHTTPServer):
    """Avoid reverse-DNS lookup while opening the local certificate setup page."""

    def server_bind(self):
        self.socket.bind(self.server_address)
        self.server_name = self.server_address[0]
        self.server_port = self.socket.getsockname()[1]


def project_env() -> dict[str, str]:
    values = {}
    for path in (ROOT / '.env', ROOT.parent / '.env'):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding='utf-8-sig').splitlines():
            line = raw.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                values.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return values


def start_caption_worker():
    env = project_env()
    if env.get('CAPTION_BACKEND', 'openai').lower() != 'livecc':
        return None, None
    from livecc_caption import REVISION
    python = ROOT.parent / '.venv-livecc' / 'Scripts' / 'python.exe'
    model = Path(env.get('LIVECC_MODEL_PATH') or ROOT / '.livecc' / 'model' / REVISION)
    if not model.is_absolute():
        model = ROOT / model
    if not python.is_file() or not model.is_dir():
        raise RuntimeError('LiveCC 전용 환경 또는 고정 리비전 모델이 없습니다. LIVECC_SETUP.md를 확인해 주세요.')
    state = ROOT / '.livecc'
    state.mkdir(exist_ok=True)
    log = (state / 'worker.log').open('a', encoding='utf-8')
    creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
    process = subprocess.Popen([str(python), '-u', str(ROOT / 'livecc_jobs.py'), '--model', str(model)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=creation_flags)
    time.sleep(.8)
    if process.poll() is not None:
        from livecc_jobs import get_worker_status
        if get_worker_status().get('online'):
            log.close()
            return None, None
        log.close()
        raise RuntimeError('LiveCC GPU worker를 시작하지 못했습니다. .livecc/worker.log를 확인해 주세요.')
    return process, log


def local_ip() -> str:
    """실제 데이터 전송 없이 기본 라우팅에 선택되는 로컬 IP를 구한다."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("192.0.2.1", 9))
            return connection.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def prepare_tls(host_ip: str, directory: Path) -> tuple[Path, Path, bytes]:
    """로컬 CA와 서버 인증서를 만들되 OS 신뢰 저장소는 수정하지 않는다."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    directory.mkdir(parents=True, exist_ok=True)
    ca_path, ca_key_path = directory / "ca.pem", directory / "ca-key.pem"
    now = datetime.now(timezone.utc)

    def save_key(path, key):
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        path.chmod(0o600)

    if ca_path.exists() and ca_key_path.exists():
        ca = x509.load_pem_x509_certificate(ca_path.read_bytes())
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        if ca.not_valid_after_utc <= now + timedelta(days=2):
            raise RuntimeError("로컬 인증서가 만료되었습니다. .local-realtime/tls 폴더를 별도 보관한 후 다시 실행하고 휴대폰 인증서를 갱신해 주세요.")
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CluLLM Local Camera CA")])
        ca = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(ca_key.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
              .not_valid_after(now + timedelta(days=365))
              .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
              .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                          data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
                                          encipher_only=False, decipher_only=False), critical=True)
              .sign(ca_key, hashes.SHA256()))
        ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
        save_key(ca_key_path, ca_key)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    names = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    if host_ip != "127.0.0.1":
        names.append(x509.IPAddress(ipaddress.ip_address(host_ip)))
    leaf = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CluLLM Camera")]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5)).not_valid_after(min(now + timedelta(days=90), ca.not_valid_after_utc))
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    certificate_path, key_path = directory / "server.pem", directory / "server-key.pem"
    certificate_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    save_key(key_path, key)
    return certificate_path, key_path, ca.public_bytes(serialization.Encoding.DER)


def setup_server(certificate: bytes, port: int):
    profile = plistlib.dumps({
        "PayloadContent": [{"PayloadCertificateFileName": "clullm-ca.cer", "PayloadContent": certificate,
                            "PayloadDescription": "이 PC의 CluLLM 카메라 HTTPS 연결용 인증서입니다.",
                            "PayloadDisplayName": "CluLLM 로컬 카메라 인증서", "PayloadIdentifier": "local.clullm.camera.cert",
                            "PayloadType": "com.apple.security.root", "PayloadUUID": str(uuid.uuid4()), "PayloadVersion": 1}],
        "PayloadDisplayName": "CluLLM 로컬 카메라", "PayloadIdentifier": "local.clullm.camera",
        "PayloadType": "Configuration", "PayloadUUID": str(uuid.uuid4()), "PayloadVersion": 1,
        "PayloadDescription": "사용하는 PC의 주소가 맞는지 확인한 다음 설치하세요. 사용 후 삭제할 수 있습니다.",
    })
    page = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>CluLLM 휴대폰 최초 설정</title><style>body{font-family:system-ui;max-width:620px;margin:32px auto;padding:20px;line-height:1.85;background:#10141d;color:#eaf0f8}a{color:#b8a5ff}h1{font-size:24px}p{color:#aebacd}</style>
    <h1>CluLLM 휴대폰 최초 설정</h1><p>같은 Wi-Fi의 본인 PC인지 주소를 확인하세요. 이 인증서는 이 PC의 카메라 페이지에 HTTPS로 연결하기 위한 것입니다.</p>
    <h2>iPhone / iPad</h2><ol><li>Safari에서 <a href="/ios.mobileconfig">로컬 카메라 인증서 프로파일</a>을 다운로드합니다.</li><li>설정 → 일반 → VPN 및 기기 관리에서 프로파일을 설치합니다.</li><li>설정 → 일반 → 정보 → 인증서 신뢰 설정에서 CluLLM Local Camera CA를 신뢰하도록 설정합니다.</li><li>PC 화면의 QR을 다시 스캔하고 ‘카메라 연결’을 누릅니다.</li></ol>
    <h2>Android</h2><ol><li><a href="/ca.cer">로컬 카메라 CA 인증서</a>를 다운로드합니다.</li><li>설정의 ‘인증서 설치’ 또는 ‘CA 인증서’를 검색해 해당 인증서를 설치합니다. 기기별 메뉴가 다릅니다.</li><li>Chrome을 다시 열고 PC 화면의 QR을 스캔합니다.</li></ol><p>사용 후에는 기기의 인증서/프로파일 설정에서 CluLLM 인증서를 삭제할 수 있습니다. 다른 서비스의 인증서 경고는 무시하지 마세요.</p>
    <p>인증서 설치가 제한된 기기는 관리자가 제공한 공인 HTTPS 주소로 접속해야 합니다.</p></html>""".encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            routes = {"/": (page, "text/html; charset=utf-8"), "/ca.cer": (certificate, "application/pkix-cert"),
                      "/ios.mobileconfig": (profile, "application/x-apple-aspen-config")}
            content = routes.get(urlsplit(self.path).path)
            if not content:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content[1])
            self.send_header("Content-Length", str(len(content[0])))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content[0])

        def log_message(self, format, *args):
            pass

    server = LocalSetupHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser(description="QR 카메라 기반 React 실시간 관제 서버")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lan-ip", help="휴대폰과 같은 Wi-Fi에 연결된 PC의 IPv4 주소")
    parser.add_argument("--public-origin", help="외부 HTTPS 프록시를 직접 구성한 경우에만 해당 주소 지정")
    parser.add_argument("--cert", type=Path, help="직접 준비한 HTTPS 인증서 PEM")
    parser.add_argument("--key", type=Path, help="직접 준비한 HTTPS 개인 키 PEM")
    parser.add_argument("--dev-http", action="store_true", help="PC에서 화면 검증용. 로컬 호스트 HTTP만 사용")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65534:
        parser.error("포트는 1024~65534 범위로 지정해 주세요.")
    if bool(args.cert) != bool(args.key):
        parser.error("--cert와 --key를 함께 지정해 주세요.")
    if args.public_origin:
        parsed = urlsplit(args.public_origin)
        if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment or parsed.username:
            parser.error("공개 주소는 경로가 없는 HTTPS 주소여야 합니다.")
    if not (ROOT / "realtime_frontend" / "dist" / "index.html").exists():
        parser.error("먼저 npm.cmd --prefix realtime_frontend install 및 npm.cmd --prefix realtime_frontend run build를 실행해 주세요.")
    import uvicorn
    from realtime_server import create_app

    ip = args.lan_ip or local_ip()
    ipaddress.ip_address(ip)
    certificate, key, setup = None, None, None
    if not args.dev_http:
        if args.cert:
            certificate, key = args.cert, args.key
        else:
            certificate, key, ca = prepare_tls(ip, ROOT / ".local-realtime" / "tls")
            setup = setup_server(ca, args.port + 1)
    protocol = "http" if args.dev_http else "https"
    origin = args.public_origin or f"{protocol}://{'localhost' if args.dev_http else ip}:{args.port}"
    token = secrets.token_urlsafe(32)
    app = create_app(admin_token=token, public_origin=origin)
    worker = worker_log = None
    try:
        worker, worker_log = start_caption_worker()
        print("\nCluLLM 실시간 영상 관제")
        print(f"관제 화면: {protocol}://localhost:{args.port}/#admin={token}")
        if setup:
            print(f"휴대폰 최초 설정: http://{ip}:{args.port + 1}/")
            print("로컬 인증서는 휴대폰에서 한 번 신뢰 설정해야 카메라 권한을 사용할 수 있습니다.")
            print("PC도 .local-realtime/tls/ca.pem을 신뢰한 브라우저로 접속하세요.")
        if args.dev_http:
            print("PC 개발 모드입니다. 휴대폰 연결에는 기본 HTTPS 실행을 사용하세요.")
        print("종료: Ctrl+C. 수신 영상과 요약 영상, 키프레임, 쇼츠 및 AI 분석 결과를 저장합니다.\n")
        uvicorn.run(app, host="127.0.0.1" if args.dev_http else "0.0.0.0", port=args.port,
                    ssl_certfile=str(certificate) if certificate else None, ssl_keyfile=str(key) if key else None,
                    ws_max_size=MAX_WS_SIZE, proxy_headers=False, access_log=False)
    finally:
        if worker and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
        if worker_log:
            worker_log.close()
        if setup:
            setup.shutdown()
            setup.server_close()


MAX_WS_SIZE = 420_000

if __name__ == "__main__":
    main()
