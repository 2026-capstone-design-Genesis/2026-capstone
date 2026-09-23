"""연결 권한, 주기 예약, 프레임 전송, 결과 저장의 통합 검증."""
import asyncio
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from realtime_server import Candidate, Session, create_app, decode_frame, MAX_CANDIDATES
from run_realtime_app import prepare_tls


def jpeg(color=(80, 120, 200)):
    content = io.BytesIO()
    Image.new("RGB", (320, 180), color).save(content, "JPEG")
    return content.getvalue()


class RealtimeAPI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.token = "test-only-controller-" + "x" * 32
        self.app = create_app(admin_token=self.token, public_origin="https://camera.example.test", output_root=Path(self.temp.name))
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def login(self):
        response = self.client.post("/api/login", json={"token": self.token})
        self.assertEqual(response.status_code, 200)

    def test_disk_history_without_live_session(self):
        folder = Path(self.temp.name) / 'saved-run' / 'history'
        folder.mkdir(parents=True)
        (folder / 'batch_0001.json').write_text(json.dumps({
            'batch': {'id': 1, 'completed_at': '2026-09-12'},
            'observations': [{'batch_id': 1, 'clip_url': '/api/sessions/old/media/analysis/clips/a.webm'}],
            'report': {'ai_summary': 'saved'},
        }), encoding='utf-8')
        self.assertEqual(self.client.get('/api/history').status_code, 401)
        self.login()
        rows = self.client.get('/api/history').json()
        self.assertEqual(len(rows), 1)
        data = self.client.get('/api/history/saved-run/batch_0001.json').json()
        self.assertEqual(data['observations'][0]['clip_url'], '/api/history-media/saved-run/analysis/clips/a.webm')
        self.assertEqual(self.client.get('/api/history-media/saved-run/history/batch_0001.json').status_code, 404)

    def new_session(self):
        self.login()
        response = self.client.post("/api/sessions", json={"interval_seconds": 300})
        self.assertEqual(response.status_code, 201)
        return response.json()

    def test_controller_and_phone_permissions(self):
        self.assertEqual(self.client.get("/api/config").status_code, 401)
        self.assertEqual(self.client.post("/api/login", json={"token": "bad-key-" * 5}).status_code, 401)
        data = self.new_session()
        self.assertIn("https://camera.example.test/phone#session=", data["phone_url"])
        self.assertNotIn("phone_token", self.client.get("/api/config").text)
        self.assertEqual(self.client.patch(f"/api/sessions/{data['id']}/interval", json={"interval_seconds": 15}).status_code, 422)
        second = self.client.post("/api/sessions", json={"interval_seconds": 300, "label": "입구"})
        self.assertEqual(second.status_code, 201)
        self.assertEqual(second.json()["label"], "입구")
        self.assertEqual(second.json()["slot"], 2)
        self.assertEqual(self.client.get("/api/config", headers={"Origin": "https://another.example"}).status_code, 401)
        with self.client.websocket_connect(f"/ws/phone/{data['id']}") as phone:
            phone.send_json({"token": "invalid"})
            with self.assertRaises(WebSocketDisconnect):
                phone.receive_json()

    def test_qr_phone_to_viewer_analysis_and_stop(self):
        data = self.new_session()
        identifier = data["id"]
        session = self.app.state.sessions[identifier]
        with self.client.websocket_connect(f"/ws/phone/{identifier}") as phone:
            phone.send_json({"token": session.phone_token})
            self.assertEqual(phone.receive_json()["type"], "ready")
            phone.send_bytes(jpeg())
            self.assertEqual(phone.receive_json()["type"], "ack")
            with self.client.websocket_connect(f"/ws/viewer/{identifier}") as viewer:
                status = viewer.receive_json()
                self.assertEqual(status["status"], "live")
                self.assertEqual(status["received_frames"], 1)
                self.assertEqual(viewer.receive_bytes(), jpeg())
                # 종료를 보내고 다음 상태를 읽어 관제 송신 루프가 닫힘을 관찰하게 한다.
                self.assertEqual(self.client.post(f"/api/sessions/{identifier}/stop").status_code, 200)
                self.assertEqual(viewer.receive_json()["status"], "ended")
            with self.assertRaises(WebSocketDisconnect):
                phone.receive_json()

    def test_four_active_sessions_limit_phone_isolation_and_slot_reuse(self):
        self.login()
        sessions = [self.client.post("/api/sessions", json={"interval_seconds": 300}).json() for _ in range(4)]
        self.assertEqual([item["slot"] for item in sessions], [1, 2, 3, 4])
        self.assertEqual([item["label"] for item in sessions], ["카메라 01", "카메라 02", "카메라 03", "카메라 04"])
        self.assertEqual(self.client.post("/api/sessions", json={"interval_seconds": 300}).status_code, 429)
        first, second = sessions[:2]
        with self.client.websocket_connect(f"/ws/phone/{first['id']}") as phone_one, self.client.websocket_connect(f"/ws/phone/{second['id']}") as phone_two:
            phone_one.send_json({"token": self.app.state.sessions[first['id']].phone_token})
            phone_two.send_json({"token": self.app.state.sessions[second['id']].phone_token})
            self.assertEqual(phone_one.receive_json()["type"], "ready")
            self.assertEqual(phone_two.receive_json()["type"], "ready")
            phone_one.send_bytes(jpeg((10, 20, 30)))
            phone_two.send_bytes(jpeg((220, 20, 30)))
            self.assertEqual(phone_one.receive_json()["type"], "ack")
            self.assertEqual(phone_two.receive_json()["type"], "ack")
            self.assertEqual(self.app.state.sessions[first['id']].received, 1)
            self.assertEqual(self.app.state.sessions[second['id']].received, 1)
        self.assertEqual(self.client.post(f"/api/sessions/{first['id']}/stop").status_code, 200)
        reused = self.client.post("/api/sessions", json={"interval_seconds": 300}).json()
        self.assertEqual(reused["slot"], 1)
        status = self.client.get(f"/api/sessions/{identifier}").json()
        self.assertEqual(status["total_keyframes"], 1)
        self.assertEqual(status["batches"][0]["reason"], "stop")
        self.assertEqual(status["keyframes"][0]["status"], "unreviewed")
        self.assertEqual(self.client.get(status["keyframes"][0]["image_url"]).status_code, 200)
        response = self.client.get(f"/api/sessions/{identifier}/report")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertTrue((session.output_dir / "metadata.json").exists())
        self.assertEqual(self.client.patch(f"/api/sessions/{identifier}/interval", json={"interval_seconds": 600}).status_code, 422)
        with self.client.websocket_connect(f"/ws/phone/{identifier}") as phone:
            phone.send_json({"token": session.phone_token})
            with self.assertRaises(WebSocketDisconnect):
                phone.receive_json()

    def test_period_change_retains_candidates_and_resets_deadline(self):
        data = self.new_session()
        identifier = data["id"]
        session = self.app.state.sessions[identifier]
        session.elapsed = 27
        for interval in (30, 60, 300, 600, 1800, 3600):
            response = self.client.patch(f"/api/sessions/{identifier}/interval", json={"interval_seconds": interval})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(session.next_at, 27 + interval)
        self.assertEqual(self.client.post(f"/api/sessions/{identifier}/analyze").status_code, 409)


class RealtimeAnalysis(unittest.IsolatedAsyncioTestCase):
    async def test_session_analyses_start_in_parallel_without_blocking_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Session(300, Path(directory))
            second = Session(300, Path(directory))
            for session in (first, second):
                session.candidates = [Candidate(0, "now", jpeg(), np.zeros(1), 0)]
                self.assertTrue(session.start_analysis("manual"))
                self.assertEqual(session.snapshot()["analysis_status"], "running")
            self.assertIsNot(first.task, second.task)
            await asyncio.gather(first.task, second.task)

    async def test_live_frames_can_outpace_archival_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(300, Path(directory))
            session.next_at = float('inf')
            for index in range(4):
                await session.receive(jpeg((80 + index, 120, 200)))
            self.assertEqual(session.received, 4)
            self.assertEqual(session.frame_version, 4)
            self.assertEqual(session.recorder.frames, 1)
            await session.finish()

    async def test_short_intervals_trigger_at_deadline(self):
        for interval in (30, 60):
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as directory:
                session = Session(interval, Path(directory))
                await session.receive(jpeg())
                self.assertFalse(session.busy)
                # 실제 대기 대신 수신 누적 시간을 마감 직전으로 이동한다.
                session.elapsed = interval - .1
                session.last_frame_mono = time.monotonic() - .2
                await session.receive(jpeg((210, 120, 40)))
                self.assertTrue(session.busy)
                await session.task
                self.assertEqual(len(session.batches), 1)
                self.assertEqual(session.batches[0]["interval_seconds"], interval)
                self.assertEqual(session.batches[0]["reason"], "scheduled")
                self.assertGreater(session.next_at, interval * 2)
                await session.finish()

    async def test_real_interval_trigger_and_disconnection_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(300, Path(directory))
            await session.receive(jpeg())
            session.elapsed = 299.9
            session.last_frame_mono = time.monotonic() - .2
            await session.receive(jpeg((210, 120, 40)))
            self.assertTrue(session.busy)
            await session.task
            self.assertEqual(len(session.batches), 1)
            self.assertEqual(session.batches[0]["reason"], "scheduled")
            self.assertGreaterEqual(session.elapsed, 300)
            self.assertGreater(session.next_at, 600)
            before = session.elapsed
            session.last_frame_mono = time.monotonic() - 20
            await session.receive(jpeg())
            self.assertEqual(session.elapsed, before)
            self.assertEqual(session.motion, 0)
            await session.finish()
            self.assertEqual(len(session.batches), 2)
            saved = json.loads((session.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved), len(session.summary))
            self.assertTrue(all(Path(item["selected_frame_copy"]).exists() for item in saved))

    async def test_candidate_memory_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(3600, Path(directory))
            session.next_at = float("inf")
            for index in range(MAX_CANDIDATES + 5):
                session.elapsed = index * 10
                if session.last_recorded_mono is not None:
                    session.last_recorded_mono -= 1 / 4
                await session.receive(jpeg())
            self.assertEqual(len(session.candidates), MAX_CANDIDATES)
            await session.finish()
            self.assertLessEqual(len(session.summary), 8)

    async def test_invalid_payload_and_image_size(self):
        for payload in (b"not a jpeg", b"x" * 400_001):
            with self.assertRaises(ValueError):
                decode_frame(payload)
        output = io.BytesIO()
        Image.new("RGB", (1400, 100)).save(output, "JPEG")
        with self.assertRaises(ValueError):
            decode_frame(output.getvalue())


class LocalHTTPS(unittest.TestCase):
    def test_certificate_san_and_reusable_ca(self):
        from cryptography import x509
        with tempfile.TemporaryDirectory() as directory:
            certificate, key, root = prepare_tls("192.168.0.10", Path(directory))
            leaf = x509.load_pem_x509_certificate(certificate.read_bytes())
            addresses = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            self.assertIn("localhost", addresses.get_values_for_type(x509.DNSName))
            self.assertIn("192.168.0.10", [str(item) for item in addresses.get_values_for_type(x509.IPAddress)])
            self.assertTrue(key.exists())
            self.assertEqual(prepare_tls("192.168.0.11", Path(directory))[2], root)


if __name__ == "__main__":
    unittest.main()
