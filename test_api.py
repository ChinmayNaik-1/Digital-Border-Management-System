import json
import sys
import traceback
from datetime import datetime

import requests


API_BASE = "http://localhost:5000"


def ts():
    return datetime.now().strftime("%H:%M:%S")


class BorderProtectionTester:
    def __init__(self):
        self.results = []
        self.user_id = None
        self.record_id = None
        self.face_capture_failed = False

    def _log(self, level: str, message: str):
        print(f"[{ts()}] [{level}] {message}")

    def _record(self, name: str, status: str, message: str = ""):
        self.results.append({"name": name, "status": status, "message": message})

    def _get(self, path: str):
        return requests.get(f"{API_BASE}{path}")

    def _post(self, path: str, json_body: dict):
        return requests.post(f"{API_BASE}{path}", json=json_body)

    def test_health(self):
        name = "Health Check"
        self._log("INFO", "Testing health endpoint...")
        try:
            resp = self._get("/api/health")
            if resp.status_code != 200:
                raise RuntimeError(f"Expected 200, got {resp.status_code}")
            data = resp.json()
            expected = {"status": "ok", "db": "ok", "minio": "ok"}
            missing = [k for k in expected if data.get(k) != expected[k]]
            if missing:
                raise RuntimeError(f"Health response missing/incorrect keys: {missing} ({data})")
            self._log("PASS", "✅ Health Check")
            self._record(name, "PASS")
        except Exception as e:
            self._log("FAIL", f"❌ Health Check - {e}")
            self._record(name, "FAIL", str(e))

    def test_create_user(self):
        name = "Create User"
        self._log("INFO", "Creating test user...")
        body = {
            "name": "Test User",
            "passport_number": "IN12345",
            "nationality": "Indian",
            "date_of_birth": "1990-01-01",
        }
        try:
            resp = self._post("/api/users", body)
            if resp.status_code != 201:
                raise RuntimeError(f"Expected 201, got {resp.status_code} {resp.text}")
            data = resp.json()
            user_id = data.get("user_id")
            if not user_id:
                raise RuntimeError(f"No user_id returned: {data}")
            self.user_id = user_id
            self._log("PASS", f"✅ Create User - user_id: {user_id}")
            self._record(name, "PASS", f"user_id={user_id}")
        except Exception as e:
            self._log("FAIL", f"❌ Create User - {e}")
            self._record(name, "FAIL", str(e))

    def test_get_user(self):
        name = "Get User"
        if not self.user_id:
            self._log("SKIP", "⚠️  Get User - skipped (no user_id)")
            self._record(name, "SKIP", "no user_id")
            return
        self._log("INFO", "Fetching created user...")
        try:
            resp = self._get(f"/api/users/{self.user_id}")
            if resp.status_code != 200:
                raise RuntimeError(f"Expected 200, got {resp.status_code} {resp.text}")
            data = resp.json()
            if data.get("user_id") != self.user_id:
                raise RuntimeError(f"Returned user_id mismatch: {data}")
            self._log("PASS", "✅ Get User")
            self._record(name, "PASS")
        except Exception as e:
            self._log("FAIL", f"❌ Get User - {e}")
            self._record(name, "FAIL", str(e))

    def test_capture_face(self):
        name = "Capture Face"
        if not self.user_id:
            self._log("SKIP", "⚠️  Capture Face - skipped (no user_id)")
            self._record(name, "SKIP", "no user_id")
            return
        self._log("INFO", "Capturing face (may fail on Windows without webcam)...")
        body = {"user_id": self.user_id, "location": "Border Gate"}
        try:
            resp = self._post("/api/face/capture", body)
            if resp.status_code == 201:
                data = resp.json()
                self.record_id = data.get("record_id")
                self._log("PASS", f"✅ Capture Face - record_id: {self.record_id}")
                self._record(name, "PASS", f"record_id={self.record_id}")
                return
            # Treat webcam / dev/video0 errors as warning
            text = resp.text or ""
            if resp.status_code in (400, 500) and ("webcam" in text.lower() or "video0" in text.lower() or "cannot open" in text.lower()):
                self.face_capture_failed = True
                self._log("WARN", f"⚠️  Capture Face (warning) - {resp.status_code} {resp.text.strip()}")
                self._record(name, "WARN", resp.text.strip())
                return
            raise RuntimeError(f"Expected 201, got {resp.status_code} {resp.text}")
        except requests.exceptions.ConnectionError as e:
            self._log("FAIL", f"❌ Capture Face - Connection error: {e}")
            self._record(name, "FAIL", str(e))
        except Exception as e:
            msg = str(e)
            if "video0" in msg or "webcam" in msg:
                self.face_capture_failed = True
                self._log("WARN", f"⚠️  Capture Face (warning) - {msg}")
                self._record(name, "WARN", msg)
            else:
                self._log("FAIL", f"❌ Capture Face - {msg}")
                self._record(name, "FAIL", msg)

    def test_get_face_records(self):
        name = "Get Face Records"
        if not self.user_id:
            self._log("SKIP", "⚠️  Get Face Records - skipped (no user_id)")
            self._record(name, "SKIP", "no user_id")
            return
        self._log("INFO", "Fetching face records...")
        try:
            resp = self._get(f"/api/face/records/{self.user_id}")
            if resp.status_code != 200:
                raise RuntimeError(f"Expected 200, got {resp.status_code} {resp.text}")
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Expected list response, got {type(data)}: {data}")
            self._log("PASS", "✅ Get Face Records")
            self._record(name, "PASS", f"count={len(data)}")
        except Exception as e:
            self._log("FAIL", f"❌ Get Face Records - {e}")
            self._record(name, "FAIL", str(e))

    def test_verify_face(self):
        name = "Verify Face"
        if not self.user_id:
            self._log("SKIP", "⚠️  Verify Face - skipped (no user_id)")
            self._record(name, "SKIP", "no user_id")
            return
        if self.face_capture_failed:
            self._log("WARN", "⚠️  Verify Face (skipped - no face captured)")
            self._record(name, "WARN", "skipped due to failed face capture")
            return
        self._log("INFO", "Verifying face...")
        body = {"user_id": self.user_id, "border_gate": "Delhi"}
        try:
            resp = self._post("/api/face/verify", body)
            if resp.status_code != 200:
                raise RuntimeError(f"Expected 200, got {resp.status_code} {resp.text}")
            self._log("PASS", "✅ Verify Face")
            self._record(name, "PASS")
        except Exception as e:
            self._log("FAIL", f"❌ Verify Face - {e}")
            self._record(name, "FAIL", str(e))

    def test_verification_logs(self):
        name = "Verification Logs"
        if not self.user_id:
            self._log("SKIP", "⚠️  Verification Logs - skipped (no user_id)")
            self._record(name, "SKIP", "no user_id")
            return
        self._log("INFO", "Fetching verification logs...")
        try:
            resp = self._get(f"/api/verification-logs/{self.user_id}")
            if resp.status_code != 200:
                raise RuntimeError(f"Expected 200, got {resp.status_code} {resp.text}")
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Expected list response, got {type(data)}: {data}")
            self._log("PASS", "✅ Verification Logs")
            self._record(name, "PASS", f"count={len(data)}")
        except Exception as e:
            self._log("FAIL", f"❌ Verification Logs - {e}")
            self._record(name, "FAIL", str(e))

    def run_all(self):
        print("============================================================")
        print("BORDER PROTECTION FACE RECOGNITION MVP - TEST SUITE")
        print("============================================================")
        self.test_health()
        self.test_create_user()
        self.test_get_user()
        self.test_capture_face()
        self.test_get_face_records()
        self.test_verify_face()
        self.test_verification_logs()
        print("\n============================================================")
        print("TEST SUMMARY")
        print("============================================================")
        passed = 0
        warn = 0
        failed = 0
        for r in self.results:
            status = r["status"]
            name = r["name"]
            if status == "PASS":
                print(f"✅ PASS: {name}")
                passed += 1
            elif status == "WARN":
                print(f"⚠️  WARN: {name} ({r.get('message','')})")
                warn += 1
            elif status == "SKIP":
                print(f"⚠️  SKIP: {name} ({r.get('message','')})")
                warn += 1
            else:
                print(f"❌ FAIL: {name} ({r.get('message','')})")
                failed += 1
        total = len(self.results)
        print("\nTotal: {}/{} tests passed ({} warnings, {} failures)".format(passed, total, warn, failed))


if __name__ == "__main__":
    tester = BorderProtectionTester()
    try:
        tester.run_all()
    except Exception:
        print("Unexpected error running tests:")
        traceback.print_exc()
        sys.exit(1)
