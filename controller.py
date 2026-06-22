import os
import time
import threading
from datetime import datetime, timezone

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

BACKEND_URL = os.getenv("BACKEND_URL", "http://103.93.132.205:5000").rstrip("/")
DEVICE_NAME = os.getenv("DEVICE_NAME", "raspberry_pi_01")

app = FastAPI(title="Smart Door Lock Controller")

app_state = "MAIN"
current_name = ""
is_transitioning = False
state_lock = threading.Lock()


class TriggerRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)


class AccessResultRequest(BaseModel):
    namaUser: str = Field(..., min_length=1)
    waktuAkses: str = Field(..., min_length=1)
    keterangan: str = Field(..., min_length=1)
    status: str = Field(..., min_length=1)


def _post_callback_to_backend(payload: dict):
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/user/access-result",
            json=payload,
            timeout=15,
        )
        return {
            "ok": response.status_code in (200, 201),
            "status_code": response.status_code,
            "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "response": str(exc),
        }


def _simulate_registration_process(user_name: str):
    global app_state, current_name, is_transitioning

    try:
        print(f"[RPi] Mulai registrasi wajah untuk: {user_name}")
        time.sleep(5)

        payload = {
            "namaUser": user_name,
            "waktuAkses": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "keterangan": "Registrasi wajah berhasil di Raspberry Pi",
            "status": "AKSES DITERIMA",
        }

        callback_result = _post_callback_to_backend(payload)
        print(f"[RPi] Callback ke backend: {callback_result}")

    finally:
        with state_lock:
            app_state = "MAIN"
            current_name = ""
            is_transitioning = False
        print("[RPi] Kembali ke mode utama")


@app.post("/api/trigger-register")
def trigger_register(req: TriggerRegisterRequest):
    global app_state, current_name, is_transitioning

    with state_lock:
        if is_transitioning:
            return {
                "status": "error",
                "message": "Kamera sedang dalam proses perpindahan. Harap tunggu."
            }

        if app_state == "REGISTER":
            return {
                "status": "error",
                "message": "Sistem sudah dalam mode registrasi."
            }

        is_transitioning = True
        app_state = "REGISTER"
        current_name = req.name

    worker = threading.Thread(
        target=_simulate_registration_process,
        args=(req.name,),
        daemon=True,
    )
    worker.start()

    return {
        "status": "success",
        "message": f"Trigger registrasi diterima untuk: {req.name}",
        "data": {
            "device_name": DEVICE_NAME,
            "app_state": app_state,
            "name": req.name
        }
    }


@app.post("/api/access-result")
def access_result(req: AccessResultRequest):
    payload = req.model_dump()
    result = _post_callback_to_backend(payload)
    if result["ok"]:
        return {
            "status": "success",
            "message": "Hasil akses berhasil diteruskan ke backend",
            "data": result
        }

    return {
        "status": "error",
        "message": "Gagal meneruskan hasil akses ke backend",
        "data": result
    }


@app.get("/api/health")
def health():
    return {
        "status": "success",
        "message": "Raspberry Pi controller is running",
        "data": {
            "device_name": DEVICE_NAME,
            "app_state": app_state,
            "backend_url": BACKEND_URL
        }
    }


def run_controller():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    print("======================================================")
    print("  SMART DOOR LOCK CONTROLLER")
    print("======================================================")
    print(f"[RPi] Backend URL   : {BACKEND_URL}")
    print(f"[RPi] Device Name   : {DEVICE_NAME}")
    print("[RPi] API siap di port 8000")
    run_controller()