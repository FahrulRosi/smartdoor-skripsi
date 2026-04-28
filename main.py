import sys
import cv2
import numpy as np
import random
from datetime import datetime
from enum import Enum

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from door.door_lock             import DoorLock
from liveness.head_pose         import HeadPoseEstimator
from liveness.blink             import BlinkDetector
from database.face_db           import get_all_faces

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ─── STATE MACHINE ────────────────────────────────────────────────────────────
class ValidationState(Enum):
    IDLE = 0                    # Menunggu wajah
    CHALLENGE = 1               # Menjalankan 1 tantangan acak (Liveness)
    RECOGNIZING = 2             # Memproses identitas (MobileFaceNet)
    UNMATCHED = 3               # Wajah tidak dikenal
    LOCKED = 4                  # Akses ditolak
    UNLOCKED = 5                # Pintu Terbuka

# ─── HUD HELPERS ──────────────────────────────────────────────────────────────
def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def _draw_validation_box(frame, x, y, w, h, status="", color=config.COLOR_CYAN):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if status:
        cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def _print_log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

# ─── MAIN UNLOCK PROCESS ──────────────────────────────────────────────────────
def run_unlock():
    _print_log("Memulai Smart Door Lock (Mode Challenge Acak)...", "SYSTEM")

    # Ambil data dari register (faces.pkl)
    registered_faces = get_all_faces()
    if not registered_faces:
        _print_log("DB KOSONG: Jalankan register.py terlebih dahulu!", "ERROR")

    if GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.IR_CUT_PIN, GPIO.OUT)
            GPIO.output(config.IR_CUT_PIN, GPIO.HIGH)
        except Exception as e:
            _print_log(f"Hardware Error: {e}", "ERROR")

    try:
        cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        model = MobileFaceNet()
        matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
        pose_estimator = HeadPoseEstimator()
        door = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=5) # Terbuka 5 detik
        _print_log("Sistem Siap!", "SYSTEM")
    except Exception as e:
        _print_log(f"Load Error: {e}", "ERROR")
        return

    # Variabel Kontrol
    state = ValidationState.IDLE
    current_challenge = None
    blink_checker = None
    challenge_frames = 0
    required_frames = 3 # Stabil selama 3 frame
    
    last_name = "Unknown"
    last_score = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            
            display = frame.copy()
            faces = detector.detect(frame)

            if not faces:
                # Reset jika wajah hilang
                if state != ValidationState.UNLOCKED:
                    state = ValidationState.IDLE
                door.lock()
                _put(display, "Menunggu Wajah...", 30, config.COLOR_YELLOW)
            else:
                face = faces[0]
                x_f, y_f, w_f, h_f = face.bbox
                
                # Alur 1: FaceMesh (Visualisasi Opsional agar RPi Ringan)
                # display = detector.draw(display, face) 

                if state == ValidationState.IDLE:
                    # Pilih 1 tantangan acak dari daftar: Yaw, Pitch, Roll, atau Blink
                    challenges = ["YAW", "PITCH", "ROLL", "BLINK"]
                    current_challenge = random.choice(challenges)
                    challenge_frames = 0
                    if current_challenge == "BLINK":
                        blink_checker = BlinkDetector(target_blinks=1)
                    state = ValidationState.CHALLENGE
                    _print_log(f"Tantangan terpilih: {current_challenge}")

                elif state == ValidationState.CHALLENGE:
                    # Alur 2 & 3: Liveness Aktif (Hanya 1 Tantangan agar Cepat)
                    passed = False
                    instruction = ""
                    
                    if current_challenge == "BLINK":
                        instruction = "Silakan Berkedip"
                        res = blink_checker.update(face, detector)
                        if res["complete"]: passed = True
                    else:
                        pose = pose_estimator.estimate(face, detector)
                        if pose["valid"]:
                            yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
                            if current_challenge == "YAW":
                                instruction = "Toleh Kiri/Kanan"
                                if abs(yaw) > config.CHALLENGE_YAW: challenge_frames += 1
                            elif current_challenge == "PITCH":
                                instruction = "Angkat/Tunduk Kepala"
                                if abs(pitch) > config.CHALLENGE_PITCH: challenge_frames += 1
                            elif current_challenge == "ROLL":
                                instruction = "Miringkan Kepala"
                                if abs(roll) > config.CHALLENGE_ROLL: challenge_frames += 1
                            
                            if challenge_frames >= required_frames: passed = True

                    if passed:
                        state = ValidationState.RECOGNIZING
                    else:
                        _draw_validation_box(display, x_f, y_f, w_f, h_f, "LIVENESS CHECK", config.COLOR_YELLOW)
                        _put(display, f"CHALLENGE: {instruction}", 70, config.COLOR_YELLOW)

                elif state == ValidationState.RECOGNIZING:
                    # Alur 4: MobileFaceNet & Pencocokan Database
                    _put(display, "Memverifikasi Identitas...", 70, config.COLOR_CYAN)
                    face_crop = model.crop_face(frame, face.bbox)
                    embedding = model.get_embedding(face_crop)
                    
                    # Bandingkan dengan data register
                    match = matcher.match(embedding)
                    
                    if match["matched"]:
                        last_name = match["name"]
                        last_score = match["score"]
                        state = ValidationState.UNLOCKED
                        _print_log(f"AKSES DIBERIKAN: {last_name}", "ACCESS")
                        door.unlock()
                    else:
                        state = ValidationState.UNMATCHED
                        last_name = "Tidak Dikenal"
                        _print_log("Akses Ditolak: Wajah tidak terdaftar.", "WARNING")

                # --- Visualisasi Status Akhir ---
                if state == ValidationState.UNLOCKED:
                    _draw_validation_box(display, x_f, y_f, w_f, h_f, last_name, config.COLOR_GREEN)
                    _put(display, f"Selamat Datang, {last_name}!", 70, config.COLOR_GREEN)
                elif state == ValidationState.UNMATCHED:
                    _draw_validation_box(display, x_f, y_f, w_f, h_f, "UNKNOWN", config.COLOR_RED)
                    _put(display, "Akses Ditolak!", 70, config.COLOR_RED)

            # Indikator Pintu
            p_color = config.COLOR_GREEN if not door.locked else config.COLOR_RED
            status_pintu = "TERBUKA" if not door.locked else "TERKUNCI"
            _put(display, f"PINTU: {status_pintu}", config.FRAME_HEIGHT - 20, p_color)

            cv2.imshow("Smart Door Lock - Unlock Mode", display)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    finally:
        if GPIO_AVAILABLE: GPIO.cleanup()
        cam.stop()
        detector.close()
        door.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unlock()