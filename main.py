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

# ─── 1. DEFINISI FUNGSI PEMBANTU (Wajib ditaruh di atas) ──────────────────────
def _print_log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6, thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def _draw_status(frame, x, y, w, h, status, color):
    # Kotak pembungkus wajah
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # Background untuk teks agar mudah dibaca
    t_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x, y - 30), (x + t_size[0] + 10, y), color, -1)
    _put(frame, status, y - 10, (255, 255, 255), x + 5, scale=0.6)

# ─── 2. STATE MACHINE ─────────────────────────────────────────────────────────
class ValidationState(Enum):
    IDLE = 0            # Menunggu wajah
    RECOGNIZING = 1     # Identifikasi nama (DB)
    CHALLENGE = 2       # Liveness (Pose + Blink)
    UNMATCHED = 3       # Wajah tidak terdaftar
    UNLOCKED = 4        # Pintu terbuka

# ─── 3. MAIN UNLOCK PROCESS ──────────────────────────────────────────────────
def run_unlock():
    _print_log("Sistem Siap (Identifikasi Dulu -> Challenge)", "SYSTEM")
    
    # Inisialisasi
    cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    model = MobileFaceNet()
    matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    pose_estimator = HeadPoseEstimator()
    door = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=5)
    
    state = ValidationState.IDLE
    last_name = ""
    challenge_sequence = []
    current_step_idx = 0
    blink_checker = None
    POSE_DIRECTIONS = ["KANAN", "KIRI", "BAWAH", "ATAS", "MIRING_KANAN", "MIRING_KIRI"]

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            display = frame.copy()
            faces = detector.detect(frame)

            if not faces:
                if state != ValidationState.UNLOCKED: 
                    state = ValidationState.IDLE
                    last_name = ""
                door.lock()
                _put(display, "Menunggu Wajah...", 40, config.COLOR_YELLOW, scale=0.8)
            else:
                face = faces[0]
                x_f, y_f, w_f, h_f = face.bbox

                # --- STATE MACHINE LOGIC ---
                
                # A. Identifikasi Nama (Dilakukan duluan)
                if state == ValidationState.IDLE:
                    state = ValidationState.RECOGNIZING

                elif state == ValidationState.RECOGNIZING:
                    face_crop = model.crop_face(frame, face.bbox)
                    embedding = model.get_embedding(face_crop)
                    match = matcher.match(embedding)
                    
                    if match["matched"]:
                        last_name = match["name"]
                        state = ValidationState.CHALLENGE
                        # Set up tantangan acak
                        pose_choice = random.choice(POSE_DIRECTIONS)
                        challenge_sequence = [pose_choice, "BLINK"]
                        current_step_idx = 0
                        blink_checker = BlinkDetector(target_blinks=1)
                    else:
                        last_name = "Wajah belum terdaftar"
                        state = ValidationState.UNMATCHED

                # B. Challenge (Hanya jika nama dikenal)
                elif state == ValidationState.CHALLENGE:
                    current_action = challenge_sequence[current_step_idx]
                    action_passed = False
                    instruction = ""
                    
                    if current_action == "BLINK":
                        instruction = "Kedipkan Mata"
                        res = blink_checker.update(face, detector)
                        if res["complete"]: action_passed = True
                    else:
                        pose = pose_estimator.estimate(face, detector)
                        if pose["valid"]:
                            yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
                            # Logika validasi arah
                            if (current_action == "KANAN" and yaw > config.CHALLENGE_YAW) or \
                               (current_action == "KIRI" and yaw < -config.CHALLENGE_YAW) or \
                               (current_action == "BAWAH" and pitch > config.CHALLENGE_PITCH) or \
                               (current_action == "ATAS" and pitch < -config.CHALLENGE_PITCH) or \
                               (current_action == "MIRING_KANAN" and roll > config.CHALLENGE_ROLL) or \
                               (current_action == "MIRING_KIRI" and roll < -config.CHALLENGE_ROLL):
                                action_passed = True
                    
                    # Tampilkan Status di layar
                    _draw_status(display, x_f, y_f, w_f, h_f, f"{last_name} | {instruction}", config.COLOR_YELLOW)
                    
                    if action_passed:
                        current_step_idx += 1
                        if current_step_idx >= len(challenge_sequence):
                            state = ValidationState.UNLOCKED
                            door.unlock()

                # C. Visual Akhir
                elif state == ValidationState.UNLOCKED:
                    _draw_status(display, x_f, y_f, w_f, h_f, f"Selamat Datang: {last_name}", config.COLOR_GREEN)
                elif state == ValidationState.UNMATCHED:
                    _draw_status(display, x_f, y_f, w_f, h_f, "Wajah belum terdaftar", config.COLOR_RED)

            # Indikator Pintu (Bawah)
            status_pintu = "TERBUKA" if not door.locked else "TERKUNCI"
            p_color = config.COLOR_GREEN if not door.locked else config.COLOR_RED
            _put(display, f"STATUS PINTU: {status_pintu}", config.FRAME_HEIGHT - 20, p_color, scale=0.5)

            cv2.imshow("Smart Door Lock", display)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cam.stop()
        door.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unlock()