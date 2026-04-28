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
from database.face_db           import get_all_faces   # ← Penting

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


def _print_log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.7, thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_status(frame, x, y, w, h, status, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    # Background teks
    t_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.rectangle(frame, (x, y - 35), (x + t_size[0] + 15, y - 5), color, -1)
    _put(frame, status, y - 12, (255, 255, 255), x + 8, scale=0.65, thickness=2)


class ValidationState(Enum):
    IDLE        = 0
    RECOGNIZING = 1
    CHALLENGE   = 2
    UNMATCHED   = 3
    UNLOCKED    = 4


def run_unlock():
    _print_log("Sistem Smart Door Lock diaktifkan", "SYSTEM")
    
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

    # Daftar tantangan
    POSE_DIRECTIONS = ["KANAN", "KIRI", "ATAS", "BAWAH", "MIRING_KANAN", "MIRING_KIRI"]

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            display = frame.copy()
            faces = detector.detect(frame)

            if not faces:
                if state != ValidationState.UNLOCKED:
                    state = ValidationState.IDLE
                    last_name = ""
                door.lock()
                _put(display, "Menunggu Wajah...", 50, config.COLOR_YELLOW, scale=0.9)
            else:
                face = faces[0]
                x_f, y_f, w_f, h_f = face.bbox

                # ==================== STATE MACHINE ====================
                
                if state == ValidationState.IDLE:
                    state = ValidationState.RECOGNIZING

                # 1. RECOGNITION (Cocokkan wajah dengan database)
                elif state == ValidationState.RECOGNIZING:
                    face_crop = model.crop_face(frame, face.bbox)
                    embedding = model.get_embedding(face_crop)
                    
                    match = matcher.match(embedding)

                    if match.get("matched", False):
                        last_name = match["name"]
                        state = ValidationState.CHALLENGE
                        
                        # Buat challenge acak
                        pose_choice = random.choice(POSE_DIRECTIONS)
                        challenge_sequence = [pose_choice, "BLINK"]
                        current_step_idx = 0
                        blink_checker = BlinkDetector(target_blinks=1)
                        
                        _print_log(f"Wajah dikenali: {last_name} | Challenge: {challenge_sequence}", "SUCCESS")
                    else:
                        last_name = "Tidak Dikenali"
                        state = ValidationState.UNMATCHED
                        _print_log("Wajah tidak terdaftar di database", "WARNING")

                # 2. CHALLENGE (Liveness Verification)
                elif state == ValidationState.CHALLENGE:
                    current_action = challenge_sequence[current_step_idx]
                    action_passed = False
                    instruction = ""

                    if current_action == "BLINK":
                        instruction = "Kedipkan Mata Sekarang"
                        if blink_checker:
                            res = blink_checker.update(face, detector)
                            if res.get("complete", False):
                                action_passed = True
                                _print_log("Blink challenge PASSED", "SUCCESS")
                    else:
                        # Pose Challenge
                        instruction = f"Toleh / Angguk ke {current_action}"
                        pose = pose_estimator.estimate(face, detector)
                        yaw, pitch, roll = pose.get("yaw", 0), pose.get("pitch", 0), pose.get("roll", 0)

                        if current_action == "KANAN" and yaw > config.CHALLENGE_YAW:
                            action_passed = True
                        elif current_action == "KIRI" and yaw < -config.CHALLENGE_YAW:
                            action_passed = True
                        elif current_action == "ATAS" and pitch < -config.CHALLENGE_PITCH:
                            action_passed = True
                        elif current_action == "BAWAH" and pitch > config.CHALLENGE_PITCH:
                            action_passed = True
                        elif current_action == "MIRING_KANAN" and roll > config.CHALLENGE_ROLL:
                            action_passed = True
                        elif current_action == "MIRING_KIRI" and roll < -config.CHALLENGE_ROLL:
                            action_passed = True

                    # Tampilkan informasi di layar
                    status_text = f"{last_name} | {instruction}"
                    _draw_status(display, x_f, y_f, w_f, h_f, status_text, config.COLOR_CYAN)

                    if action_passed:
                        current_step_idx += 1
                        if current_step_idx >= len(challenge_sequence):
                            state = ValidationState.UNLOCKED
                            door.unlock()
                            _print_log(f"CHALLENGE BERHASIL → Pintu terbuka untuk {last_name}", "SUCCESS")
                        else:
                            _print_log(f"Step {current_step_idx} selesai, lanjut ke step berikutnya", "INFO")

                # 3. Hasil Akhir
                elif state == ValidationState.UNLOCKED:
                    _draw_status(display, x_f, y_f, w_f, h_f, f"SELAMAT DATANG, {last_name}", config.COLOR_GREEN)

                elif state == ValidationState.UNMATCHED:
                    _draw_status(display, x_f, y_f, w_f, h_f, "WAJAH TIDAK Dikenali", config.COLOR_RED)

            # Status Pintu
            status_pintu = "TERBUKA" if not door.locked else "TERKUNCI"
            p_color = config.COLOR_GREEN if not door.locked else config.COLOR_RED
            _put(display, f"PINTU: {status_pintu}", config.FRAME_HEIGHT - 30, p_color, scale=0.75)

            cv2.imshow("Smart Door Lock", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        _print_log(f"Error utama: {e}", "ERROR")
        import traceback
        traceback.print_exc()
    finally:
        cam.stop()
        door.cleanup()
        cv2.destroyAllWindows()
        _print_log("Sistem Smart Door ditutup.", "SYSTEM")


if __name__ == "__main__":
    run_unlock()