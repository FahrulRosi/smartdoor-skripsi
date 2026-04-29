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
from liveness.anti_spoofing     import SilentAntiSpoofing  # <-- Modul Anti-Spoofing dimasukkan

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
    # Background teks utama
    t_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.rectangle(frame, (x, y - 35), (x + t_size[0] + 15, y - 5), color, -1)
    _put(frame, status, y - 12, (255, 255, 255), x + 8, scale=0.65, thickness=2)


def _draw_challenge_info(frame, step_idx, total_steps, instruction):
    """Fungsi khusus untuk menggambar kotak keterangan challenge di tengah atas layar"""
    box_w, box_h = 450, 70
    x = (config.FRAME_WIDTH - box_w) // 2
    y = 10
    
    # Kotak background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), config.COLOR_CYAN, 2)
    
    # Teks progress dan instruksi
    _put(frame, f"Tahap Liveness ({step_idx + 1}/{total_steps})", y + 25, config.COLOR_YELLOW, x + 15, scale=0.6)
    _put(frame, instruction, y + 55, config.COLOR_WHITE, x + 15, scale=0.75, thickness=2)


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
    
    anti_spoof = SilentAntiSpoofing() # <-- Inisialisasi Anti Spoofing

    state = ValidationState.IDLE
    last_name = ""
    user_data = None
    challenge_sequence = []
    current_step_idx = 0
    blink_checker = None

    # Daftar arah tantangan pose
    POSE_DIRECTIONS = ["KANAN", "KIRI", "ATAS", "BAWAH", "MIRING_KANAN", "MIRING_KIRI"]

    # Mapping aksi ke teks instruksi yang mudah dibaca user
    INSTRUCTION_TEXT = {
        "BLINK": "Tantangan: Kedipkan Mata Anda",
        "KANAN": "Tantangan: Toleh Kepala ke KANAN",
        "KIRI":  "Tantangan: Toleh Kepala ke KIRI",
        "ATAS":  "Tantangan: Dongak Kepala ke ATAS",
        "BAWAH": "Tantangan: Tunduk Kepala ke BAWAH",
        "MIRING_KANAN": "Tantangan: Miringkan Kepala ke KANAN",
        "MIRING_KIRI":  "Tantangan: Miringkan Kepala ke KIRI"
    }

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
                    user_data = None
                door.lock()
                _put(display, "Menunggu Wajah...", 50, config.COLOR_YELLOW, scale=0.9)
            else:
                face = faces[0]
                x_f, y_f, w_f, h_f = face.bbox

                # ==================== CEK ANTI-SPOOFING ====================
                # Mencegat foto/video dari layar HP atau kertas cetak
                spoof = anti_spoof.is_real(frame, face.bbox)
                if not spoof.get("real", True):
                    state = ValidationState.IDLE
                    last_name = ""
                    user_data = None
                    _draw_status(display, x_f, y_f, w_f, h_f, "WAJAH PALSU / FOTO HP!", config.COLOR_RED)
                    _put(display, f"Skor Liveness: {spoof.get('score', 0):.3f}", y_f - 40, config.COLOR_RED)
                
                else:
                    # ==================== STATE MACHINE ====================
                    if state == ValidationState.IDLE:
                        state = ValidationState.RECOGNIZING

                    # 1. RECOGNITION (Cocokkan wajah dengan database)
                    elif state == ValidationState.RECOGNIZING:
                        face_crop = model.crop_face(frame, face.bbox)
                        embedding = model.get_embedding(face_crop)
                        
                        match = matcher.match(embedding)
                        skor = match.get("score", 0.0)

                        if match.get("matched", False):
                            last_name = match["name"]
                            
                            all_faces_in_db = get_all_faces()
                            user_data = next((f for f in all_faces_in_db if f["name"] == last_name), None)
                            
                            state = ValidationState.CHALLENGE
                            
                            # Buat challenge acak (1 pose + 1 blink)
                            pose_choice = random.choice(POSE_DIRECTIONS)
                            challenge_sequence = [pose_choice, "BLINK"]
                            current_step_idx = 0
                            blink_checker = BlinkDetector(target_blinks=1)
                            
                            _print_log(f"Wajah dikenali: {last_name} | Skor: {skor:.3f} | Challenge: {challenge_sequence}", "SUCCESS")
                        else:
                            last_name = "Tidak Dikenali"
                            user_data = None
                            state = ValidationState.UNMATCHED
                            _print_log(f"Ditolak! Skor kemiripan: {skor:.3f} (Butuh minimal: {config.MATCH_THRESHOLD})", "WARNING")

                    # 2. CHALLENGE (Liveness Verification - Gerakan)
                    elif state == ValidationState.CHALLENGE:
                        current_action = challenge_sequence[current_step_idx]
                        action_passed = False
                        
                        # Teks instruksi untuk ditampilkan di layar
                        instruction_text = INSTRUCTION_TEXT.get(current_action, "Ikuti instruksi...")

                        # Cek aksi pengguna
                        if current_action == "BLINK":
                            if blink_checker:
                                res = blink_checker.update(face, detector)
                                if res.get("complete", False):
                                    action_passed = True
                                    _print_log("Blink challenge PASSED", "SUCCESS")
                        else:
                            # Cek Pose Challenge
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

                        # Gambar Keterangan/Instruksi di HUD
                        _draw_status(display, x_f, y_f, w_f, h_f, f"User: {last_name}", config.COLOR_CYAN)
                        _draw_challenge_info(display, current_step_idx, len(challenge_sequence), instruction_text)

                        # Pindah step jika aksi berhasil dilakukan
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
                        _draw_status(display, x_f, y_f, w_f, h_f, "WAJAH TIDAK DIKENALI", config.COLOR_RED)

            # Status Pintu di Pojok Kiri Bawah
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