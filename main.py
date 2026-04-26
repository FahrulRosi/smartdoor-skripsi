import sys
import cv2
import numpy as np
from datetime import datetime
from enum import Enum

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from door.door_lock             import DoorLock
from liveness.anti_spoofing     import SilentAntiSpoofing
from database.face_db           import get_all_faces

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ─── STATE MACHINE ────────────────────────────────────────────────────────────
class ValidationState(Enum):
    IDLE = 0                    # Menunggu wajah
    RECOGNIZING = 1             # Sedang memproses identitas
    UNMATCHED = 2               # Wajah tidak ada di database
    SPOOFED = 3                 # Terdeteksi Foto atau Layar HP
    UNLOCKED = 4                # Pintu Terbuka (Selama wajah ada di kamera)

# ─── HUD HELPERS ──────────────────────────────────────────────────────────────
def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def _draw_validation_box(frame, x, y, w, h, name="", score=0.0, color=config.COLOR_CYAN, status=""):
    """Gambar kotak wajah dengan label nama di bagian ATAS kotak"""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    if name:
        label = f"{name} ({score:.2f})"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x, y - label_height - 15), (x + label_width + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if status:
        cv2.putText(frame, status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def _print_log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def run_unlock():
    _print_log("Memulai Smart Door Lock (Mode Instan Tanpa Waktu)...", "SYSTEM")

    registered_faces = get_all_faces()
    if not registered_faces:
        _print_log("PERINGATAN: Database kosong. Jalankan register.py dahulu.", "WARNING")

    if GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.IR_CUT_PIN, GPIO.OUT)
            GPIO.output(config.IR_CUT_PIN, GPIO.HIGH)
        except Exception as e:
            _print_log(f"Gagal setup hardware: {e}", "ERROR")

    try:
        cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    except Exception as e:
        _print_log(f"Kamera gagal: {e}", "ERROR")
        return

    try:
        detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        anti_spoofing = SilentAntiSpoofing(model_path=config.ANTI_SPOOFING_MODEL, threshold=config.ANTI_SPOOFING_THRESHOLD)
        model = MobileFaceNet()
        matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
        
        # Pintu diatur dengan unlock_duration=0 agar tidak ada timer bawaan yang menyala
        door = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=0) 
        _print_log("Sistem Siap!", "SYSTEM")
    except Exception as e:
        _print_log(f"Gagal load AI: {e}", "ERROR")
        return

    state = ValidationState.IDLE
    consecutive_matches = 0
    min_required_matches = 3 
    last_match_name = ""
    last_match_score = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            
            display = frame.copy()
            faces = detector.detect(frame)

            # ──── LOGIKA PENGUNCIAN INSTAN ────
            if not faces:
                # Jika tidak ada wajah sama sekali, PAKSA pintu terkunci
                if state == ValidationState.UNLOCKED:
                    _print_log("Wajah hilang dari kamera. Pintu dikunci instan.", "SYSTEM")
                
                state = ValidationState.IDLE
                consecutive_matches = 0
                door.lock() # Perintah langsung mengunci pintu
                
                _put(display, "Menunggu Wajah...", 70, config.COLOR_YELLOW, scale=0.6)
            else:
                face = faces[0]
                x, y, w, h = face.bbox

                if state != ValidationState.UNLOCKED:
                    # 1. Cek Liveness Aktif (Anti-Spoofing)
                    liveness = anti_spoofing.is_real(frame, face.bbox)
                    
                    if not liveness["real"]:
                        state = ValidationState.SPOOFED
                        door.lock()
                    else:
                        # 2. Cek Identitas (MobileFaceNet)
                        state = ValidationState.RECOGNIZING
                        face_crop = model.crop_face(frame, face.bbox)
                        embedding = model.get_embedding(face_crop)
                        match = matcher.match(embedding)

                        if match["matched"]:
                            last_match_name = match["name"]
                            last_match_score = match["score"]
                            consecutive_matches += 1
                            
                            # 3. BUKA PINTU INSTAN
                            if consecutive_matches >= min_required_matches:
                                state = ValidationState.UNLOCKED
                                _print_log(f"AKSES DIBERIKAN: {last_match_name}", "ACCESS")
                                door.unlock() # Perintah membuka pintu
                        else:
                            state = ValidationState.UNMATCHED
                            last_match_name = "Tidak Dikenal"
                            last_match_score = match["score"]
                            consecutive_matches = 0
                            door.lock()

            # --- Indikator Status Pintu Kiri Atas ---
            p_color = config.COLOR_GREEN if not door.locked else config.COLOR_RED
            door_msg = "TERBUKA" if not door.locked else "TERKUNCI"
            _put(display, f"PINTU: {door_msg}", 30, p_color, scale=0.7)

            # --- Visualisasi Box Wajah ---
            if faces:
                if state == ValidationState.SPOOFED:
                    _draw_validation_box(display, x, y, w, h, "PALSU", color=config.COLOR_RED)
                    _put(display, "AKSES DITOLAK: FOTO TERDETEKSI!", 120, config.COLOR_RED, scale=0.5)

                elif state == ValidationState.UNMATCHED:
                    _draw_validation_box(display, x, y, w, h, "Unknown", last_match_score, color=config.COLOR_RED)
                    _put(display, "Wajah tidak terdaftar.", 120, config.COLOR_RED, scale=0.5)

                elif state == ValidationState.RECOGNIZING:
                    _draw_validation_box(display, x, y, w, h, "Verifikasi...", 0.0, color=config.COLOR_CYAN)

                elif state == ValidationState.UNLOCKED:
                    _draw_validation_box(display, x, y, w, h, last_match_name, last_match_score, color=config.COLOR_GREEN)
                    _put(display, f"Selamat Datang, {last_match_name}!", 120, config.COLOR_GREEN, scale=0.8)

            cv2.imshow("Smart Door Lock", display)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    finally:
        if GPIO_AVAILABLE: GPIO.cleanup()
        cam.stop()
        detector.close()
        door.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unlock()