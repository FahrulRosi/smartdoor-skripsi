import sys
import cv2
import time
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
    IDLE = 0                    # Menunggu deteksi wajah
    FACE_DETECTED = 1           # Wajah terdeteksi
    CHECKING_LIVENESS = 2       # Validasi anti-spoofing
    RECOGNIZING = 3             # Face recognition
    MATCHED = 4                 # Wajah match dengan database
    UNMATCHED = 5               # Wajah tidak di database
    SPOOFED = 6                 # Spoofing terdeteksi
    CONFIRMING = 7              # Waiting for confirmation (countdown)
    UNLOCKED = 8                # Pintu terbuka

# ─── HUD HELPERS ──────────────────────────────────────────────────────────────
def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def _draw_validation_box(frame, x, y, w, h, status, name="", score=0.0, color=config.COLOR_CYAN):
    """Gambar kotak validasi utama dengan info"""
    # Kotak luar (tebal)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    
    # Garis vertikal & horizontal
    cv2.line(frame, (x + 5, y + 5), (x + 20, y + 5), color, 2)
    cv2.line(frame, (x + 5, y + 5), (x + 5, y + 20), color, 2)
    cv2.line(frame, (x + w - 20, y + 5), (x + w - 5, y + 5), color, 2)
    cv2.line(frame, (x + w - 5, y + 5), (x + w - 5, y + 20), color, 2)
    
    # Status banner atas
    banner_h = 30
    cv2.rectangle(frame, (x, y), (x + w, y + banner_h), color, -1)
    cv2.putText(frame, status, (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Info box bawah
    if name:
        info_text = f"{name} ({score:.3f})"
        info_h = 25
        cv2.rectangle(frame, (x, y + h - info_h), (x + w, y + h), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (x + 10, y + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def _draw_confidence_bar(frame, score, threshold, x, y, width=200, height=15):
    """Gambar bar confidence dengan threshold line"""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    
    # Threshold line (kuning)
    threshold_pos = int(x + width * threshold)
    cv2.line(frame, (threshold_pos, y), (threshold_pos, y + height), config.COLOR_YELLOW, 2)
    
    # Score bar
    if score > 0:
        score_pos = int(x + width * min(score, 1.0))
        bar_color = config.COLOR_GREEN if score >= threshold else config.COLOR_RED
        cv2.rectangle(frame, (x, y), (score_pos, y + height), bar_color, -1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), config.COLOR_WHITE, 1)

def _draw_countdown_timer(frame, seconds_left, x, y):
    """Gambar countdown timer besar"""
    # Background lingkaran
    center = (x, y)
    radius = 40
    cv2.circle(frame, center, radius, config.COLOR_GREEN, -1)
    cv2.circle(frame, center, radius, config.COLOR_WHITE, 2)
    
    # Text countdown
    text = str(int(seconds_left))
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = x - text_size[0] // 2
    text_y = y + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

def _print_log(msg, level="INFO"):
    """Print log dengan timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def _display_registered_faces(registered_faces):
    """Tampilkan daftar wajah terdaftar"""
    _print_log("="*60, "INFO")
    _print_log(f"Database Wajah Terdaftar: {len(registered_faces)} orang", "INFO")
    for idx, face_data in enumerate(registered_faces, 1):
        _print_log(f"  {idx}. {face_data['name']}", "INFO")
    _print_log("="*60, "INFO")

# ─── MAIN PROCESS ─────────────────────────────────────────────────────────────
def run_unlock():
    _print_log("Memulai Smart Door Lock...", "SYSTEM")

    # Load registered faces
    registered_faces = get_all_faces()
    if not registered_faces:
        _print_log("⚠️  Tidak ada wajah terdaftar! Jalankan register.py terlebih dahulu.", "WARNING")
    else:
        _display_registered_faces(registered_faces)

    # 1. Setup GPIO & IR-CUT
    IR_CUT_PIN = 12
    if GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(IR_CUT_PIN, GPIO.OUT)
            GPIO.output(IR_CUT_PIN, GPIO.HIGH)
            _print_log("IR-CUT Filter Enabled", "CAMERA")
        except Exception as e:
            _print_log(f"Gagal setup IR-CUT: {e}", "ERROR")

    # 2. Inisialisasi camera
    try:
        cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        _print_log("Camera Stream Started", "CAMERA")
    except Exception as e:
        _print_log(f"Gagal inisialisasi camera: {e}", "ERROR")
        return

    detector = FaceMeshDetector(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    # 3. Inisialisasi Model AI
    try:
        anti_spoofing = SilentAntiSpoofing(model_path="liveness/2.7_80x80_MiniFASNetV2.onnx", threshold=0.85)
        model = MobileFaceNet()
        matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
        door = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=config.UNLOCK_DURATION)
        _print_log("Semua model AI berhasil dimuat", "SYSTEM")
    except Exception as e:
        _print_log(f"Gagal load model: {e}", "ERROR")
        return

    # ─── STATE VARIABLES ───
    state = ValidationState.IDLE
    last_match_name = None
    last_match_score = 0.0
    confirmation_time = 0.0
    confirmation_duration = 3.0  # 3 detik untuk konfirmasi
    frame_count = 0
    consecutive_matches = 0
    min_consecutive_matches = 5  # Perlu 5 frame dengan match yang sama

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            
            display = frame.copy()
            frame_count += 1

            # ─── HEADER ───
            p_color = config.COLOR_GREEN if not door.locked else config.COLOR_RED
            _put(display, f"Pintu: {door.status()}", 25, p_color, scale=0.7)
            _put(display, f"Frame: {frame_count}", 55, config.COLOR_WHITE, scale=0.5)

            # ─── STATE MACHINE ───
            faces = detector.detect(frame)

            if not faces:
                state = ValidationState.IDLE
                consecutive_matches = 0
                _put(display, "🔍 Silahkan hadapkan wajah ke kamera", 90, config.COLOR_YELLOW, scale=0.7)
                
            else:
                face = faces[0]
                x, y, w, h = face.bbox
                
                # Tampilkan FaceMesh & bbox
                display = detector.draw(display, face)
                cv2.rectangle(display, (x, y), (x + w, y + h), config.COLOR_CYAN, 2)

                # ──── STATE 1: FACE DETECTED ────
                if state == ValidationState.IDLE:
                    state = ValidationState.FACE_DETECTED
                    consecutive_matches = 0
                    _print_log("🔵 Wajah terdeteksi - Memulai validasi", "INFO")

                # ──── STATE 2: CHECK LIVENESS ────
                if state in [ValidationState.FACE_DETECTED, ValidationState.CHECKING_LIVENESS]:
                    state = ValidationState.CHECKING_LIVENESS
                    liveness = anti_spoofing.is_real(frame, face.bbox)
                    
                    if not liveness["real"]:
                        state = ValidationState.SPOOFED
                        _print_log("🚫 Spoofing Terdeteksi!", "SECURITY")
                    else:
                        state = ValidationState.RECOGNIZING

                # ──── STATE 3: RECOGNIZE ────
                if state == ValidationState.RECOGNIZING:
                    face_crop = model.crop_face(frame, face.bbox)
                    embedding = model.get_embedding(face_crop)
                    match = matcher.match(embedding)

                    if match["matched"]:
                        state = ValidationState.MATCHED
                        last_match_name = match["name"]
                        last_match_score = match["score"]
                        consecutive_matches += 1
                        _print_log(f"✅ Match: {last_match_name} ({last_match_score:.4f}) [{consecutive_matches}/5]", "INFO")
                    else:
                        state = ValidationState.UNMATCHED
                        last_match_name = match["name"]
                        last_match_score = match["score"]
                        consecutive_matches = 0
                        _print_log(f"❌ No match: Closest {last_match_name} ({last_match_score:.4f})", "INFO")

                # ──── STATE 4: CONFIRMATION ────
                if state == ValidationState.MATCHED and consecutive_matches >= min_consecutive_matches:
                    state = ValidationState.CONFIRMING
                    confirmation_time = time.time()
                    _print_log(f"⏳ Konfirmasi access untuk {last_match_name}...", "INFO")

                # ──── STATE 5: UNLOCK ────
                if state == ValidationState.CONFIRMING:
                    elapsed = time.time() - confirmation_time
                    remaining = max(0, confirmation_duration - elapsed)
                    
                    if remaining > 0:
                        # Masih countdown
                        _draw_countdown_timer(display, remaining, config.FRAME_WIDTH - 50, 80)
                    else:
                        # Waktu selesai - buka pintu
                        state = ValidationState.UNLOCKED
                        _print_log(f"🔓 ACCESS GRANTED: {last_match_name}", "ACCESS")
                        door.unlock()

                # ──── RENDER STATE ────
                if state == ValidationState.SPOOFED:
                    _draw_validation_box(display, x, y, w, h, "❌ SPOOFING", "Foto/Layar", 0.0, config.COLOR_RED)
                    _put(display, "Akses Ditolak: Spoofing Terdeteksi", 150, config.COLOR_RED, scale=0.8)
                    
                elif state == ValidationState.RECOGNIZING:
                    _draw_validation_box(display, x, y, w, h, "🔄 Verifikasi", "", 0.0, config.COLOR_YELLOW)
                    _put(display, "Sedang menganalisis...", 150, config.COLOR_YELLOW, scale=0.7)
                    
                elif state == ValidationState.MATCHED:
                    match_pct = int((consecutive_matches / min_consecutive_matches) * 100)
                    _draw_validation_box(display, x, y, w, h, f"✅ Match ({match_pct}%)", last_match_name, last_match_score, config.COLOR_GREEN)
                    _draw_confidence_bar(display, last_match_score, config.MATCH_THRESHOLD, x + 10, y + h + 15)
                    _put(display, f"Mengkonfirmasi: {last_match_name}", 150, config.COLOR_GREEN, scale=0.8)
                    
                elif state == ValidationState.UNMATCHED:
                    _draw_validation_box(display, x, y, w, h, "❓ Unknown", last_match_name if last_match_name else "Unknown", last_match_score, config.COLOR_RED)
                    _draw_confidence_bar(display, last_match_score, config.MATCH_THRESHOLD, x + 10, y + h + 15)
                    _put(display, "Wajah tidak terdaftar", 150, config.COLOR_RED, scale=0.8)
                    
                elif state == ValidationState.CONFIRMING:
                    _draw_validation_box(display, x, y, w, h, "🔐 Buka Pintu", last_match_name, last_match_score, config.COLOR_GREEN)
                    _put(display, "Menunggu countdown...", 150, config.COLOR_GREEN, scale=0.8)
                    
                elif state == ValidationState.UNLOCKED:
                    _draw_validation_box(display, x, y, w, h, "🔓 TERBUKA", last_match_name, last_match_score, config.COLOR_GREEN)
                    _put(display, "Pintu sedang terbuka - Masuk!", 150, config.COLOR_GREEN, scale=0.9)

            cv2.imshow("Smart Door Lock", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _print_log("Exit diminta user", "INFO")
                break

    except Exception as e:
        _print_log(f"Error: {e}", "ERROR")
    
    finally:
        _print_log("Cleanup...", "SYSTEM")
        if GPIO_AVAILABLE: 
            GPIO.cleanup()
        cam.stop()
        detector.close()
        door.cleanup()
        cv2.destroyAllWindows()
        _print_log("Smart Door Lock Stopped", "SYSTEM")

if __name__ == "__main__":
    run_unlock()