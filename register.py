import sys
import cv2
import time
from datetime import datetime
from enum import Enum
import numpy as np

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from database.face_db           import save_face
from liveness.anti_spoofing     import SilentAntiSpoofing

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ─── REGISTRATION STAGES ──────────────────────────────────────────────────────
class RegistrationStage(Enum):
    IDLE = 0
    FACEMESH = 1
    YAW = 2
    PITCH = 3
    ROLL = 4
    BLINK = 5
    EXTRACTION = 6
    COMPLETE = 7

STAGE_NAMES = {
    RegistrationStage.FACEMESH: "1️⃣ Deteksi Wajah (Lurus)",
    RegistrationStage.YAW: "2️⃣ Toleh Kepala (Kiri/Kanan)",
    RegistrationStage.PITCH: "3️⃣ Angkat/Tunduk Kepala",
    RegistrationStage.ROLL: "4️⃣ Miringkan Kepala",
    RegistrationStage.BLINK: "5️⃣ Berkedip",
    RegistrationStage.EXTRACTION: "6️⃣ Ekstraksi Data",
}

# ─── HUD HELPERS ──────────────────────────────────────────────────────────────
def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def _draw_stage_progress_bar(frame, current_stage, total_stages=6):
    """Gambar progress bar tahap yang rapi"""
    bar_width = 350
    bar_height = 25
    x = (config.FRAME_WIDTH - bar_width) // 2
    y = 15
    
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (30, 30, 30), -1)
    
    # Hitung progress
    stage_val = current_stage.value
    if stage_val > total_stages: stage_val = total_stages
    progress = (stage_val - 1) / total_stages
    
    progress_width = int(bar_width * progress)
    if progress_width > 0:
        cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height), config.COLOR_GREEN, -1)
    
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), config.COLOR_WHITE, 2)
    
    text = f"Tahap {stage_val if stage_val <= 5 else 6}/6"
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    t_x = x + (bar_width - t_size[0]) // 2
    t_y = y + (bar_height + t_size[1]) // 2
    cv2.putText(frame, text, (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

def _draw_status_panel(frame, stage, instruction, progress="", details=""):
    """Gambar panel instruksi di bagian atas layar"""
    panel_height = 100
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_height), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_height), config.COLOR_CYAN, 2)
    
    stage_name = STAGE_NAMES.get(stage, "Proses...")
    _put(frame, stage_name, 75, config.COLOR_GREEN, 20, scale=0.85, thickness=2)
    _put(frame, instruction, 105, config.COLOR_YELLOW, 20, scale=0.65, thickness=2)
    
    if progress:
        _put(frame, f"Progress: {progress}", 125, config.COLOR_CYAN, 20, scale=0.6)
    if details:
        _put(frame, details, 145, config.COLOR_WHITE, 20, scale=0.55)

def _draw_validation_box(frame, face_bbox, status, color):
    """Gambar validation box melingkupi wajah"""
    x, y, w, h = face_bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    
    corner_len = 25
    cv2.line(frame, (x, y), (x + corner_len, y), color, 3)
    cv2.line(frame, (x, y), (x, y + corner_len), color, 3)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, 3)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, 3)
    
    status_bg_h = 30
    cv2.rectangle(frame, (x, y - status_bg_h - 5), (x + 180, y - 5), color, -1)
    cv2.putText(frame, status, (x + 5, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

def _print_log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def run_register(name):
    _print_log(f"Memulai registrasi untuk: {name}", "SYSTEM")
    print("\n" + "="*70)
    print("  MEMULAI REGISTRASI WAJAH")
    print("  Selesaikan 5 instruksi gerakan kepala untuk mencegah Spoofing.")
    print("="*70 + "\n")

    # Setup IR-CUT
    IR_CUT_PIN = 12
    if GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(IR_CUT_PIN, GPIO.OUT)
            GPIO.output(IR_CUT_PIN, GPIO.HIGH)
            _print_log("IR-CUT Filter Enabled", "CAMERA")
        except Exception as e:
            _print_log(f"Gagal setup IR-CUT: {e}", "WARNING")

    # Load Kamera & Model
    try:
        cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        _print_log("Camera Stream Started", "CAMERA")
    except Exception as e:
        _print_log(f"Gagal inisialisasi camera: {e}", "ERROR")
        return

    detector = FaceMeshDetector(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    liveness = LivenessManager()
    model = MobileFaceNet()
    spoof_ai = SilentAntiSpoofing(model_path="liveness/2.7_80x80_MiniFASNetV2.onnx", threshold=0.85)

    liveness.start_register()

    current_stage = RegistrationStage.FACEMESH
    extraction_in_progress = False
    frame_count = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            
            display = frame.copy()
            frame_count += 1
            faces = detector.detect(frame)

            # ──── KONDISI: TIDAK ADA WAJAH ────
            if not faces:
                cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.COLOR_RED, 4)
                _draw_stage_progress_bar(display, current_stage)
                _draw_status_panel(display, current_stage, 
                                 "🔍 Wajah tidak terdeteksi", 
                                 "Silahkan hadapkan wajah ke kamera")
            else:
                face = faces[0]
                display = detector.draw(display, face)
                
                # Cek Liveness Model AI (Spoofing Foto/Layar)
                liveness_check = spoof_ai.is_real(frame, face.bbox)
                
                if not liveness_check["real"]:
                    _draw_validation_box(display, face.bbox, "❌ SPOOFING", config.COLOR_RED)
                    _draw_stage_progress_bar(display, current_stage)
                    _draw_status_panel(display, current_stage, "Spoofing Terdeteksi!", "Gunakan wajah asli!")
                    _print_log("🚫 Spoofing Detected", "SECURITY")

                else:
                    # ──── UPDATE LOGIKA LIVENESS GERAKAN ────
                    result = liveness.update_register(face, detector)
                    
                    # Update status tahapan UI
                    step_to_stage = {
                        "FACEMESH": RegistrationStage.FACEMESH,
                        "YAW": RegistrationStage.YAW,
                        "PITCH": RegistrationStage.PITCH,
                        "ROLL": RegistrationStage.ROLL,
                        "BLINK": RegistrationStage.BLINK,
                        "DONE": RegistrationStage.EXTRACTION,
                    }
                    
                    # [PENTING] Jika statusnya "WAIT" (Cooldown transisi), tahan di tahap saat ini
                    if result["step"] != "WAIT":
                        current_stage = step_to_stage.get(result["step"], current_stage)
                    
                    # Logika Warna UI Berdasarkan Status
                    if result["step"] == "DONE" or "DONE" in result.get("progress", ""):
                        box_color = config.COLOR_GREEN
                        status_text = "✅ PASSED"
                    elif result["step"] == "WAIT":
                        box_color = config.COLOR_YELLOW  # Warna Jeda/Transisi
                        status_text = "⏳ TAHAN LURUS"
                    elif result["status"] == "pending":
                        box_color = config.COLOR_CYAN
                        status_text = "🔍 CHECKING"
                    else:
                        box_color = config.COLOR_WHITE
                        status_text = "..."

                    # Gambar UI
                    _draw_validation_box(display, face.bbox, status_text, box_color)
                    _draw_stage_progress_bar(display, current_stage)
                    
                    # Ekstraksi sudut kepala untuk indikator teks
                    details = ""
                    if "yaw" in result: details += f"YAW: {result['yaw']}   "
                    if "pitch" in result: details += f"PITCH: {result['pitch']}   "
                    if "roll" in result: details += f"ROLL: {result['roll']}"
                    
                    _draw_status_panel(display, current_stage, 
                                     result["instruction"], 
                                     result.get("progress", ""),
                                     details)

                    # ──── TAHAP FINAL: EKSTRAKSI & SIMPAN DATABASE ────
                    if result["status"] == "complete" and not extraction_in_progress:
                        _print_log("✅ Semua tahap gerakan berhasil diselesaikan!", "SUCCESS")
                        extraction_in_progress = True
                        
                        # Beri jeda 1 detik agar user bisa bernapas sebelum ambil foto database
                        cv2.imshow("Register", display)
                        cv2.waitKey(1000)

                        # Proses Vektor Wajah (Embedding)
                        _print_log("Mengekstraksi fitur wajah...", "SYSTEM")
                        face_crop = model.crop_face(frame, face.bbox)
                        embedding = model.get_embedding(face_crop)
                        
                        # Simpan ke faces.pkl
                        save_face(name, embedding)
                        
                        # Layar Sukses
                        current_stage = RegistrationStage.COMPLETE
                        display = frame.copy()
                        display = detector.draw(display, face)
                        _draw_validation_box(display, face.bbox, "✅ BERHASIL", config.COLOR_GREEN)
                        
                        cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.COLOR_GREEN, 12)
                        _put(display, "🎉 REGISTRASI BERHASIL!", 
                             config.FRAME_HEIGHT // 2 - 50, config.COLOR_GREEN, 
                             x=(config.FRAME_WIDTH - 400) // 2, scale=1.6, thickness=3)
                        _put(display, f"Nama: {name}", 
                             config.FRAME_HEIGHT // 2 + 10, config.COLOR_GREEN,
                             x=(config.FRAME_WIDTH - 150) // 2, scale=1.1, thickness=2)
                        _put(display, "Data wajah Anda telah disimpan di database.", 
                             config.FRAME_HEIGHT // 2 + 50, config.COLOR_WHITE,
                             x=(config.FRAME_WIDTH - 380) // 2, scale=0.7)
                        _put(display, "Menutup kamera dalam 4 detik...", 
                             config.FRAME_HEIGHT // 2 + 90, config.COLOR_YELLOW,
                             x=(config.FRAME_WIDTH - 250) // 2, scale=0.6)
                        
                        cv2.imshow("Register", display)
                        cv2.waitKey(4000)
                        break

            cv2.imshow("Register", display)
            
            # Tekan Q untuk keluar
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                _print_log("Registrasi dibatalkan oleh user", "WARNING")
                break

    except Exception as e:
        _print_log(f"Terjadi Kesalahan: {e}", "ERROR")
    
    finally:
        _print_log("Membersihkan memori dan mematikan kamera...", "SYSTEM")
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()
        _print_log("Program Selesai.", "SYSTEM")

if __name__ == "__main__":
    name = input("\n📝 Masukkan nama Anda (tanpa spasi/simbol aneh): ").strip()
    
    if not name:
        print("❌ Nama tidak boleh kosong!")
        sys.exit(1)
    
    run_register(name)