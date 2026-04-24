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
    RegistrationStage.FACEMESH: "1️⃣ Deteksi Wajah",
    RegistrationStage.YAW: "2️⃣ Toleh Kepala",
    RegistrationStage.PITCH: "3️⃣ Angkat/Tunduk",
    RegistrationStage.ROLL: "4️⃣ Miringkan Kepala",
    RegistrationStage.BLINK: "5️⃣ Berkedip",
    RegistrationStage.EXTRACTION: "6️⃣ Ekstraksi Data",
}

# ─── HUD HELPERS ──────────────────────────────────────────────────────────────
def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def _draw_progress_circle(frame, progress_pct, x, y, radius=30, color=config.COLOR_GREEN):
    """Gambar circular progress"""
    cv2.circle(frame, (x, y), radius, (50, 50, 50), -1)
    cv2.circle(frame, (x, y), radius, config.COLOR_WHITE, 2)
    
    # Angle untuk progress
    angle = int(360 * progress_pct / 100)
    if angle > 0:
        cv2.ellipse(frame, (x, y), (radius - 2, radius - 2), 0, 0, angle, color, 3)
    
    # Text
    text = f"{progress_pct}%"
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    t_x = x - t_size[0] // 2
    t_y = y + t_size[1] // 2
    cv2.putText(frame, text, (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def _draw_stage_progress_bar(frame, current_stage, total_stages=6):
    """Gambar progress bar tahap"""
    bar_width = 350
    bar_height = 25
    x = (config.FRAME_WIDTH - bar_width) // 2
    y = 15
    
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (30, 30, 30), -1)
    
    progress = (current_stage.value - 1) / total_stages
    progress_width = int(bar_width * progress)
    cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height), config.COLOR_GREEN, -1)
    
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), config.COLOR_WHITE, 2)
    
    text = f"Tahap {current_stage.value - 1}/6"
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    t_x = x + (bar_width - t_size[0]) // 2
    t_y = y + (bar_height + t_size[1]) // 2
    cv2.putText(frame, text, (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

def _draw_status_panel(frame, stage, instruction, progress="", details=""):
    """Gambar panel status lengkap"""
    panel_height = 100
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_height), config.COLOR_CYAN, 2)
    
    # Stage name
    stage_name = STAGE_NAMES.get(stage, "Unknown")
    _put(frame, stage_name, 75, config.COLOR_GREEN, 20, scale=0.85, thickness=2)
    
    # Instruction
    _put(frame, instruction, 105, config.COLOR_YELLOW, 20, scale=0.65)
    
    # Progress
    if progress:
        _put(frame, f"Progress: {progress}", 125, config.COLOR_CYAN, 20, scale=0.6)
    
    # Details
    if details:
        _put(frame, details, 145, config.COLOR_WHITE, 20, scale=0.55)

def _draw_validation_box(frame, face_bbox, status, color):
    """Gambar validation box di sekitar wajah"""
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
    """Print log dengan timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def run_register(name):
    """Alur registrasi berurutan dengan validasi frame stabil"""
    
    _print_log(f"Memulai registrasi untuk: {name}", "SYSTEM")
    print("\n" + "="*70)
    print("  REGISTRASI DENGAN VALIDASI STABIL (15 FRAME PER TAHAP)")
    print("  1. Deteksi Wajah (Lurus)")
    print("  2. Toleh Kepala (20° +)")
    print("  3. Angkat/Tunduk (15° +)")
    print("  4. Miringkan Kepala (15° +)")
    print("  5. Berkedip (2x)")
    print("  6. Ekstraksi Data (MobileFaceNet)")
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

    # Inisialisasi
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

    # State variables
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

            # ──── NO FACE ────
            if not faces:
                # Debug: Tampilkan status deteksi
                cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), 
                              config.COLOR_RED, 4)
                _draw_stage_progress_bar(display, current_stage)
                _draw_status_panel(display, current_stage, 
                                 "🔍 Scanning... (FaceMesh)", 
                                 "Silahkan hadapkan wajah ke kamera")
                
                # Debug text
                _put(display, "DEBUG: FaceMesh sedang mencari", config.FRAME_HEIGHT - 80, 
                     config.COLOR_YELLOW, scale=0.6)
                _put(display, f"Frame: {frame_count} | Brightness: {np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)):.0f}", 
                     config.FRAME_HEIGHT - 50, config.COLOR_WHITE, scale=0.5)

            else:
                face = faces[0]
                display = detector.draw(display, face)
                
                # Anti-spoofing check
                liveness_check = spoof_ai.is_real(frame, face.bbox)
                
                if not liveness_check["real"]:
                    _draw_validation_box(display, face.bbox, "❌ SPOOFING", config.COLOR_RED)
                    _draw_stage_progress_bar(display, current_stage)
                    _draw_status_panel(display, current_stage, 
                                     "Spoofing Terdeteksi!", 
                                     "Gunakan wajah asli")
                    _put(display, "RESTART", config.FRAME_HEIGHT - 50, config.COLOR_RED,
                         scale=0.9, thickness=2)
                    _print_log("🚫 Spoofing Detected", "SECURITY")

                else:
                    # Liveness check
                    result = liveness.update_register(face, detector)
                    
                    # Map step to stage
                    step_to_stage = {
                        "FACEMESH": RegistrationStage.FACEMESH,
                        "YAW": RegistrationStage.YAW,
                        "PITCH": RegistrationStage.PITCH,
                        "ROLL": RegistrationStage.ROLL,
                        "BLINK": RegistrationStage.BLINK,
                        "DONE": RegistrationStage.EXTRACTION,
                    }
                    
                    current_stage = step_to_stage.get(result["step"], current_stage)
                    
                    # Determine box color
                    if "DONE" in result.get("progress", ""):
                        box_color = config.COLOR_GREEN
                        status_text = "✅ PASSED"
                    elif result["status"] == "pending":
                        box_color = config.COLOR_YELLOW
                        status_text = "⏳ WAIT"
                    else:
                        box_color = config.COLOR_CYAN
                        status_text = "🔍 CHECK"

                    _draw_validation_box(display, face.bbox, status_text, box_color)
                    _draw_stage_progress_bar(display, current_stage)
                    
                    # Extract details
                    details = ""
                    if "yaw" in result:
                        details += result["yaw"] + " "
                    if "pitch" in result:
                        details += result["pitch"] + " "
                    if "roll" in result:
                        details += result["roll"]
                    
                    _draw_status_panel(display, current_stage, 
                                     result["instruction"], 
                                     result.get("progress", ""),
                                     details)

                    # Log progress
                    if frame_count % 30 == 0:  # Log setiap 1 detik (30 fps)
                        _print_log(f"Step: {result['step']} | {result.get('progress', '')}", "PROGRESS")

                    # ──── JIKA LIVENESS SELESAI ────
                    if result["status"] == "complete" and not extraction_in_progress:
                        _print_log("✅ Semua tahap liveness berhasil!", "SUCCESS")
                        extraction_in_progress = True
                        
                        cv2.imshow("Register", display)
                        cv2.waitKey(1000)

                        # Extract embedding
                        face_crop = model.crop_face(frame, face.bbox)
                        embedding = model.get_embedding(face_crop)
                        save_face(name, embedding)
                        _print_log(f"✅ Embedding '{name}' disimpan", "SUCCESS")

                        # Success screen
                        current_stage = RegistrationStage.COMPLETE
                        display = frame.copy()
                        display = detector.draw(display, face)
                        _draw_validation_box(display, face.bbox, "✅ BERHASIL", config.COLOR_GREEN)
                        
                        cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                                    config.COLOR_GREEN, 12)
                        _put(display, "🎉 REGISTRASI BERHASIL!", 
                             config.FRAME_HEIGHT // 2 - 50, config.COLOR_GREEN, 
                             x=(config.FRAME_WIDTH - 400) // 2, scale=1.6, thickness=3)
                        _put(display, f"Nama: {name}", 
                             config.FRAME_HEIGHT // 2 + 10, config.COLOR_GREEN,
                             x=(config.FRAME_WIDTH - 150) // 2, scale=1.1, thickness=2)
                        _put(display, "Keluar dalam 5 detik...", 
                             config.FRAME_HEIGHT // 2 + 70, config.COLOR_WHITE,
                             x=(config.FRAME_WIDTH - 300) // 2, scale=0.8)
                        
                        cv2.imshow("Register", display)
                        cv2.waitKey(5000)
                        break

            cv2.imshow("Register", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
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
        cv2.destroyAllWindows()
        _print_log("Registrasi Selesai", "SYSTEM")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   SMART DOOR LOCK - FACE REGISTRATION (STABLE VALIDATION)")
    print("="*70)
    
    name = input("\n📝 Masukkan nama untuk registrasi: ").strip()
    
    if not name:
        print("❌ Nama tidak boleh kosong!")
        sys.exit(1)
    
    print(f"\n✅ Memulai registrasi untuk: '{name}'")
    print("🎥 Tekan 'Q' untuk membatalkan\n")
    
    run_register(name)