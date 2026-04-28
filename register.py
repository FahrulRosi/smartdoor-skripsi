import sys
import cv2
from datetime import datetime
from enum import Enum
import numpy as np

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher  
from database.face_db           import save_face

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ─── REGISTRATION STAGES ──────────────────────────────────────────────────────
class RegistrationStage(Enum):
    IDLE = 0
    FACEMESH = 1      # 1. FaceMesh
    YAW = 2           # 2a. Liveness Yaw
    PITCH = 3         # 2b. Liveness Pitch
    ROLL = 4          # 2c. Liveness Roll
    BLINK = 5         # 3. Liveness Blink
    EXTRACTION = 6    # 4. MobileFaceNet Validation
    COMPLETE = 7

STAGE_NAMES = {
    RegistrationStage.FACEMESH: "1. FaceMesh (Deteksi Struktur)",
    RegistrationStage.YAW: "2. Liveness (Toleh Kiri & Kanan)",
    RegistrationStage.PITCH: "2. Liveness (Angkat & Tunduk)",
    RegistrationStage.ROLL: "2. Liveness (Miring Kepala)",
    RegistrationStage.BLINK: "3. Liveness (Kedipkan Mata)",
    RegistrationStage.EXTRACTION: "4. Validasi MobileFaceNet",
}

# ─── HUD HELPERS ──────────────────────────────────────────────────────────────
def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def _draw_stage_progress_bar(frame, current_stage, total_stages=6):
    bar_width = 350
    bar_height = 25
    x = (config.FRAME_WIDTH - bar_width) // 2
    y = 15
    
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (30, 30, 30), -1)
    
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
    panel_height = 100
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_height), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_height), config.COLOR_CYAN, 2)
    
    stage_name = STAGE_NAMES.get(stage, "Proses...")
    _put(frame, stage_name, 75, config.COLOR_GREEN, 20, scale=0.85, thickness=2)
    _put(frame, instruction, 105, config.COLOR_YELLOW, 20, scale=0.65, thickness=2)
    
    if progress:
        _put(frame, f"Status: {progress}", 125, config.COLOR_CYAN, 20, scale=0.6)
    if details:
        _put(frame, details, 145, config.COLOR_WHITE, 20, scale=0.55)

def _draw_validation_box(frame, face_bbox, status, color):
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
    print("  MEMULAI REGISTRASI WAJAH (ALUR 4 TAHAP)")
    print("  1. FaceMesh | 2. Liveness Pose | 3. Liveness Blink | 4. Validasi MobileFaceNet")
    print("="*70 + "\n")

    if GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.IR_CUT_PIN, GPIO.OUT)
            GPIO.output(config.IR_CUT_PIN, GPIO.HIGH)
        except Exception as e:
            _print_log(f"Gagal setup IR-CUT: {e}", "WARNING")

    try:
        cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    except Exception as e:
        _print_log(f"Gagal inisialisasi camera: {e}", "ERROR")
        return

    # Inisialisasi Modul 
    detector = FaceMeshDetector(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE, 
                                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE)
    liveness = LivenessManager()
    model = MobileFaceNet()
    matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD) 

    liveness.start_register()
    current_stage = RegistrationStage.FACEMESH
    extraction_in_progress = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            
            display = frame.copy()
            
            # TAHAP 1: FaceMesh Detector mendeteksi struktur wajah
            faces = detector.detect(frame)

            if not faces:
                cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.COLOR_RED, 4)
                _draw_stage_progress_bar(display, current_stage)
                _draw_status_panel(display, current_stage, "🔍 Wajah tidak terdeteksi", "Silahkan hadapkan wajah ke kamera")
            else:
                face = faces[0]
                
                # MENGGAMBAR FACEMESH: Baris ini diaktifkan kembali
                display = detector.draw(display, face) 
                
                # TAHAP 2 & 3: Liveness Detection (Yaw, Pitch, Roll) & Blink
                if current_stage != RegistrationStage.EXTRACTION and not extraction_in_progress:
                    result = liveness.update_register(face, detector)
                    
                    step_to_stage = {
                        "FACEMESH": RegistrationStage.FACEMESH,
                        "YAW": RegistrationStage.YAW,
                        "PITCH": RegistrationStage.PITCH,
                        "ROLL": RegistrationStage.ROLL,
                        "BLINK": RegistrationStage.BLINK,
                        "DONE": RegistrationStage.EXTRACTION,
                    }
                    
                    if result["step"] != "WAIT":
                        current_stage = step_to_stage.get(result["step"], current_stage)
                    
                    if result["step"] == "DONE" or "DONE" in result.get("progress", ""):
                        box_color = config.COLOR_GREEN
                        status_text = "✅ PASSED"
                    elif result["step"] == "WAIT":
                        box_color = config.COLOR_YELLOW  
                        status_text = "⏳ TAHAN LURUS"
                    else:
                        box_color = config.COLOR_CYAN
                        status_text = "🔍 VALIDATING"

                    _draw_validation_box(display, face.bbox, status_text, box_color)
                    _draw_stage_progress_bar(display, current_stage)
                    
                    details = ""
                    if "yaw" in result: details += f"YAW: {result['yaw']} "
                    if "pitch" in result: details += f"PITCH: {result['pitch']} "
                    if "roll" in result: details += f"ROLL: {result['roll']}"
                    
                    _draw_status_panel(display, current_stage, result["instruction"], result.get("progress", ""), details)

                # TAHAP 4: MobileFaceNet (Ekstraksi & Validasi Data Masuk)
                if current_stage == RegistrationStage.EXTRACTION and not extraction_in_progress:
                    pose = liveness.pose_estimator.estimate(face, detector)
                    yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
                    
                    # Wajah harus benar-benar lurus menghadap kamera untuk kualitas database yang baik
                    if abs(yaw) < config.EXTRACTION_MAX_YAW and abs(pitch) < config.EXTRACTION_MAX_PITCH and abs(roll) < config.EXTRACTION_MAX_ROLL:
                        _print_log("Posisi wajah ideal. MobileFaceNet mulai memvalidasi...", "SUCCESS")
                        extraction_in_progress = True
                        
                        face_crop = model.crop_face(frame, face.bbox)
                        embedding = model.get_embedding(face_crop)
                        
                        # Validasi ganda: Memeriksa apakah wajah sudah terdaftar di database
                        match = matcher.match(embedding)
                        
                        display = frame.copy()
                        
                        if match["matched"]:
                            _draw_validation_box(display, face.bbox, "❌ SUDAH TERDAFTAR", config.COLOR_RED)
                            cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.COLOR_RED, 12)
                            _put(display, "REGISTRASI DITOLAK!", config.FRAME_HEIGHT // 2 - 50, config.COLOR_RED, x=(config.FRAME_WIDTH - 350) // 2, scale=1.4, thickness=3)
                            _put(display, f"Wajah ini sudah terdaftar atas nama: {match['name']}", config.FRAME_HEIGHT // 2 + 10, config.COLOR_RED, x=(config.FRAME_WIDTH - 500) // 2, scale=0.8, thickness=2)
                            _print_log(f"Wajah ditolak. Wajah ini terdeteksi sebagai {match['name']}.", "WARNING")
                        else:
                            save_face(name, embedding)
                            _draw_validation_box(display, face.bbox, "✅ BERHASIL", config.COLOR_GREEN)
                            cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.COLOR_GREEN, 12)
                            _put(display, "🎉 REGISTRASI BERHASIL!", config.FRAME_HEIGHT // 2 - 50, config.COLOR_GREEN, x=(config.FRAME_WIDTH - 400) // 2, scale=1.6, thickness=3)
                            _put(display, f"Nama: {name}", config.FRAME_HEIGHT // 2 + 10, config.COLOR_GREEN, x=(config.FRAME_WIDTH - 150) // 2, scale=1.1, thickness=2)
                            _put(display, "Data wajah baru telah divalidasi dan disimpan.", config.FRAME_HEIGHT // 2 + 50, config.COLOR_WHITE, x=(config.FRAME_WIDTH - 400) // 2, scale=0.7)
                        
                        current_stage = RegistrationStage.COMPLETE
                        cv2.imshow("Register", display)
                        cv2.waitKey(4000) 
                        break
                    else:
                        _draw_validation_box(display, face.bbox, "TAHAN LURUS", config.COLOR_YELLOW)
                        _draw_stage_progress_bar(display, current_stage)
                        _draw_status_panel(display, current_stage, "Tatap LURUS ke kamera untuk proses MobileFaceNet", "Menunggu wajah kembali lurus...", f"Y: {yaw:.1f} P: {pitch:.1f} R: {roll:.1f}")

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _print_log("Registrasi dibatalkan oleh user", "WARNING")
                break

    except Exception as e:
        _print_log(f"Terjadi Kesalahan: {e}", "ERROR")
    finally:
        if GPIO_AVAILABLE: GPIO.cleanup()
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()
        _print_log("Program Selesai.", "SYSTEM")

if __name__ == "__main__":
    name = input("\n📝 Masukkan nama Anda: ").strip()
    if name: run_register(name)
    else: print("❌ Nama tidak boleh kosong!")