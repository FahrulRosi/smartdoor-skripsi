import sys
import cv2
from enum import Enum
import numpy as np

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from database.face_db           import save_face
from liveness.anti_spoofing     import SilentAntiSpoofing

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


# ─── REGISTRATION STAGES ──────────────────────────────────────────────────────
class RegistrationStage(Enum):
    IDLE       = 0
    FACEMESH   = 1
    YAW        = 2
    PITCH      = 3
    ROLL       = 4
    BLINK      = 5
    EXTRACTION = 6
    COMPLETE   = 7


STAGE_NAMES = {
    RegistrationStage.FACEMESH:   "1. FaceMesh (Deteksi Struktur Wajah)",
    RegistrationStage.YAW:        "2a. Liveness - Toleh Kiri & Kanan",
    RegistrationStage.PITCH:      "2b. Liveness - Ngangguk Atas & Bawah",
    RegistrationStage.ROLL:       "2c. Liveness - Miring Kiri & Kanan",
    RegistrationStage.BLINK:      "3. Liveness - Kedipkan Mata",
    RegistrationStage.EXTRACTION: "4. Ekstraksi MobileFaceNet",
}


# ─── FEATURE CAPTURE HELPERS ──────────────────────────────────────────────────

def _capture_facemesh_vector(face) -> np.ndarray | None:
    """Capture & normalize FaceMesh landmarks (468 points × 3 = 1404 dim)"""
    try:
        lm = face.landmarks
        if lm is None or len(lm) == 0:
            return None

        arr = np.array(lm, dtype=np.float32)        # shape: (468, 3)

        # Normalisasi: center + scale
        center = np.mean(arr, axis=0)
        arr = arr - center

        scale = np.max(np.linalg.norm(arr, axis=1)) + 1e-8
        arr = arr / scale

        return arr.flatten()                        # (1404,)
    except Exception as e:
        _print_log(f"Gagal capture facemesh vector: {e}", "WARNING")
        return None


def _capture_pose_snapshot(pose: dict, tag: str) -> dict:
    """Tanpa timestamp"""
    return {
        "tag":   tag,
        "yaw":   float(pose.get("yaw",   0.0)),
        "pitch": float(pose.get("pitch", 0.0)),
        "roll":  float(pose.get("roll",  0.0)),
    }


def _capture_blink_vector(face, detector) -> dict | None:
    try:
        LEFT_EYE  = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        lm = face.landmarks
        if lm is None or len(lm) < 400:
            return None

        def eye_aspect_ratio(indices):
            pts = np.array([lm[i][:2] for i in indices], dtype=np.float32)
            A = np.linalg.norm(pts[1] - pts[5])
            B = np.linalg.norm(pts[2] - pts[4])
            C = np.linalg.norm(pts[0] - pts[3])
            return (A + B) / (2.0 * C + 1e-6)

        left_ear  = eye_aspect_ratio(LEFT_EYE)
        right_ear = eye_aspect_ratio(RIGHT_EYE)
        avg_ear   = (left_ear + right_ear) / 2.0

        return {
            "left_ear":  float(left_ear),
            "right_ear": float(right_ear),
            "avg_ear":   float(avg_ear),
            "blink_detected": avg_ear < config.BLINK_EAR_THRESHOLD,
        }
    except Exception as e:
        _print_log(f"Gagal capture blink vector: {e}", "WARNING")
        return None


# ─── HUD HELPERS ──────────────────────────────────────────────────────────────

def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_stage_progress_bar(frame, current_stage, total_stages=6):
    bar_width  = 380
    bar_height = 28
    x = (config.FRAME_WIDTH - bar_width) // 2
    y = 15

    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (40, 40, 40), -1)
    progress = min((current_stage.value - 1) / total_stages, 1.0)
    progress_width = int(bar_width * progress)

    if progress_width > 0:
        cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height), config.COLOR_GREEN, -1)

    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), config.COLOR_WHITE, 2)

    text = f"Tahap {min(current_stage.value, 6)}/6"
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    t_x = x + (bar_width - t_size[0]) // 2
    t_y = y + (bar_height + t_size[1]) // 2
    cv2.putText(frame, text, (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _draw_status_panel(frame, stage, instruction, progress="", details=""):
    panel_h = 110
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_h), (25, 25, 35), -1)
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_h), config.COLOR_CYAN, 2)

    stage_name = STAGE_NAMES.get(stage, "Proses Registrasi")
    _put(frame, stage_name, 75, config.COLOR_GREEN, 20, scale=0.85, thickness=2)
    _put(frame, instruction, 105, config.COLOR_YELLOW, 20, scale=0.65, thickness=2)

    if progress:
        _put(frame, f"Status: {progress}", 130, config.COLOR_CYAN, 20, scale=0.6)
    if details:
        _put(frame, details, 150, config.COLOR_WHITE, 20, scale=0.55)


def _draw_validation_box(frame, face_bbox, status, color):
    x, y, w, h = face_bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    cl = 28
    cv2.line(frame, (x, y), (x + cl, y), color, 3)
    cv2.line(frame, (x, y), (x, y + cl), color, 3)
    cv2.line(frame, (x + w, y), (x + w - cl, y), color, 3)
    cv2.line(frame, (x + w, y), (x + w, y + cl), color, 3)

    cv2.rectangle(frame, (x, y - 35), (x + 190, y - 5), color, -1)
    cv2.putText(frame, status, (x + 8, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)


def _print_log(msg, level="INFO"):
    print(f"[{level}] {msg}")


# ─── MAIN REGISTRATION ────────────────────────────────────────────────────────

def run_register(name: str):
    _print_log(f"Memulai registrasi untuk: {name}", "SYSTEM")
    print("\n" + "="*80)
    print("          REGISTRASI WAJAH MULTI-FITUR")
    print("  FaceMesh → Pose Liveness → Blink → MobileFaceNet")
    print("="*80 + "\n")

    if GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.IR_CUT_PIN, GPIO.OUT)
            GPIO.output(config.IR_CUT_PIN, GPIO.HIGH)
        except Exception as e:
            _print_log(f"Gagal setup GPIO IR-CUT: {e}", "WARNING")

    cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()

    # Inisialisasi modul
    detector   = FaceMeshDetector(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    liveness   = LivenessManager()
    model      = MobileFaceNet()
    matcher    = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    anti_spoof = SilentAntiSpoofing()

    liveness.start_register()
    current_stage = RegistrationStage.FACEMESH
    extraction_in_progress = False

    # Container untuk semua data yang akan disimpan
    captured = {
        "facemesh_vector": None,
        "yaw_snapshots": [],
        "pitch_snapshots": [],
        "roll_snapshots": [],
        "blink_closed": None,
        "blink_open": None,
        "mobilefacenet_embedding": None,
    }

    _blink_was_closed = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            display = frame.copy()
            faces = detector.detect(frame)

            if not faces:
                cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.COLOR_RED, 5)
                _draw_stage_progress_bar(display, current_stage)
                _draw_status_panel(display, current_stage, "Wajah tidak terdeteksi", "Hadapkan wajah ke kamera")
            else:
                face = faces[0]
                display = detector.draw(display, face)

                # Anti-Spoofing Check
                spoof_res = anti_spoof.is_real(frame, face.bbox)
                if not spoof_res.get("real", True):
                    _draw_validation_box(display, face.bbox, "WAJAH PALSU!", config.COLOR_RED)
                    _draw_status_panel(display, current_stage, "Terdeteksi Spoofing!", f"Score: {spoof_res.get('score', 0):.3f}")
                    cv2.imshow("Register", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Tahap 1: Capture FaceMesh
                if captured["facemesh_vector"] is None:
                    vec = _capture_facemesh_vector(face)
                    if vec is not None:
                        captured["facemesh_vector"] = vec
                        _print_log(f"FaceMesh vector captured: shape={vec.shape}", "SUCCESS")

                # Liveness stages
                if current_stage != RegistrationStage.EXTRACTION and not extraction_in_progress:
                    result = liveness.update_register(face, detector)
                    new_step = result.get("step", "WAIT")

                    step_map = {
                        "FACEMESH": RegistrationStage.FACEMESH,
                        "YAW":      RegistrationStage.YAW,
                        "PITCH":    RegistrationStage.PITCH,
                        "ROLL":     RegistrationStage.ROLL,
                        "BLINK":    RegistrationStage.BLINK,
                        "DONE":     RegistrationStage.EXTRACTION,
                    }
                    current_stage = step_map.get(new_step, current_stage)

                    pose = liveness.pose_estimator.estimate(face, detector)

                    # Capture Pose Snapshots
                    if current_stage == RegistrationStage.YAW:
                        yaw = pose.get("yaw", 0.0)
                        snaps = captured["yaw_snapshots"]
                        if yaw < -config.YAW_THRESHOLD and not any(s["tag"] == "yaw_left" for s in snaps):
                            snaps.append(_capture_pose_snapshot(pose, "yaw_left"))
                            _print_log("YAW Left captured", "SUCCESS")
                        if yaw > config.YAW_THRESHOLD and not any(s["tag"] == "yaw_right" for s in snaps):
                            snaps.append(_capture_pose_snapshot(pose, "yaw_right"))
                            _print_log("YAW Right captured", "SUCCESS")

                    if current_stage == RegistrationStage.PITCH:
                        pitch = pose.get("pitch", 0.0)
                        snaps = captured["pitch_snapshots"]
                        if pitch > config.PITCH_THRESHOLD and not any(s["tag"] == "pitch_up" for s in snaps):
                            snaps.append(_capture_pose_snapshot(pose, "pitch_up"))
                            _print_log("PITCH Up captured", "SUCCESS")
                        if pitch < -config.PITCH_THRESHOLD and not any(s["tag"] == "pitch_down" for s in snaps):
                            snaps.append(_capture_pose_snapshot(pose, "pitch_down"))
                            _print_log("PITCH Down captured", "SUCCESS")

                    if current_stage == RegistrationStage.ROLL:
                        roll = pose.get("roll", 0.0)
                        snaps = captured["roll_snapshots"]
                        if roll < -config.ROLL_THRESHOLD and not any(s["tag"] == "roll_left" for s in snaps):
                            snaps.append(_capture_pose_snapshot(pose, "roll_left"))
                            _print_log("ROLL Left captured", "SUCCESS")
                        if roll > config.ROLL_THRESHOLD and not any(s["tag"] == "roll_right" for s in snaps):
                            snaps.append(_capture_pose_snapshot(pose, "roll_right"))
                            _print_log("ROLL Right captured", "SUCCESS")

                    # Capture Blink
                    if current_stage == RegistrationStage.BLINK:
                        blink_vec = _capture_blink_vector(face, detector)
                        if blink_vec:
                            if blink_vec["blink_detected"] and not _blink_was_closed:
                                captured["blink_closed"] = blink_vec
                                _blink_was_closed = True
                                _print_log(f"BLINK closed captured | EAR={blink_vec['avg_ear']:.3f}", "SUCCESS")
                            elif not blink_vec["blink_detected"] and _blink_was_closed:
                                captured["blink_open"] = blink_vec
                                _blink_was_closed = False
                                _print_log(f"BLINK open captured | EAR={blink_vec['avg_ear']:.3f}", "SUCCESS")

                    # Draw UI
                    status_text = "PASSED" if result.get("step") == "DONE" else "TAHAN LURUS" if result.get("step") == "WAIT" else "VALIDATING"
                    box_color = config.COLOR_GREEN if result.get("step") == "DONE" else config.COLOR_YELLOW if result.get("step") == "WAIT" else config.COLOR_CYAN

                    _draw_validation_box(display, face.bbox, status_text, box_color)
                    _draw_stage_progress_bar(display, current_stage)

                    details = f"Y:{pose.get('yaw',0):.1f} P:{pose.get('pitch',0):.1f} R:{pose.get('roll',0):.1f}"
                    _draw_status_panel(display, current_stage, result.get("instruction", ""), result.get("progress", ""), details)

                # Tahap Ekstraksi Embedding
                if current_stage == RegistrationStage.EXTRACTION and not extraction_in_progress:
                    pose = liveness.pose_estimator.estimate(face, detector)
                    yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

                    if (abs(yaw) < config.EXTRACTION_MAX_YAW and
                        abs(pitch) < config.EXTRACTION_MAX_PITCH and
                        abs(roll) < config.EXTRACTION_MAX_ROLL):

                        _print_log("Posisi wajah ideal → mulai ekstraksi embedding...", "SUCCESS")
                        extraction_in_progress = True

                        face_crop = model.crop_face(frame, face.bbox)
                        embedding = model.get_embedding(face_crop)

                        captured["mobilefacenet_embedding"] = embedding

                        # Validasi duplikat
                        match = matcher.match(embedding)
                        display = frame.copy()

                        if match.get("matched", False):
                            _draw_validation_box(display, face.bbox, "SUDAH TERDAFTAR", config.COLOR_RED)
                            cv2.rectangle(display, (0,0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.COLOR_RED, 12)
                            _put(display, "REGISTRASI DITOLAK", config.FRAME_HEIGHT//2 - 40, config.COLOR_RED, 
                                 x=config.FRAME_WIDTH//2 - 200, scale=1.5, thickness=3)
                            _put(display, f"Wajah sudah terdaftar sebagai: {match.get('name')}", 
                                 config.FRAME_HEIGHT//2 + 20, config.COLOR_RED, scale=0.9)
                        else:
                            save_face(name, embedding, captured)

                            _draw_validation_box(display, face.bbox, "BERHASIL", config.COLOR_GREEN)
                            cv2.rectangle(display, (0,0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.COLOR_GREEN, 15)
                            _put(display, "REGISTRASI BERHASIL!", config.FRAME_HEIGHT//2 - 60, config.COLOR_GREEN, 
                                 x=config.FRAME_WIDTH//2 - 220, scale=1.8, thickness=4)
                            _put(display, f"Nama: {name}", config.FRAME_HEIGHT//2 + 10, config.COLOR_GREEN, scale=1.2)

                        current_stage = RegistrationStage.COMPLETE
                        cv2.imshow("Register", display)
                        cv2.waitKey(5000)
                        break

                    else:
                        _draw_validation_box(display, face.bbox, "TAHAN LURUS", config.COLOR_YELLOW)
                        _draw_status_panel(display, current_stage, 
                                           "Tatap LURUS ke kamera", 
                                           "Menunggu posisi stabil...",
                                           f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}")

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                _print_log("Registrasi dibatalkan oleh user", "WARNING")
                break

    except Exception as e:
        _print_log(f"Error selama registrasi: {e}", "ERROR")
        import traceback
        traceback.print_exc()
    finally:
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()
        _print_log("Program registrasi selesai.", "SYSTEM")


if __name__ == "__main__":
    print("=== Sistem Registrasi Wajah Smart Door ===\n")
    name = input("Masukkan nama lengkap Anda: ").strip()
    if name:
        run_register(name)
    else:
        print("Nama tidak boleh kosong!")