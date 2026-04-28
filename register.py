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
from liveness.anti_spoofing     import SilentAntiSpoofing

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ─── REGISTRATION STAGES ──────────────────────────────────────────────────────
class RegistrationStage(Enum):
    IDLE       = 0
    FACEMESH   = 1   # 1. FaceMesh landmark capture
    YAW        = 2   # 2a. Liveness Yaw  (toleh kiri & kanan)
    PITCH      = 3   # 2b. Liveness Pitch (ngangguk atas & bawah)
    ROLL       = 4   # 2c. Liveness Roll  (miring kiri & kanan)
    BLINK      = 5   # 3. Liveness Blink  (kedipkan mata)
    EXTRACTION = 6   # 4. MobileFaceNet embedding
    COMPLETE   = 7

STAGE_NAMES = {
    RegistrationStage.FACEMESH:   "1. FaceMesh (Deteksi Struktur)",
    RegistrationStage.YAW:        "2a. Liveness (Toleh Kiri & Kanan)",
    RegistrationStage.PITCH:      "2b. Liveness (Ngangguk Atas & Bawah)",
    RegistrationStage.ROLL:       "2c. Liveness (Miring Kiri & Kanan)",
    RegistrationStage.BLINK:      "3. Liveness (Kedipkan Mata)",
    RegistrationStage.EXTRACTION: "4. Validasi MobileFaceNet",
}

# ─── FEATURE-CAPTURE HELPERS ──────────────────────────────────────────────────

def _capture_facemesh_vector(face) -> np.ndarray | None:
    """
    Tahap 1 – FaceMesh.
    Mengambil 468 landmark (x, y, z) dan meratakannya menjadi vektor 1404-dim.
    Dikembalikan sebagai float32 agar konsisten dengan embedding lain.
    """
    try:
        lm = face.landmarks  # list of (x, y, z) atau atribut sejenis
        if lm is None or len(lm) == 0:
            return None
        arr = np.array(lm, dtype=np.float32).flatten()  # (468, 3) → (1404,)
        return arr
    except Exception as e:
        _print_log(f"Gagal capture facemesh vector: {e}", "WARNING")
        return None


def _capture_pose_snapshot(pose: dict, tag: str) -> dict:
    """
    Tahap 2 – Pose (Yaw / Pitch / Roll).
    Menyimpan snapshot nilai sudut pada momen tertentu (misal saat toleh kiri).
    Mengembalikan dict kecil berisi tag + nilai sudut.
    """
    return {
        "tag":   tag,                          # mis. "yaw_left", "pitch_up", …
        "yaw":   float(pose.get("yaw",   0.0)),
        "pitch": float(pose.get("pitch", 0.0)),
        "roll":  float(pose.get("roll",  0.0)),
    }


def _capture_blink_vector(face, detector) -> dict | None:
    """
    Tahap 3 – Blink.
    Menghitung Eye Aspect Ratio (EAR) kiri dan kanan dari landmark mata,
    lalu menyimpannya beserta flag apakah terdeteksi blink.
    """
    try:
        # Indeks landmark mata (MediaPipe FaceMesh):
        # Mata kiri  (dari sudut pandang kamera): 33, 160, 158, 133, 153, 144
        # Mata kanan (dari sudut pandang kamera): 362, 385, 387, 263, 373, 380
        LEFT_EYE  = [33,  160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        lm = face.landmarks
        if lm is None or len(lm) < 400:
            return None

        def ear(indices):
            pts = np.array([lm[i][:2] for i in indices], dtype=np.float32)
            # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
            A = np.linalg.norm(pts[1] - pts[5])
            B = np.linalg.norm(pts[2] - pts[4])
            C = np.linalg.norm(pts[0] - pts[3])
            return (A + B) / (2.0 * C + 1e-6)

        left_ear  = ear(LEFT_EYE)
        right_ear = ear(RIGHT_EYE)
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
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _draw_stage_progress_bar(frame, current_stage, total_stages=6):
    bar_width  = 350
    bar_height = 25
    x = (config.FRAME_WIDTH - bar_width) // 2
    y = 15

    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (30, 30, 30), -1)

    stage_val = current_stage.value
    if stage_val > total_stages:
        stage_val = total_stages
    progress       = (stage_val - 1) / total_stages
    progress_width = int(bar_width * progress)
    if progress_width > 0:
        cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height), config.COLOR_GREEN, -1)

    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), config.COLOR_WHITE, 2)

    text   = f"Tahap {stage_val if stage_val <= 5 else 6}/6"
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    t_x    = x + (bar_width  - t_size[0]) // 2
    t_y    = y + (bar_height + t_size[1]) // 2
    cv2.putText(frame, text, (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _draw_status_panel(frame, stage, instruction, progress="", details=""):
    panel_height = 100
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_height), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 50 + panel_height), config.COLOR_CYAN, 2)

    stage_name = STAGE_NAMES.get(stage, "Proses...")
    _put(frame, stage_name,  75, config.COLOR_GREEN,  20, scale=0.85, thickness=2)
    _put(frame, instruction, 105, config.COLOR_YELLOW, 20, scale=0.65, thickness=2)
    if progress:
        _put(frame, f"Status: {progress}", 125, config.COLOR_CYAN,  20, scale=0.6)
    if details:
        _put(frame, details,               145, config.COLOR_WHITE, 20, scale=0.55)


def _draw_validation_box(frame, face_bbox, status, color):
    x, y, w, h = face_bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    corner_len = 25
    cv2.line(frame, (x,     y),     (x + corner_len, y),            color, 3)
    cv2.line(frame, (x,     y),     (x,              y + corner_len), color, 3)
    cv2.line(frame, (x + w, y),     (x + w - corner_len, y),         color, 3)
    cv2.line(frame, (x + w, y),     (x + w,          y + corner_len), color, 3)

    status_bg_h = 30
    cv2.rectangle(frame, (x, y - status_bg_h - 5), (x + 180, y - 5), color, -1)
    cv2.putText(frame, status, (x + 5, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def _print_log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


# ─── MAIN REGISTRATION FUNCTION ───────────────────────────────────────────────

def run_register(name):
    _print_log(f"Memulai registrasi untuk: {name}", "SYSTEM")
    print("\n" + "=" * 70)
    print("  MEMULAI REGISTRASI WAJAH (ALUR 4 TAHAP + CAPTURE VEKTOR TIAP TAHAP)")
    print("  1. FaceMesh | 2. Liveness Pose | 3. Liveness Blink | 4. MobileFaceNet")
    print("=" * 70 + "\n")

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

    # ── Inisialisasi modul ───────────────────────────────────────────────────
    detector   = FaceMeshDetector(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    liveness   = LivenessManager()
    model      = MobileFaceNet()
    matcher    = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    anti_spoof = SilentAntiSpoofing()

    liveness.start_register()
    current_stage        = RegistrationStage.FACEMESH
    extraction_in_progress = False

    # ── Wadah untuk menampung semua vektor per tahap ─────────────────────────
    captured = {
        # Tahap 1 – FaceMesh
        "facemesh_vector":    None,          # np.ndarray (1404,) float32

        # Tahap 2 – Pose
        "yaw_snapshots":      [],            # list of dict {"tag","yaw","pitch","roll"}
        "pitch_snapshots":    [],
        "roll_snapshots":     [],

        # Tahap 3 – Blink
        "blink_closed":       None,          # dict EAR saat mata menutup
        "blink_open":         None,          # dict EAR saat mata terbuka kembali

        # Tahap 4 – MobileFaceNet (embedding utama untuk recognition)
        "mobilefacenet_embedding": None,     # np.ndarray (512,) float32
    }

    # Flag internal untuk mendeteksi transisi yaw/pitch/roll
    _prev_liveness_step = "FACEMESH"
    _blink_was_closed   = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            display = frame.copy()

            # ── TAHAP 1: FaceMesh Detector ───────────────────────────────────
            faces = detector.detect(frame)

            if not faces:
                cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                              config.COLOR_RED, 4)
                _draw_stage_progress_bar(display, current_stage)
                _draw_status_panel(display, current_stage,
                                   "Wajah tidak terdeteksi",
                                   "Silahkan hadapkan wajah ke kamera")
            else:
                face = faces[0]
                display = detector.draw(display, face)

                # ── Anti-Spoofing ─────────────────────────────────────────────
                spoof_res = anti_spoof.is_real(frame, face.bbox)
                if not spoof_res["real"]:
                    _draw_validation_box(display, face.bbox, "WAJAH PALSU!", config.COLOR_RED)
                    _draw_status_panel(display, current_stage,
                                       "Terdeteksi Foto/Video!",
                                       f"Skor: {spoof_res['score']}",
                                       "Gunakan wajah asli!")
                    cv2.imshow("Register", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        _print_log("Registrasi dibatalkan oleh user", "WARNING")
                        break
                    continue

                # ── TAHAP 1 capture: ambil FaceMesh vector sekali saja ────────
                if captured["facemesh_vector"] is None:
                    vec = _capture_facemesh_vector(face)
                    if vec is not None:
                        captured["facemesh_vector"] = vec
                        _print_log(f"[CAPTURE] FaceMesh vector: shape={vec.shape}, "
                                   f"mean={vec.mean():.4f}", "SUCCESS")

                # ── TAHAP 2 & 3: Liveness (Pose + Blink) ─────────────────────
                if current_stage != RegistrationStage.EXTRACTION and not extraction_in_progress:
                    result = liveness.update_register(face, detector)

                    step_to_stage = {
                        "FACEMESH": RegistrationStage.FACEMESH,
                        "YAW":      RegistrationStage.YAW,
                        "PITCH":    RegistrationStage.PITCH,
                        "ROLL":     RegistrationStage.ROLL,
                        "BLINK":    RegistrationStage.BLINK,
                        "DONE":     RegistrationStage.EXTRACTION,
                    }

                    new_step = result.get("step", "WAIT")
                    if new_step != "WAIT":
                        current_stage = step_to_stage.get(new_step, current_stage)

                    # Dapatkan nilai pose saat ini untuk capture
                    pose = liveness.pose_estimator.estimate(face, detector)

                    # ── Capture YAW snapshots ─────────────────────────────────
                    # Kita tangkap satu snapshot saat stage pertama kali masuk YAW
                    # dan satu snapshot saat selesai YAW (saat bergerak dari YAW→PITCH)
                    if current_stage == RegistrationStage.YAW:
                        yaw_val = pose.get("yaw", 0.0)
                        snapshots = captured["yaw_snapshots"]

                        # Snapshot saat toleh kiri (yaw < -threshold)
                        if (yaw_val < -config.YAW_THRESHOLD and
                                not any(s["tag"] == "yaw_left" for s in snapshots)):
                            snap = _capture_pose_snapshot(pose, "yaw_left")
                            snapshots.append(snap)
                            _print_log(f"[CAPTURE] YAW kiri: {snap}", "SUCCESS")

                        # Snapshot saat toleh kanan (yaw > +threshold)
                        if (yaw_val > config.YAW_THRESHOLD and
                                not any(s["tag"] == "yaw_right" for s in snapshots)):
                            snap = _capture_pose_snapshot(pose, "yaw_right")
                            snapshots.append(snap)
                            _print_log(f"[CAPTURE] YAW kanan: {snap}", "SUCCESS")

                    # ── Capture PITCH snapshots ───────────────────────────────
                    if current_stage == RegistrationStage.PITCH:
                        pitch_val = pose.get("pitch", 0.0)
                        snapshots = captured["pitch_snapshots"]

                        if (pitch_val > config.PITCH_THRESHOLD and
                                not any(s["tag"] == "pitch_up" for s in snapshots)):
                            snap = _capture_pose_snapshot(pose, "pitch_up")
                            snapshots.append(snap)
                            _print_log(f"[CAPTURE] PITCH atas: {snap}", "SUCCESS")

                        if (pitch_val < -config.PITCH_THRESHOLD and
                                not any(s["tag"] == "pitch_down" for s in snapshots)):
                            snap = _capture_pose_snapshot(pose, "pitch_down")
                            snapshots.append(snap)
                            _print_log(f"[CAPTURE] PITCH bawah: {snap}", "SUCCESS")

                    # ── Capture ROLL snapshots ────────────────────────────────
                    if current_stage == RegistrationStage.ROLL:
                        roll_val = pose.get("roll", 0.0)
                        snapshots = captured["roll_snapshots"]

                        if (roll_val < -config.ROLL_THRESHOLD and
                                not any(s["tag"] == "roll_left" for s in snapshots)):
                            snap = _capture_pose_snapshot(pose, "roll_left")
                            snapshots.append(snap)
                            _print_log(f"[CAPTURE] ROLL kiri: {snap}", "SUCCESS")

                        if (roll_val > config.ROLL_THRESHOLD and
                                not any(s["tag"] == "roll_right" for s in snapshots)):
                            snap = _capture_pose_snapshot(pose, "roll_right")
                            snapshots.append(snap)
                            _print_log(f"[CAPTURE] ROLL kanan: {snap}", "SUCCESS")

                    # ── Capture BLINK vectors ─────────────────────────────────
                    if current_stage == RegistrationStage.BLINK:
                        blink_vec = _capture_blink_vector(face, detector)
                        if blink_vec is not None:
                            if blink_vec["blink_detected"] and not _blink_was_closed:
                                # Mata baru saja menutup
                                captured["blink_closed"] = blink_vec
                                _blink_was_closed = True
                                _print_log(f"[CAPTURE] BLINK closed: EAR={blink_vec['avg_ear']:.3f}",
                                           "SUCCESS")

                            if not blink_vec["blink_detected"] and _blink_was_closed:
                                # Mata kembali terbuka setelah blink
                                captured["blink_open"] = blink_vec
                                _blink_was_closed = False
                                _print_log(f"[CAPTURE] BLINK open: EAR={blink_vec['avg_ear']:.3f}",
                                           "SUCCESS")

                    # ── HUD ──────────────────────────────────────────────────
                    if result["step"] == "DONE" or "DONE" in result.get("progress", ""):
                        box_color   = config.COLOR_GREEN
                        status_text = "PASSED"
                    elif result["step"] == "WAIT":
                        box_color   = config.COLOR_YELLOW
                        status_text = "TAHAN LURUS"
                    else:
                        box_color   = config.COLOR_CYAN
                        status_text = "VALIDATING"

                    _draw_validation_box(display, face.bbox, status_text, box_color)
                    _draw_stage_progress_bar(display, current_stage)

                    details = ""
                    if "yaw"   in result: details += f"YAW: {result['yaw']} "
                    if "pitch" in result: details += f"PITCH: {result['pitch']} "
                    if "roll"  in result: details += f"ROLL: {result['roll']}"
                    _draw_status_panel(display, current_stage,
                                       result["instruction"],
                                       result.get("progress", ""),
                                       details)

                # ── TAHAP 4: MobileFaceNet – ekstraksi & validasi ─────────────
                if current_stage == RegistrationStage.EXTRACTION and not extraction_in_progress:
                    pose     = liveness.pose_estimator.estimate(face, detector)
                    yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

                    if (abs(yaw)   < config.EXTRACTION_MAX_YAW and
                            abs(pitch) < config.EXTRACTION_MAX_PITCH and
                            abs(roll)  < config.EXTRACTION_MAX_ROLL):

                        _print_log("Posisi wajah ideal. MobileFaceNet mulai memvalidasi…", "SUCCESS")
                        extraction_in_progress = True

                        face_crop = model.crop_face(frame, face.bbox)
                        embedding = model.get_embedding(face_crop)

                        # Simpan embedding MobileFaceNet ke wadah capture
                        captured["mobilefacenet_embedding"] = embedding
                        _print_log(f"[CAPTURE] MobileFaceNet embedding: shape={embedding.shape}, "
                                   f"norm={np.linalg.norm(embedding):.4f}", "SUCCESS")

                        # Cetak ringkasan semua data yang berhasil ditangkap
                        _print_log("=" * 50, "SYSTEM")
                        _print_log("RINGKASAN DATA YANG DITANGKAP:", "SYSTEM")
                        _print_log(f"  FaceMesh vector     : {'OK' if captured['facemesh_vector'] is not None else 'GAGAL'} "
                                   f"({captured['facemesh_vector'].shape if captured['facemesh_vector'] is not None else '-'})",
                                   "SYSTEM")
                        _print_log(f"  YAW snapshots       : {len(captured['yaw_snapshots'])} snapshot(s) "
                                   f"{[s['tag'] for s in captured['yaw_snapshots']]}",
                                   "SYSTEM")
                        _print_log(f"  PITCH snapshots     : {len(captured['pitch_snapshots'])} snapshot(s) "
                                   f"{[s['tag'] for s in captured['pitch_snapshots']]}",
                                   "SYSTEM")
                        _print_log(f"  ROLL snapshots      : {len(captured['roll_snapshots'])} snapshot(s) "
                                   f"{[s['tag'] for s in captured['roll_snapshots']]}",
                                   "SYSTEM")
                        _print_log(f"  BLINK closed EAR    : "
                                   f"{captured['blink_closed']['avg_ear']:.3f if captured['blink_closed'] else 'GAGAL'}",
                                   "SYSTEM")
                        _print_log(f"  BLINK open  EAR     : "
                                   f"{captured['blink_open']['avg_ear']:.3f if captured['blink_open'] else 'GAGAL'}",
                                   "SYSTEM")
                        _print_log(f"  MobileFaceNet emb   : shape={embedding.shape}", "SYSTEM")
                        _print_log("=" * 50, "SYSTEM")

                        # ── Validasi ganda: cek duplikat ──────────────────────
                        match   = matcher.match(embedding)
                        display = frame.copy()

                        if match["matched"]:
                            _draw_validation_box(display, face.bbox, "SUDAH TERDAFTAR", config.COLOR_RED)
                            cv2.rectangle(display, (0, 0),
                                          (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                                          config.COLOR_RED, 12)
                            _put(display, "REGISTRASI DITOLAK!",
                                 config.FRAME_HEIGHT // 2 - 50, config.COLOR_RED,
                                 x=(config.FRAME_WIDTH - 350) // 2, scale=1.4, thickness=3)
                            _put(display, f"Wajah sudah terdaftar: {match['name']}",
                                 config.FRAME_HEIGHT // 2 + 10, config.COLOR_RED,
                                 x=(config.FRAME_WIDTH - 500) // 2, scale=0.8, thickness=2)
                            _print_log(f"Wajah ditolak – terdeteksi sebagai {match['name']}.", "WARNING")
                        else:
                            # ── Simpan semua data yang telah dikumpulkan ──────
                            save_face(name, embedding, captured)

                            _draw_validation_box(display, face.bbox, "BERHASIL", config.COLOR_GREEN)
                            cv2.rectangle(display, (0, 0),
                                          (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                                          config.COLOR_GREEN, 12)
                            _put(display, "REGISTRASI BERHASIL!",
                                 config.FRAME_HEIGHT // 2 - 50, config.COLOR_GREEN,
                                 x=(config.FRAME_WIDTH - 400) // 2, scale=1.6, thickness=3)
                            _put(display, f"Nama: {name}",
                                 config.FRAME_HEIGHT // 2 + 10, config.COLOR_GREEN,
                                 x=(config.FRAME_WIDTH - 150) // 2, scale=1.1, thickness=2)
                            _put(display, "Data wajah lengkap telah disimpan.",
                                 config.FRAME_HEIGHT // 2 + 50, config.COLOR_WHITE,
                                 x=(config.FRAME_WIDTH - 400) // 2, scale=0.7)

                        current_stage = RegistrationStage.COMPLETE
                        cv2.imshow("Register", display)
                        cv2.waitKey(4000)
                        break

                    else:
                        _draw_validation_box(display, face.bbox, "TAHAN LURUS", config.COLOR_YELLOW)
                        _draw_stage_progress_bar(display, current_stage)
                        _draw_status_panel(display, current_stage,
                                           "Tatap LURUS ke kamera untuk proses MobileFaceNet",
                                           "Menunggu wajah kembali lurus…",
                                           f"Y: {yaw:.1f}  P: {pitch:.1f}  R: {roll:.1f}")

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _print_log("Registrasi dibatalkan oleh user", "WARNING")
                break

    except Exception as e:
        _print_log(f"Terjadi Kesalahan: {e}", "ERROR")
        import traceback; traceback.print_exc()
    finally:
        if GPIO_AVAILABLE: GPIO.cleanup()
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()
        _print_log("Program Selesai.", "SYSTEM")


if __name__ == "__main__":
    name = input("\n Masukkan nama Anda: ").strip()
    if name:
        run_register(name)
    else:
        print("Nama tidak boleh kosong!")