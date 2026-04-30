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
# --- UBAH IMPORT KE KELAS FIREBASE ---
from database.face_db           import FaceDatabase
from liveness.anti_spoofing     import SilentAntiSpoofing

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ─── STAGES ───────────────────────────────────────────────────────────────────
class RegistrationStage(Enum):
    IDLE=0; FACEMESH=1; YAW=2; PITCH=3; ROLL=4; BLINK=5; EXTRACTION=6; COMPLETE=7

STAGE_NAMES = {
    RegistrationStage.FACEMESH:   "1. FaceMesh (Deteksi Struktur 3D)",
    RegistrationStage.YAW:        "2a. Liveness (Toleh Kiri & Kanan)",
    RegistrationStage.PITCH:      "2b. Liveness (Ngangguk Atas & Bawah)",
    RegistrationStage.ROLL:       "2c. Liveness (Miring Kiri & Kanan)",
    RegistrationStage.BLINK:      "3. Liveness (Kedipkan Mata)",
    RegistrationStage.EXTRACTION: "4. Validasi MobileFaceNet",
}

STEP_TO_STAGE = {
    "FACEMESH": RegistrationStage.FACEMESH, "YAW": RegistrationStage.YAW,
    "PITCH":    RegistrationStage.PITCH,    "ROLL": RegistrationStage.ROLL,
    "BLINK":    RegistrationStage.BLINK,    "DONE": RegistrationStage.EXTRACTION,
}

# Indeks landmark MediaPipe untuk EAR blink
_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ─── CAPTURE HELPERS ──────────────────────────────────────────────────────────
def _capture_facemesh(face):
    """
    Diperbarui untuk mengambil sumbu Z agar struktur kedalaman wajah 
    (3D structure) bisa terdeteksi dengan akurat.
    """
    try:
        lm = face.landmarks
        if lm is None or len(lm) == 0:
            return None
        
        # Menggunakan .x, .y, dan .z dari NormalizedLandmark
        points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in lm], 
                         dtype=np.float32)
        return points.flatten()  # 1404 dimensi (468 * 3)
    except Exception as e:
        _log(f"Gagal capture facemesh: {e}", "WARNING")
        return None

def _capture_pose(pose, tag):
    return {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")} | {"tag": tag}

def _capture_blink(face):
    try:
        lm = face.landmarks
        if lm is None or len(lm) < 400:
            return None

        def ear(eye_indices):
            points = np.array([[lm[i].x, lm[i].y] for i in eye_indices], 
                            dtype=np.float32)
            
            vertical1 = np.linalg.norm(points[1] - points[5])
            vertical2 = np.linalg.norm(points[2] - points[4])
            horizontal = np.linalg.norm(points[0] - points[3])
            
            return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)

        l = ear(_LEFT_EYE)
        r = ear(_RIGHT_EYE)
        avg = (l + r) / 2.0

        return {
            "left_ear":  float(l),
            "right_ear": float(r),
            "avg_ear":   float(avg)
        }
    except Exception as e:
        _log(f"Gagal capture blink: {e}", "WARNING")
        return None


# ─── HUD HELPERS ──────────────────────────────────────────────────────────────
def _put(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def _draw_progress_bar(frame, stage, total=6):
    W, bw, bh = config.FRAME_WIDTH, 350, 25
    x, y = (W - bw) // 2, 15
    sv   = min(stage.value, total)
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (30,30,30), -1)
    pw = int(bw * (sv-1) / total) if sv > 0 else 0
    if pw > 0: 
        cv2.rectangle(frame, (x, y), (x+pw, y+bh), config.COLOR_GREEN, -1)
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), config.COLOR_WHITE, 2)
    txt  = f"Tahap {sv if sv<=5 else 6}/6"
    ts   = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    cv2.putText(frame, txt, (x+(bw-ts[0])//2, y+(bh+ts[1])//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

def _draw_panel(frame, stage, instruction, progress="", details=""):
    cv2.rectangle(frame, (0,50), (config.FRAME_WIDTH, 150), (20,20,20), -1)
    cv2.rectangle(frame, (0,50), (config.FRAME_WIDTH, 150), config.COLOR_CYAN, 2)
    _put(frame, STAGE_NAMES.get(stage,"Proses..."), 75,  config.COLOR_GREEN,  20, 0.85, 2)
    _put(frame, instruction,                        105, config.COLOR_YELLOW, 20, 0.65, 2)
    if progress: _put(frame, f"Status: {progress}", 125, config.COLOR_CYAN,  20, 0.6)
    if details:  _put(frame, details,               145, config.COLOR_WHITE, 20, 0.55)

def _draw_box(frame, bbox, status, color):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 3)
    cl = 25
    for (a,b),(c,d) in [((x,y),(x+cl,y)),((x,y),(x,y+cl)),((x+w,y),(x+w-cl,y)),((x+w,y),(x+w,y+cl))]:
        cv2.line(frame, (a,b), (c,d), color, 3)
    cv2.rectangle(frame, (x,y-35), (x+180,y-5), color, -1)
    cv2.putText(frame, status, (x+5,y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

def _log(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_register(name):
    _log(f"Memulai registrasi: {name}", "SYSTEM")

    # ─── INISIALISASI DATABASE FIREBASE ───
    db_url = "https://smart-door-lock-feb6b-default-rtdb.asia-southeast1.firebasedatabase.app/"
    credentials_path = "serviceAccount.json"
    try:
        _log("Menghubungkan ke Firebase...", "SYSTEM")
        face_db = FaceDatabase(db_url, credentials_path)
        
        # Cek nama di Firebase sebelum menyalakan kamera
        if face_db.check_user_exists(name):
            _log(f"Registrasi Dibatalkan: Nama '{name}' sudah terdaftar di Database!", "ERROR")
            print("Silakan gunakan nama lain.")
            return
    except Exception as e:
        _log(f"Gagal koneksi ke Firebase: {e}", "ERROR")
        return
    # ─────────────────────────────────────────

    if GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(config.IR_CUT_PIN, GPIO.OUT)
            GPIO.output(config.IR_CUT_PIN, GPIO.HIGH)
        except Exception as e:
            _log(f"Gagal IR-CUT: {e}", "WARNING")

    try:
        cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    except Exception as e:
        _log(f"Gagal kamera: {e}", "ERROR")
        return

    detector = FaceMeshDetector(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE, 
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
    )
    liveness   = LivenessManager()
    model      = MobileFaceNet()
    matcher    = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    anti_spoof = SilentAntiSpoofing()
    liveness.start_register()

    stage  = RegistrationStage.FACEMESH
    in_ext = False

    # Wadah hasil capture tiap tahap
    cap = {
        "facemesh_vector":         None,   
        "yaw_snapshots":           [],     
        "pitch_snapshots":         [],     
        "roll_snapshots":          [],     
        "blink_closed":            None,   
        "blink_open":              None,
        "mobilefacenet_embedding": None,   
    }

    _pose_buf         = {"yaw": {}, "pitch": {}, "roll": {}}
    _blink_buf        = {"closed": None, "open": None}
    _prev_step        = "FACEMESH"  

    POSE_CFG = {
        RegistrationStage.YAW:   ("yaw_snapshots",   "yaw_left",  "yaw_right",  "yaw",   config.YAW_THRESHOLD),
        RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up",  "pitch_down", "pitch", config.PITCH_THRESHOLD),
        RegistrationStage.ROLL:  ("roll_snapshots",  "roll_left", "roll_right", "roll",  config.ROLL_THRESHOLD),
    }
    STEP_COMMIT = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            display = frame.copy()
            faces   = detector.detect(frame)

            if not faces:
                cv2.rectangle(display,(0,0),(config.FRAME_WIDTH,config.FRAME_HEIGHT),config.COLOR_RED,4)
                _draw_panel(display, stage, "Wajah tidak terdeteksi", "Hadapkan wajah ke kamera")
            else:
                face    = faces[0]
                display = detector.draw(display, face)

                # Anti-spoofing
                spoof = anti_spoof.is_real(frame, face.bbox)
                if not spoof["real"]:
                    _draw_box(display, face.bbox, "WAJAH PALSU!", config.COLOR_RED)
                    _draw_panel(display, stage, "Terdeteksi Foto/Video!", f"Skor: {spoof['score']}", "Gunakan wajah asli!")
                    cv2.imshow("Register", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
                    continue

                if stage != RegistrationStage.EXTRACTION and not in_ext:
                    pose = liveness.pose_estimator.estimate(face, detector)

                    # ─── 1. CAPTURE DATA DULU ───
                    if stage == RegistrationStage.FACEMESH:
                        if cap["facemesh_vector"] is None:
                            fm = _capture_facemesh(face)
                            if fm is not None:
                                cap["facemesh_vector"] = fm
                                _log(f"[COMMIT] FaceMesh: {len(fm)} dimensi", "SUCCESS")

                    if stage in POSE_CFG:
                        _, tag_neg, tag_pos, axis, thr = POSE_CFG[stage]
                        val = pose.get(axis, 0.0)
                        buf = _pose_buf[axis]
                        cap_thr = thr * 0.7  

                        if val < -cap_thr:
                            if tag_neg not in buf or val < buf[tag_neg][axis]:
                                buf[tag_neg] = _capture_pose(pose, tag_neg)
                        if val > cap_thr:
                            if tag_pos not in buf or val > buf[tag_pos][axis]:
                                buf[tag_pos] = _capture_pose(pose, tag_pos)

                    if stage == RegistrationStage.BLINK:
                        bv = _capture_blink(face)
                        if bv:
                            if _blink_buf["closed"] is None or bv["avg_ear"] < _blink_buf["closed"]["avg_ear"]:
                                _blink_buf["closed"] = bv
                            if _blink_buf["open"] is None or bv["avg_ear"] > _blink_buf["open"]["avg_ear"]:
                                _blink_buf["open"] = bv

                    # ─── 2. UPDATE LIVENESS MANAGER ───
                    result   = liveness.update_register(face, detector)
                    cur_step = result["step"]

                    # ─── 3. COMMIT DATA JIKA TAHAP SELESAI ───
                    if cur_step != "WAIT" and cur_step != _prev_step:
                        
                        if _prev_step in STEP_COMMIT:
                            axis    = STEP_COMMIT[_prev_step]
                            cap_key = POSE_CFG[STEP_TO_STAGE[_prev_step]][0]
                            cap[cap_key] = list(_pose_buf[axis].values())
                            _log(f"[COMMIT] {_prev_step} snapshots: {[s['tag'] for s in cap[cap_key]]}", "SUCCESS")
                            _pose_buf[axis] = {}  

                        if cur_step == "DONE" and _prev_step == "BLINK":
                            cap["blink_closed"] = _blink_buf["closed"]
                            cap["blink_open"]   = _blink_buf["open"]
                            
                            bc_ear = f"{cap['blink_closed']['avg_ear']:.3f}" if cap['blink_closed'] else "N/A"
                            bo_ear = f"{cap['blink_open']['avg_ear']:.3f}" if cap['blink_open'] else "N/A"
                            _log(f"[COMMIT] BLINK EAR: Closed={bc_ear}, Open={bo_ear}", "SUCCESS")

                        _prev_step = cur_step

                    # ─── 4. RUBAH STAGE ───
                    if cur_step != "WAIT":
                        stage = STEP_TO_STAGE.get(cur_step, stage)

                    # Tampilan Status
                    step = result["step"]
                    box_color   = config.COLOR_GREEN  if step == "DONE" or "DONE" in result.get("progress","") \
                             else config.COLOR_YELLOW if step == "WAIT" else config.COLOR_CYAN
                    status_text = "PASSED" if box_color==config.COLOR_GREEN else \
                                  "TAHAN LURUS" if box_color==config.COLOR_YELLOW else "VALIDATING"
                    _draw_box(display, face.bbox, status_text, box_color)
                    _draw_progress_bar(display, stage)
                    details = " ".join(f"{k.upper()}: {result[k]}" for k in ("yaw","pitch","roll") if k in result)
                    _draw_panel(display, stage, result["instruction"], result.get("progress",""), details)

                # ─── TAHAP EKSTRAKSI & VALIDASI ───
                if stage == RegistrationStage.EXTRACTION and not in_ext:
                    pose             = liveness.pose_estimator.estimate(face, detector)
                    yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

                    if abs(yaw) < config.EXTRACTION_MAX_YAW and \
                       abs(pitch) < config.EXTRACTION_MAX_PITCH and \
                       abs(roll) < config.EXTRACTION_MAX_ROLL:
                        
                        in_ext = True
                        
                        if cap["facemesh_vector"] is None:
                            fm = _capture_facemesh(face)
                            if fm is not None:
                                cap["facemesh_vector"] = fm
                                _log("FaceMesh di-capture di tahap ekstraksi (fallback)", "WARNING")

                        valid_to_save = True
                        missing = []
                        
                        if cap["facemesh_vector"] is None:
                            valid_to_save = False
                            missing.append("FaceMesh")
                        if len(cap["yaw_snapshots"]) < 2:
                            valid_to_save = False
                            missing.append("Yaw (Kiri & Kanan)")
                        if len(cap["pitch_snapshots"]) < 2:
                            valid_to_save = False
                            missing.append("Pitch (Atas & Bawah)")
                        if len(cap["roll_snapshots"]) < 2:
                            valid_to_save = False
                            missing.append("Roll (Kiri & Kanan)")
                        if cap["blink_closed"] is None or cap["blink_open"] is None:
                            valid_to_save = False
                            missing.append("Blink (Mata Tutup & Buka)")

                        if not valid_to_save:
                            _log("REGISTRASI DIBATALKAN! Ada gerakan yang tidak terekam.", "ERROR")
                            _log(f"Data tidak lengkap: {', '.join(missing)}", "ERROR")
                            
                            _draw_box(display, face.bbox, "DATA TIDAK VALID", config.COLOR_RED)
                            cv2.rectangle(display,(0,0),(config.FRAME_WIDTH,config.FRAME_HEIGHT),config.COLOR_RED,12)
                            _put(display,"REGISTRASI GAGAL!", config.FRAME_HEIGHT//2-50, config.COLOR_RED, 
                                 (config.FRAME_WIDTH-350)//2, 1.4, 3)
                            _put(display,"Gerakan Liveness tidak lengkap.", config.FRAME_HEIGHT//2+10, 
                                 config.COLOR_RED, (config.FRAME_WIDTH-450)//2, 0.8, 2)
                            
                            cv2.imshow("Register", display)
                            cv2.waitKey(4000)
                            break

                        # Ekstraksi Embedding
                        embedding = model.get_embedding(model.crop_face(frame, face.bbox))
                        cap["mobilefacenet_embedding"] = embedding
                        _log(f"[COMMIT] MobileFaceNet: 512-dim embedding", "SUCCESS")

                        _log("=" * 70, "SYSTEM")
                        _log("  DATA LIVENESS VALID! SIAP DISIMPAN KE FIREBASE", "SYSTEM")
                        _log("=" * 70, "SYSTEM")
                        
                        bc, bo = cap["blink_closed"], cap["blink_open"]
                        for label, val in [
                            ("1. FaceMesh",        f"{len(cap['facemesh_vector'])} dimensi"),
                            ("2. YAW",             f"{len(cap['yaw_snapshots'])} snapshots"),
                            ("3. PITCH",           f"{len(cap['pitch_snapshots'])} snapshots"),
                            ("4. ROLL",            f"{len(cap['roll_snapshots'])} snapshots"),
                            ("5. BLINK EAR",       f"Closed={bc['avg_ear']:.3f}, Open={bo['avg_ear']:.3f}" if bc and bo else "N/A"),
                            ("6. MobileFaceNet",   "512-dim embedding"),
                        ]:
                            _log(f"  {label:<20}: {val}", "SUCCESS")
                        _log("=" * 70, "SYSTEM")

                        match = matcher.match(embedding)
                        skor = match.get("score", 0.0) 
                        _log(f"Cek Duplikat -> Cocok: {match['matched']} | Terdeteksi sbg: {match.get('name')} | Skor: {skor:.3f}", "INFO")

                        display = frame.copy()

                        if match["matched"]:
                            _draw_box(display, face.bbox, "SUDAH TERDAFTAR", config.COLOR_RED)
                            cv2.rectangle(display,(0,0),(config.FRAME_WIDTH,config.FRAME_HEIGHT),config.COLOR_RED,12)
                            _put(display,"REGISTRASI DITOLAK!", config.FRAME_HEIGHT//2-50, config.COLOR_RED, 
                                 (config.FRAME_WIDTH-350)//2, 1.4, 3)
                            _put(display,f"Sudah terdaftar sebagai: {match['name']}", 
                                 config.FRAME_HEIGHT//2+10, config.COLOR_RED, (config.FRAME_WIDTH-500)//2, 0.8, 2)
                            cv2.imshow("Register", display)
                            cv2.waitKey(4000)
                            break
                        else:
                            # ─── MENYIMPAN KE FIREBASE ───
                            _log(f"Mengirim data wajah '{name}' ke Cloud...", "SYSTEM")
                            save_success = face_db.save_face(name, embedding, cap)
                            
                            if save_success:
                                _draw_box(display, face.bbox, "BERHASIL", config.COLOR_GREEN)
                                cv2.rectangle(display,(0,0),(config.FRAME_WIDTH,config.FRAME_HEIGHT),config.COLOR_GREEN,12)
                                _put(display,"REGISTRASI BERHASIL!", config.FRAME_HEIGHT//2-50, config.COLOR_GREEN, 
                                     (config.FRAME_WIDTH-400)//2, 1.6, 3)
                                _put(display,f"Nama: {name}", config.FRAME_HEIGHT//2+10, config.COLOR_GREEN, 
                                     (config.FRAME_WIDTH-150)//2, 1.1, 2)
                            else:
                                _draw_box(display, face.bbox, "GAGAL SIMPAN", config.COLOR_RED)
                                cv2.rectangle(display,(0,0),(config.FRAME_WIDTH,config.FRAME_HEIGHT),config.COLOR_RED,12)
                                _put(display,"GAGAL KONEKSI DATABASE!", config.FRAME_HEIGHT//2-50, config.COLOR_RED, 
                                     (config.FRAME_WIDTH-450)//2, 1.6, 3)

                        stage = RegistrationStage.COMPLETE
                        cv2.imshow("Register", display)
                        cv2.waitKey(2000)
                        break
                    else:
                        _draw_box(display, face.bbox, "TAHAN LURUS", config.COLOR_YELLOW)
                        _draw_progress_bar(display, stage)
                        _draw_panel(display, stage, "Tatap LURUS ke kamera untuk ekstraksi embedding",
                                    "Menunggu posisi netral...", f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}")

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _log("Dibatalkan oleh user", "WARNING")
                break

    except Exception as e:
        _log(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
    finally:
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()
        _log("Program selesai.", "SYSTEM")


if __name__ == "__main__":
    name = input("\nMasukkan nama Anda: ").strip()
    if name:
        run_register(name)
        
        try:
            lanjut = input("\n[SISTEM] Registrasi selesai. Ingin langsung menjalankan Smart Door Lock? (y/n): ").strip().lower()
            if lanjut == 'y':
                from main import run_unlock
                print("\nMemulai sistem utama...\n")
                run_unlock()
            else:
                print("\nKembali ke terminal.")
        except ImportError:
            print("\n[Peringatan] Gagal memuat main.py.")
            print("Pastikan kode utama di main.py Anda sudah terbungkus dalam fungsi bernama 'run_unlock()'.")
        except Exception as e:
            print(f"\nTerjadi kesalahan saat menjalankan main: {e}")
    else:
        print("Nama tidak boleh kosong!")