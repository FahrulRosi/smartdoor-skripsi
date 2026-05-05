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
from database.face_db           import FaceDatabase
from liveness.anti_spoofing     import SilentAntiSpoofing

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


# ─── ENUMS & CONFIGURATIONS ───────────────────────────────────────────────────
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


def _log(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")


# ─── DATA EXTRACTOR CLASS ─────────────────────────────────────────────────────
class DataExtractor:
    """Kelas helper untuk mengekstrak data dari MediaPipe FaceMesh."""
    _LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    @staticmethod
    def capture_facemesh(face):
        try:
            lm = face.landmarks
            if not lm: return None
            points = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
            return points.flatten()
        except Exception as e:
            _log(f"Gagal capture facemesh: {e}", "WARNING")
            return None

    @staticmethod
    def capture_pose(pose, tag):
        return {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")} | {"tag": tag}

    @staticmethod
    def capture_blink(face):
        try:
            lm = face.landmarks
            if not lm or len(lm) < 400: return None

            def ear(eye_indices):
                pts = np.array([[lm[i].x, lm[i].y] for i in eye_indices], dtype=np.float32)
                v1, v2 = np.linalg.norm(pts[1] - pts[5]), np.linalg.norm(pts[2] - pts[4])
                h = np.linalg.norm(pts[0] - pts[3])
                return (v1 + v2) / (2.0 * h + 1e-6)

            l_ear, r_ear = ear(DataExtractor._LEFT_EYE), ear(DataExtractor._RIGHT_EYE)
            return {"left_ear": float(l_ear), "right_ear": float(r_ear), "avg_ear": float((l_ear + r_ear) / 2.0)}
        except Exception as e:
            _log(f"Gagal capture blink: {e}", "WARNING")
            return None


# ─── UI MANAGER CLASS ─────────────────────────────────────────────────────────
class RegistrationUI:
    """Kelas untuk menangani penggambaran elemen visual di frame."""
    @staticmethod
    def put_text(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.6, thickness=1):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    @staticmethod
    def draw_progress_bar(frame, stage, total=6):
        W, bw, bh = config.FRAME_WIDTH, 350, 25
        x, y = (W - bw) // 2, 15
        sv = min(stage.value, total)
        
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (30,30,30), -1)
        pw = int(bw * (sv-1) / total) if sv > 0 else 0
        if pw > 0: 
            cv2.rectangle(frame, (x, y), (x+pw, y+bh), config.COLOR_GREEN, -1)
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), config.COLOR_WHITE, 2)
        
        txt = f"Tahap {sv if sv<=5 else 6}/6"
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(frame, txt, (x+(bw-ts[0])//2, y+(bh+ts[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    @staticmethod
    def draw_panel(frame, stage, instruction, progress="", details=""):
        cv2.rectangle(frame, (0,50), (config.FRAME_WIDTH, 150), (20,20,20), -1)
        cv2.rectangle(frame, (0,50), (config.FRAME_WIDTH, 150), config.COLOR_CYAN, 2)
        RegistrationUI.put_text(frame, STAGE_NAMES.get(stage,"Proses..."), 75,  config.COLOR_GREEN,  20, 0.85, 2)
        RegistrationUI.put_text(frame, instruction,                        105, config.COLOR_YELLOW, 20, 0.65, 2)
        if progress: RegistrationUI.put_text(frame, f"Status: {progress}", 125, config.COLOR_CYAN,  20, 0.6)
        if details:  RegistrationUI.put_text(frame, details,               145, config.COLOR_WHITE, 20, 0.55)

    @staticmethod
    def draw_box(frame, bbox, status, color):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 3)
        cl = 25
        for (a,b),(c,d) in [((x,y),(x+cl,y)),((x,y),(x,y+cl)),((x+w,y),(x+w-cl,y)),((x+w,y),(x+w,y+cl))]:
            cv2.line(frame, (a,b), (c,d), color, 3)
        cv2.rectangle(frame, (x,y-35), (x+180,y-5), color, -1)
        cv2.putText(frame, status, (x+5,y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    @staticmethod
    def show_full_screen_message(frame, title, subtitle, color):
        cv2.rectangle(frame, (0,0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), color, 12)
        RegistrationUI.put_text(frame, title, config.FRAME_HEIGHT//2-50, color, (config.FRAME_WIDTH-400)//2, 1.4, 3)
        RegistrationUI.put_text(frame, subtitle, config.FRAME_HEIGHT//2+10, color, (config.FRAME_WIDTH-450)//2, 0.8, 2)


# ─── MAIN APP CLASS ───────────────────────────────────────────────────────────
class FaceRegistrationApp:
    """Kelas utama pengelola alur pendaftaran wajah."""
    
    POSE_CFG = {
        RegistrationStage.YAW:   ("yaw_snapshots",   "yaw_left",  "yaw_right",  "yaw",   config.YAW_THRESHOLD),
        RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up",  "pitch_down", "pitch", config.PITCH_THRESHOLD),
        RegistrationStage.ROLL:  ("roll_snapshots",  "roll_left", "roll_right", "roll",  config.ROLL_THRESHOLD),
    }
    STEP_COMMIT = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}

    def __init__(self, name):
        self.name = name
        self.db_url = getattr(config, "FIREBASE_URL", "")
        self.credentials_path = getattr(config, "FIREBASE_CREDENTIALS", "serviceAccount.json")
        
        self.stage = RegistrationStage.FACEMESH
        self.in_ext = False
        
        self.captured_data = {
            "facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [],
            "roll_snapshots": [], "blink_closed": None, "blink_open": None,
            "mobilefacenet_embedding": None,
        }
        
        self._pose_buf = {"yaw": {}, "pitch": {}, "roll": {}}
        self._blink_buf = {"closed": None, "open": None}
        self._prev_step = "FACEMESH"

    def _init_hardware_and_services(self):
        _log("Menghubungkan ke Firebase...", "SYSTEM")
        self.face_db = FaceDatabase(self.db_url, self.credentials_path)
        
        if self.face_db.check_user_exists(self.name):
            _log(f"Registrasi Dibatalkan: Nama '{self.name}' sudah terdaftar!", "ERROR")
            return False

        if GPIO_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(config.IR_CUT_PIN, GPIO.OUT)
                GPIO.output(config.IR_CUT_PIN, GPIO.HIGH)
            except Exception as e:
                _log(f"Gagal IR-CUT: {e}", "WARNING")

        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, apply_enhancement=getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', False)).start()
        self.detector = FaceMeshDetector(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE)
        self.liveness = LivenessManager()
        self.model = MobileFaceNet()
        self.matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
        self.anti_spoof = SilentAntiSpoofing()
        
        # --- PERBAIKAN FORMAT DATA FIREBASE ---
        try:
            raw_faces = self.face_db.load_all_faces()
            if raw_faces:
                processed_faces = {}
                for user_name, user_data in raw_faces.items():
                    # Jika data turun sebagai List/Array murni
                    if isinstance(user_data, (list, np.ndarray)):
                        processed_faces[user_name] = np.array(user_data, dtype=np.float32)
                    # Jika data turun sebagai Nested Dictionary (Struktur JSON Firebase)
                    elif isinstance(user_data, dict):
                        if 'embedding' in user_data:
                            processed_faces[user_name] = np.array(user_data['embedding'], dtype=np.float32)
                        elif 'mobilefacenet_embedding' in user_data:
                            processed_faces[user_name] = np.array(user_data['mobilefacenet_embedding'], dtype=np.float32)
                
                # Memastikan data disuntikkan dengan benar ke FaceMatcher
                if hasattr(self.matcher, 'load_faces'):
                    self.matcher.load_faces(processed_faces)
                else:
                    self.matcher.known_faces = processed_faces
                    
                _log(f"Berhasil memuat dan memformat {len(processed_faces)} profil wajah ke memori.", "SUCCESS")
            else:
                _log("Database wajah masih kosong.", "INFO")
        except Exception as e:
            _log(f"Gagal memuat data wajah dari Firebase: {e}", "ERROR")
        # ---------------------------------------

        self.liveness.start_register()
        
        return True

    def _record_data_buffers(self, face, pose):
        if self.stage == RegistrationStage.FACEMESH and self.captured_data["facemesh_vector"] is None:
            fm = DataExtractor.capture_facemesh(face)
            if fm is not None:
                self.captured_data["facemesh_vector"] = fm
                _log(f"[COMMIT] FaceMesh: {len(fm)} dimensi", "SUCCESS")

        if self.stage in self.POSE_CFG:
            _, tag_neg, tag_pos, axis, thr = self.POSE_CFG[self.stage]
            val, buf, cap_thr = pose.get(axis, 0.0), self._pose_buf[axis], thr * 0.7  

            if val < -cap_thr and (tag_neg not in buf or val < buf[tag_neg][axis]):
                buf[tag_neg] = DataExtractor.capture_pose(pose, tag_neg)
            if val > cap_thr and (tag_pos not in buf or val > buf[tag_pos][axis]):
                buf[tag_pos] = DataExtractor.capture_pose(pose, tag_pos)

        if self.stage == RegistrationStage.BLINK:
            bv = DataExtractor.capture_blink(face)
            if bv:
                if self._blink_buf["closed"] is None or bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]:
                    self._blink_buf["closed"] = bv
                if self._blink_buf["open"] is None or bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]:
                    self._blink_buf["open"] = bv

    def _commit_stage_data(self, cur_step):
        if cur_step == "WAIT" or cur_step == self._prev_step:
            return

        if self._prev_step in self.STEP_COMMIT:
            axis = self.STEP_COMMIT[self._prev_step]
            cap_key = self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]
            self.captured_data[cap_key] = list(self._pose_buf[axis].values())
            _log(f"[COMMIT] {self._prev_step} snapshots: {[s['tag'] for s in self.captured_data[cap_key]]}", "SUCCESS")
            self._pose_buf[axis] = {}  

        if cur_step == "DONE" and self._prev_step == "BLINK":
            self.captured_data["blink_closed"] = self._blink_buf["closed"]
            self.captured_data["blink_open"] = self._blink_buf["open"]
            _log("[COMMIT] BLINK EAR tersimpan", "SUCCESS")

        self._prev_step = cur_step
        self.stage = STEP_TO_STAGE.get(cur_step, self.stage)

    def _get_missing_data(self):
        missing = []
        if self.captured_data["facemesh_vector"] is None: missing.append("FaceMesh")
        if len(self.captured_data["yaw_snapshots"]) < 2: missing.append("Yaw")
        if len(self.captured_data["pitch_snapshots"]) < 2: missing.append("Pitch")
        if len(self.captured_data["roll_snapshots"]) < 2: missing.append("Roll")
        if not self.captured_data["blink_closed"] or not self.captured_data["blink_open"]: missing.append("Blink")
        return missing

    def _process_extraction_stage(self, frame, face, display):
        pose = self.liveness.pose_estimator.estimate(face, self.detector)
        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

        if abs(yaw) >= config.EXTRACTION_MAX_YAW or abs(pitch) >= config.EXTRACTION_MAX_PITCH or abs(roll) >= config.EXTRACTION_MAX_ROLL:
            RegistrationUI.draw_box(display, face.bbox, "TAHAN LURUS", config.COLOR_YELLOW)
            RegistrationUI.draw_progress_bar(display, self.stage)
            RegistrationUI.draw_panel(display, self.stage, "Tatap LURUS ke kamera untuk ekstraksi embedding", "Menunggu posisi netral...")
            return

        self.in_ext = True
        missing = self._get_missing_data()

        if missing:
            _log(f"Data tidak lengkap: {', '.join(missing)}", "ERROR")
            RegistrationUI.show_full_screen_message(display, "REGISTRASI GAGAL!", "Gerakan Liveness tidak lengkap.", config.COLOR_RED)
            cv2.imshow("Register", display)
            cv2.waitKey(4000)
            self.stage = RegistrationStage.COMPLETE
            return

        # Ekstraksi Embedding
        embedding = self.model.get_embedding(self.model.crop_face(frame, face.bbox))
        self.captured_data["mobilefacenet_embedding"] = embedding
        
        # --- PERBAIKAN LOGIKA ANTI-DUPLIKASI WAJAH ---
        # Menurunkan threshold secara paksa agar lebih mudah mendeteksi kesamaan
        # Meskipun posenya agak berbeda, jika mirip > 50% akan diblokir
        self.matcher.threshold = 0.50 
        match = self.matcher.match(embedding)

        skor = match.get("score", 0.0)
        nama_terdekat = match.get("name", "Unknown")
        _log(f"[DUPLIKASI CHECK] Wajah terdeteksi mirip dengan: '{nama_terdekat}' (Skor: {skor:.4f})", "SYSTEM")

        if match.get("matched", False):
            RegistrationUI.show_full_screen_message(display, "REGISTRASI DITOLAK!", f"Muka ini adalah milik: {nama_terdekat}", config.COLOR_RED)
            cv2.imshow("Register", display)
            cv2.waitKey(4000)
            self.stage = RegistrationStage.COMPLETE
            return
        # ---------------------------------------------

        # Simpan ke Firebase
        _log(f"Mengirim data wajah '{self.name}' ke Cloud...", "SYSTEM")
        if self.face_db.save_face(self.name, embedding, self.captured_data):
            RegistrationUI.show_full_screen_message(display, "REGISTRASI BERHASIL!", f"Nama: {self.name}", config.COLOR_GREEN)
        else:
            RegistrationUI.show_full_screen_message(display, "GAGAL KONEKSI DATABASE!", "Periksa koneksi internet Anda.", config.COLOR_RED)
        
        cv2.imshow("Register", display)
        cv2.waitKey(2000)
        self.stage = RegistrationStage.COMPLETE

    def run(self):
        _log(f"Memulai registrasi: {self.name}", "SYSTEM")
        if not self._init_hardware_and_services():
            return

        try:
            while self.stage != RegistrationStage.COMPLETE:
                ret, frame = self.cam.read()
                if not ret: continue
                display = frame.copy()
                faces = self.detector.detect(frame)

                enhancement_status = "ON" if getattr(self.cam, 'apply_enhancement', False) else "OFF"
                color_enh = config.COLOR_GREEN if enhancement_status == "ON" else config.COLOR_RED
                RegistrationUI.put_text(display, f"CLAHE: {enhancement_status}", 20, color_enh, x=config.FRAME_WIDTH - 120, scale=0.5)

                if not faces:
                    RegistrationUI.draw_panel(display, self.stage, "Wajah tidak terdeteksi", "Hadapkan wajah ke kamera")
                else:
                    face = faces[0]
                    display = self.detector.draw(display, face)

                    # 1. Cek Anti Spoofing
                    spoof = self.anti_spoof.is_real(frame, face.bbox)
                    if not spoof["real"]:
                        RegistrationUI.draw_box(display, face.bbox, "WAJAH PALSU!", config.COLOR_RED)
                        RegistrationUI.draw_panel(display, self.stage, "Terdeteksi Foto/Video!", f"Skor: {spoof['score']}", "Gunakan wajah asli!")
                        cv2.imshow("Register", display)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"): break
                        elif key == ord("e"):
                            if hasattr(self.cam, 'apply_enhancement'):
                                self.cam.apply_enhancement = not self.cam.apply_enhancement
                        continue

                    # 2. Proses Tahapan Registrasi
                    if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                        pose = self.liveness.pose_estimator.estimate(face, self.detector)
                        
                        self._record_data_buffers(face, pose)
                        result = self.liveness.update_register(face, self.detector)
                        self._commit_stage_data(result["step"])

                        box_color = config.COLOR_GREEN if result["step"] == "DONE" or "DONE" in result.get("progress","") else \
                                    config.COLOR_YELLOW if result["step"] == "WAIT" else config.COLOR_CYAN
                        status_text = "PASSED" if box_color==config.COLOR_GREEN else "TAHAN LURUS" if box_color==config.COLOR_YELLOW else "VALIDATING"
                        
                        RegistrationUI.draw_box(display, face.bbox, status_text, box_color)
                        RegistrationUI.draw_progress_bar(display, self.stage)
                        RegistrationUI.draw_panel(display, self.stage, result["instruction"], result.get("progress",""))

                    # 3. Proses Tahap Validasi dan Ekstraksi Akhir
                    elif self.stage == RegistrationStage.EXTRACTION and not self.in_ext:
                        self._process_extraction_stage(frame, face, display)

                cv2.imshow("Register", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    _log("Dibatalkan oleh user", "WARNING")
                    break
                elif key == ord("e"):
                    if hasattr(self.cam, 'apply_enhancement'):
                        self.cam.apply_enhancement = not self.cam.apply_enhancement
                        _log(f"CLAHE Enhancement diset ke: {self.cam.apply_enhancement}", "SYSTEM")

        except Exception as e:
            _log(f"Error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        finally:
            if GPIO_AVAILABLE: GPIO.cleanup()
            self.cam.stop()
            self.detector.close()
            cv2.destroyAllWindows()
            _log("Program selesai.", "SYSTEM")


if __name__ == "__main__":
    name = input("\nMasukkan nama Anda: ").strip()
    if name:
        app = FaceRegistrationApp(name)
        app.run()
        print("\n[SISTEM] Registrasi selesai. Kembali ke terminal.")
    else:
        print("Nama tidak boleh kosong!")