import cv2
import numpy as np
from enum import Enum
from datetime import datetime
import os

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
    "PITCH": RegistrationStage.PITCH, "ROLL": RegistrationStage.ROLL,
    "BLINK": RegistrationStage.BLINK, "DONE": RegistrationStage.EXTRACTION,
}

FACE_MATCHER_THRESHOLD = 0.60
ALLOW_DUPLICATE = os.getenv("REGISTER_ALLOW_DUPLICATE", "false").lower() == "true"

def _log(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")

class DataExtractor:
    _LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    @staticmethod
    def capture_facemesh(face):
        try:
            if not face.landmarks: return None
            points = np.array([[l.x, l.y, l.z] for l in face.landmarks], dtype=np.float32)
            return points.flatten()
        except Exception: return None

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
            return {"left_ear": float(l_ear), "right_ear": float(r_ear), 
                    "avg_ear": float((l_ear + r_ear) / 2.0)}
        except Exception: return None

class RegistrationUI:
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
        if pw > 0: cv2.rectangle(frame, (x, y), (x+pw, y+bh), config.COLOR_GREEN, -1)
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), config.COLOR_WHITE, 2)
        txt = f"Tahap {sv if sv<=5 else 6}/6"
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(frame, txt, (x+(bw-ts[0])//2, y+(bh+ts[1])//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    @staticmethod
    def draw_panel(frame, stage, instruction, progress="", details=""):
        cv2.rectangle(frame, (0,50), (config.FRAME_WIDTH, 150), (20,20,20), -1)
        cv2.rectangle(frame, (0,50), (config.FRAME_WIDTH, 150), config.COLOR_CYAN, 2)
        RegistrationUI.put_text(frame, STAGE_NAMES.get(stage,"Proses..."), 75,  
                                config.COLOR_GREEN,  20, 0.85, 2)
        RegistrationUI.put_text(frame, instruction, 105, config.COLOR_YELLOW, 20, 0.65, 2)
        if progress: 
            RegistrationUI.put_text(frame, progress, 125, config.COLOR_CYAN, 20, 0.6)
        if details:  
            RegistrationUI.put_text(frame, details, 145, config.COLOR_WHITE, 20, 0.55)

    @staticmethod
    def draw_box(frame, bbox, status, color):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 3)
        cv2.rectangle(frame, (x,y-35), (x+180,y-5), color, -1)
        cv2.putText(frame, status, (x+5,y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    @staticmethod
    def show_message(frame, title, subtitle, color):
        cv2.rectangle(frame, (0,0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), color, 12)
        RegistrationUI.put_text(frame, title, config.FRAME_HEIGHT//2-50, color, 
                                (config.FRAME_WIDTH-400)//2, 1.4, 3)
        RegistrationUI.put_text(frame, subtitle, config.FRAME_HEIGHT//2+10, color, 
                                (config.FRAME_WIDTH-450)//2, 0.8, 2)

class FaceRegistrationApp:
    POSE_CFG = {
        RegistrationStage.YAW:   ("yaw_snapshots",   "yaw_left",  "yaw_right",  "yaw",   config.YAW_THRESHOLD),
        RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up",  "pitch_down", "pitch", config.PITCH_THRESHOLD),
        RegistrationStage.ROLL:  ("roll_snapshots",  "roll_left", "roll_right", "roll",  config.ROLL_THRESHOLD),
    }
    STEP_COMMIT = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}

    def __init__(self, name):
        self.name = name
        self.stage = RegistrationStage.FACEMESH
        self.in_ext = False
        self.hold_frames = 0 
        
        self.captured_data = {
            "facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [],
            "roll_snapshots": [], "blink_closed": None, "blink_open": None,
            "headpose_vector": None
        }
        self._pose_buf = {"yaw": {}, "pitch": {}, "roll": {}}
        self._blink_buf = {"closed": None, "open": None}
        self._prev_step = "FACEMESH"
        
        self.db = FaceDatabase()
        if self.db.check_user_exists(self.name):
            _log(f"❌ Registrasi Dibatalkan: '{self.name}' sudah terdaftar!", "ERROR")
            self.stage = RegistrationStage.COMPLETE
            return

        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        
        self.detector = FaceMeshDetector(
            min_detection_confidence=getattr(config, 'MIN_DETECTION_CONFIDENCE', 0.5), 
            min_tracking_confidence=getattr(config, 'MIN_TRACKING_CONFIDENCE', 0.5)
        )
        
        self.liveness = LivenessManager()
        self.model = MobileFaceNet()
        self.anti_spoof = SilentAntiSpoofing()
        
        self.matcher = FaceMatcher(threshold=FACE_MATCHER_THRESHOLD)
        try:
            self.matcher.known_faces = self.db.load_all_faces()
        except: pass
        
        self.liveness.start_register()
        _log(f"✅ Inisialisasi registrasi untuk: {self.name}", "SYSTEM")

    def _record_data_buffers(self, face, pose):
        """Memaksa pengguna menahan pose beberapa frame sebelum snapshot diambil (Anti-Skip)"""
        if self.stage == RegistrationStage.FACEMESH and self.captured_data["facemesh_vector"] is None:
            fm = DataExtractor.capture_facemesh(face)
            if fm is not None: 
                self.hold_frames += 1
                if self.hold_frames >= 10: 
                    self.captured_data["facemesh_vector"] = fm
                    self.hold_frames = 0

        if self.stage in self.POSE_CFG:
            _, tag_neg, tag_pos, axis, thr = self.POSE_CFG[self.stage]
            val, buf, cap_thr = pose.get(axis, 0.0), self._pose_buf[axis], thr * 0.7  
            
            if val < -cap_thr:
                self.hold_frames += 1
                if self.hold_frames >= 3: 
                    if tag_neg not in buf or val < buf[tag_neg][axis]: 
                        buf[tag_neg] = DataExtractor.capture_pose(pose, tag_neg)
            elif val > cap_thr:
                self.hold_frames += 1
                if self.hold_frames >= 3:
                    if tag_pos not in buf or val > buf[tag_pos][axis]: 
                        buf[tag_pos] = DataExtractor.capture_pose(pose, tag_pos)
            else:
                self.hold_frames = 0

        if self.stage == RegistrationStage.BLINK:
            bv = DataExtractor.capture_blink(face)
            if bv:
                if self._blink_buf["closed"] is None or bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]: 
                    self._blink_buf["closed"] = bv
                if self._blink_buf["open"] is None or bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]: 
                    self._blink_buf["open"] = bv

    def _commit_stage_data(self, cur_step):
        if cur_step == "WAIT" or cur_step == self._prev_step: return
        
        # --- PENGAWAS ANTI-SKIP ---
        if self._prev_step in self.STEP_COMMIT:
            axis = self.STEP_COMMIT[self._prev_step]
            cap_key = self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]
            
            if len(self._pose_buf[axis].values()) < 2:
                _log(f"⚠️ Mencegah Skip! Data {axis.upper()} tidak lengkap. Mengulang tahapan.", "WARNING")
                if hasattr(self.liveness, '_register_step'):
                    self.liveness._register_step -= 1
                    self.liveness._pose_state = "WAITING_EXTREME"
                return 
            
            self.captured_data[cap_key] = list(self._pose_buf[axis].values())
            self._pose_buf[axis] = {}  
            
        if cur_step == "DONE" and self._prev_step == "BLINK":
            if not self._blink_buf["closed"] or not self._blink_buf["open"]:
                _log(f"⚠️ Mencegah Skip! Kedipan belum sempurna.", "WARNING")
                if hasattr(self.liveness, '_register_step'):
                    self.liveness._register_step -= 1
                return
            self.captured_data["blink_closed"] = self._blink_buf["closed"]
            self.captured_data["blink_open"] = self._blink_buf["open"]
            
        self._prev_step = cur_step
        self.stage = STEP_TO_STAGE.get(cur_step, self.stage)

    def _get_missing_data(self):
        missing = []
        if self.captured_data["facemesh_vector"] is None: missing.append("FaceMesh")
        if len(self.captured_data["yaw_snapshots"]) < 2: missing.append("Yaw")
        if len(self.captured_data["pitch_snapshots"]) < 2: missing.append("Pitch")
        if len(self.captured_data["roll_snapshots"]) < 2: missing.append("Roll")
        if not self.captured_data["blink_closed"] or not self.captured_data["blink_open"]: 
            missing.append("Blink")
        return missing

    def _check_spoof(self, frame, face_bbox) -> dict:
        return self.anti_spoof.is_real(frame, face_bbox)

    def _handle_spoof_detected(self, display, spoof_result, face_bbox):
        score = spoof_result.get("score", 0.0)
        
        # --- KETERANGAN UMUM SPOOFING ---
        status_text = "TERDETEKSI SPOOFING"

        RegistrationUI.draw_box(display, face_bbox, status_text, config.COLOR_RED)
        RegistrationUI.draw_panel(display, self.stage, 
            f"❌ Peringatan: {status_text}!", 
            f"Skor Kepalsuan: {score:.2f} - Mohon gunakan wajah asli")

    def _handle_duplicate_face(self, display, matched_name, match_score):
        _log(f"⚠️ WAJAH DUPLIKASI: '{self.name}' == '{matched_name}' (Score: {match_score:.4f})", "WARNING")
        RegistrationUI.show_message(display, 
            "❌ WAJAH SUDAH TERDAFTAR!", 
            f"Nama: {matched_name}\nScore: {match_score:.4f}", 
            config.COLOR_RED)

    def _extract_and_save(self, frame, face, display, yaw, pitch, roll):
        embedding = self.model.get_embedding(self.model.crop_face(frame, face.bbox))
        self.captured_data["headpose_vector"] = [float(yaw), float(pitch), float(roll)]
        
        match = self.matcher.match(embedding)
        if match["matched"] and not ALLOW_DUPLICATE:
            self._handle_duplicate_face(display, match['name'], match['score'])
            cv2.imshow("Register", display)
            cv2.waitKey(4000)
            self.stage = RegistrationStage.COMPLETE
            return False
        elif match["matched"] and ALLOW_DUPLICATE:
            _log(f"⚠️ ADMIN: Duplikasi diizinkan {match['name']} → {self.name}", "WARNING")

        if self.db.check_user_exists(self.name):
            RegistrationUI.show_message(display, "❌ GAGAL!", 
                f"Nama '{self.name}' sudah digunakan", config.COLOR_RED)
            cv2.imshow("Register", display)
            cv2.waitKey(4000)
            self.stage = RegistrationStage.COMPLETE
            return False

        _log(f"💾 Menyimpan: '{self.name}'...", "SYSTEM")
        if self.db.save_face(self.name, embedding, self.captured_data):
            RegistrationUI.show_message(display, "✅ BERHASIL!", 
                f"Nama: {self.name}", config.COLOR_GREEN)
            _log(f"✅ Registrasi BERHASIL: {self.name}", "SUCCESS")
        else:
            RegistrationUI.show_message(display, "❌ GAGAL!", 
                "Koneksi Database Error", config.COLOR_RED)
            _log(f"❌ Registrasi GAGAL: Database error", "ERROR")
        
        cv2.imshow("Register", display)
        cv2.waitKey(3000)
        self.stage = RegistrationStage.COMPLETE
        return True

    def _process_extraction_stage(self, frame, face, display):
        pose = self.liveness.pose_estimator.estimate(face, self.detector)
        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
        is_frontal = (abs(yaw) < 15 and abs(pitch) < 15 and abs(roll) < 15)
        
        if is_frontal:
            spoof = self._check_spoof(frame, face.bbox)
            if not spoof.get("real", True):
                self._handle_spoof_detected(display, spoof, face.bbox)
                RegistrationUI.draw_progress_bar(display, self.stage)
                return

        if abs(yaw) >= config.EXTRACTION_MAX_YAW or abs(pitch) >= config.EXTRACTION_MAX_PITCH or abs(roll) >= config.EXTRACTION_MAX_ROLL:
            RegistrationUI.draw_box(display, face.bbox, "TAHAN LURUS", config.COLOR_YELLOW)
            RegistrationUI.draw_progress_bar(display, self.stage)
            RegistrationUI.draw_panel(display, self.stage, 
                "Tatap LURUS ke kamera untuk ekstraksi", "Menunggu posisi netral...")
            return

        self.in_ext = True
        
        missing = self._get_missing_data()
        if missing:
            RegistrationUI.show_message(display, "❌ REGISTRASI GAGAL!", 
                f"Kurang: {', '.join(missing)}", config.COLOR_RED)
            cv2.imshow("Register", display)
            cv2.waitKey(4000)
            self.stage = RegistrationStage.COMPLETE
            return

        self._extract_and_save(frame, face, display, yaw, pitch, roll)

    def run(self):
        if self.stage == RegistrationStage.COMPLETE: return
        _log(f"🎬 Memulai registrasi: {self.name}", "SYSTEM")

        try:
            while self.stage != RegistrationStage.COMPLETE:
                ret, frame = self.cam.read()
                if not ret: continue
                display = frame.copy()
                faces = self.detector.detect(frame)

                if not faces:
                    RegistrationUI.draw_panel(display, self.stage, 
                        "Wajah tidak terdeteksi", "Hadapkan wajah ke kamera")
                else:
                    face = faces[0]
                    display = self.detector.draw(display, face)

                    spoof = self._check_spoof(frame, face.bbox)
                    if not spoof.get("real", True):
                        self._handle_spoof_detected(display, spoof, face.bbox)
                        cv2.imshow("Register", display)
                        if cv2.waitKey(1) & 0xFF == ord("q"): break
                        continue 

                    # Liveness Manager tetap memandu UI, tapi Register.py bertindak sbg Pengawas Data
                    if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                        pose = self.liveness.pose_estimator.estimate(face, self.detector)
                        self._record_data_buffers(face, pose) 
                        result = self.liveness.update_register(face, self.detector)
                        self._commit_stage_data(result["step"])

                        box_color = config.COLOR_GREEN if result["step"] == "DONE" else config.COLOR_CYAN
                        RegistrationUI.draw_box(display, face.bbox, "VALIDATING", box_color)
                        RegistrationUI.draw_progress_bar(display, self.stage)
                        RegistrationUI.draw_panel(display, self.stage, result["instruction"], 
                                                 result.get("progress",""))

                    elif self.stage == RegistrationStage.EXTRACTION and not self.in_ext:
                        self._process_extraction_stage(frame, face, display)

                cv2.imshow("Register", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break

        except Exception as e: 
            _log(f"❌ Error: {e}", "ERROR")
        finally:
            if GPIO_AVAILABLE: GPIO.cleanup()
            self.cam.stop()
            self.detector.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("\nMasukkan Nama Panggilan: ").strip()
    if name: FaceRegistrationApp(name).run()