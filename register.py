import cv2
import os
import numpy as np
import time
import threading
from enum import Enum
from datetime import datetime

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from database.face_db           import FaceDatabase
from liveness.anti_spoofing     import SilentAntiSpoofing

GPIO_AVAILABLE = True
try: import RPi.GPIO as GPIO
except ImportError: GPIO_AVAILABLE = False

class RegistrationStage(Enum):
    IDLE=0; FACEMESH=1; YAW=2; PITCH=3; ROLL=4; BLINK=5; EXTRACTION=6; COMPLETE=7

STAGE_NAMES = {
    RegistrationStage.FACEMESH: "1. FaceMesh (3D)", RegistrationStage.YAW: "2a. Liveness (Yaw)",
    RegistrationStage.PITCH: "2b. Liveness (Pitch)", RegistrationStage.ROLL: "2c. Liveness (Roll)",
    RegistrationStage.BLINK: "3. Liveness (Blink)", RegistrationStage.EXTRACTION: "4. Ekstraksi Fitur"
}
STEP_TO_STAGE = {
    "FACEMESH": RegistrationStage.FACEMESH, "YAW": RegistrationStage.YAW,
    "PITCH": RegistrationStage.PITCH, "ROLL": RegistrationStage.ROLL,
    "BLINK": RegistrationStage.BLINK, "DONE": RegistrationStage.EXTRACTION
}

def _log(msg, level="INFO"): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")

class DataExtractor:
    _LEFT_EYE = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    @staticmethod
    def capture_facemesh(face): 
        try:
            if not face.landmarks: return None
            return np.array([[l.x, l.y, l.z] for l in face.landmarks], dtype=np.float32).flatten()
        except Exception: return None
    
    @staticmethod
    def capture_pose(pose, tag): 
        return {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")} | {"tag": tag}
    
    @classmethod
    def capture_blink(cls, face):
        try:
            if not face.landmarks or len(face.landmarks) < 400: return None
            def ear(idx):
                pts = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in idx], dtype=np.float32)
                return (np.linalg.norm(pts[1]-pts[5]) + np.linalg.norm(pts[2]-pts[4])) / (2.0 * np.linalg.norm(pts[0]-pts[3]) + 1e-6)
            l_ear, r_ear = ear(cls._LEFT_EYE), ear(cls._RIGHT_EYE)
            return {"left_ear": float(l_ear), "right_ear": float(r_ear), "avg_ear": float((l_ear+r_ear)/2.0)}
        except Exception: return None

class RegistrationUI:
    @staticmethod
    def draw_hud(frame, stage, instruction, progress, score_txt, status, bbox, box_color):
        x, y, w, h = bbox if bbox else (0,0,0,0)
        
        if bbox:
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 3)
            cv2.rectangle(frame, (x, y-35), (x+180, y-5), box_color, -1)
            cv2.putText(frame, status, (x+5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        
        cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 175), (20,20,20), -1)
        cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 175), config.COLOR_CYAN, 2)
        
        def pt(txt, y_pos, color=config.COLOR_WHITE, size=0.6, thk=1):
            cv2.putText(frame, txt, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, size, color, thk)
            
        pt(STAGE_NAMES.get(stage, "Proses..."), 75, config.COLOR_GREEN, 0.85, 2)
        pt(instruction, 105, config.COLOR_YELLOW, 0.65, 2)
        if progress: pt(progress, 130, config.COLOR_CYAN, 0.6)
        if score_txt: pt(score_txt, 155, config.COLOR_WHITE, 0.5)

        W, bw, bh = config.FRAME_WIDTH, 350, 25
        bx, by = (W - bw) // 2, 15
        sv = min(stage.value, 6)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (30,30,30), -1)
        if sv > 0: cv2.rectangle(frame, (bx, by), (bx + int(bw * (sv-1)/6), by+bh), config.COLOR_GREEN, -1)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), config.COLOR_WHITE, 2)
        cv2.putText(frame, f"Tahap {sv if sv<=5 else 6}/6", (bx+130, by+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    @staticmethod
    def show_msg(frame, title, sub, color):
        cv2.rectangle(frame, (0,0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), color, 12)
        cv2.putText(frame, title, ((config.FRAME_WIDTH-400)//2, config.FRAME_HEIGHT//2-50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(frame, sub, ((config.FRAME_WIDTH-450)//2, config.FRAME_HEIGHT//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

class FaceRegistrationApp:
    POSE_CFG = {
        RegistrationStage.YAW: ("yaw_snapshots", "yaw_left", "yaw_right", "yaw", getattr(config, 'YAW_THRESHOLD', 25.0)), 
        RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up", "pitch_down", "pitch", getattr(config, 'PITCH_THRESHOLD', 20.0)), 
        RegistrationStage.ROLL: ("roll_snapshots", "roll_left", "roll_right", "roll", getattr(config, 'ROLL_THRESHOLD', 25.0))
    }
    STEP_COMMIT = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}

    def __init__(self, name):
        self.name = name
        self.stage = RegistrationStage.FACEMESH
        self.in_ext, self.hold_frames, self.print_counter = False, 0, 0
        self.last_match_score = 0.0 
        
        # Variabel untuk Threading
        self.is_running = True
        self.display_frame = None
        self.frame_lock = threading.Lock()
        
        self.captured_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None}
        self._pose_buf, self._blink_buf, self._prev_step = {"yaw": {}, "pitch": {}, "roll": {}}, {"closed": None, "open": None}, "FACEMESH"
        
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
        
        self.liveness, self.model, self.anti_spoof = LivenessManager(), MobileFaceNet(), SilentAntiSpoofing()
        
        self.matcher = FaceMatcher(threshold=0.60)
        try: self.matcher.known_faces = self.db.load_all_faces()
        except: pass
        
        self.liveness.start_register()
        _log(f"✅ Inisialisasi registrasi untuk: {self.name}", "SYSTEM")

    def _record_data_buffers(self, face, pose):
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
                if self.hold_frames >= 3 and (tag_neg not in buf or val < buf[tag_neg][axis]):
                    buf[tag_neg] = DataExtractor.capture_pose(pose, tag_neg)
            elif val > cap_thr:
                self.hold_frames += 1
                if self.hold_frames >= 3 and (tag_pos not in buf or val > buf[tag_pos][axis]):
                    buf[tag_pos] = DataExtractor.capture_pose(pose, tag_pos)
            else:
                self.hold_frames = 0

        if self.stage == RegistrationStage.BLINK:
            bv = DataExtractor.capture_blink(face)
            if bv:
                if not self._blink_buf["closed"] or bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]: 
                    self._blink_buf["closed"] = bv
                if not self._blink_buf["open"] or bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]: 
                    self._blink_buf["open"] = bv

    def _commit_stage_data(self, cur_step):
        if cur_step in ("WAIT", self._prev_step): return
        
        if self._prev_step in self.STEP_COMMIT:
            axis = self.STEP_COMMIT[self._prev_step]
            if len(self._pose_buf[axis]) < 2:
                if hasattr(self.liveness, '_register_step'): self.liveness._register_step -= 1
                return 
            self.captured_data[self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]] = list(self._pose_buf[axis].values())
            self._pose_buf[axis] = {}  
            
        if cur_step == "DONE" and self._prev_step == "BLINK":
            if not self._blink_buf["closed"] or not self._blink_buf["open"]:
                if hasattr(self.liveness, '_register_step'): self.liveness._register_step -= 1
                return
            self.captured_data.update({"blink_closed": self._blink_buf["closed"], "blink_open": self._blink_buf["open"]})
            
        self._prev_step = cur_step
        self.stage = STEP_TO_STAGE.get(cur_step, self.stage)

    def _process_thread(self):
        """Worker thread untuk menangani deteksi dan pemrosesan utama"""
        try:
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                ret, frame = self.cam.read()
                if not ret: 
                    time.sleep(0.01)
                    continue
                
                display, faces = frame.copy(), self.detector.detect(frame)

                if not faces:
                    RegistrationUI.draw_hud(display, self.stage, "Wajah tidak terdeteksi", "Hadapkan wajah ke kamera", "", "NO FACE", None, config.COLOR_RED)
                    with self.frame_lock:
                        self.display_frame = display
                    continue

                face = faces[0]
                display = self.detector.draw(display, face)
                
                pose = self.liveness.pose_estimator.estimate(face, self.detector)
                spoof = self.anti_spoof.is_real(frame, face.bbox)
                ear_data = DataExtractor.capture_blink(face)
                
                sp_score = spoof.get("score", 0.0)
                ear_val = ear_data["avg_ear"] if ear_data else 0.0
                score_txt = f"Real:{sp_score:.2f} | Y:{pose.get('yaw',0):.1f} P:{pose.get('pitch',0):.1f} R:{pose.get('roll',0):.1f} | EAR:{ear_val:.2f}"
                
                old_stage = self.stage
                current_instruction = self.stage.name 

                if not spoof.get("real", True):
                    current_instruction = "❌ SPOOFING DETECTED"
                    RegistrationUI.draw_hud(display, self.stage, "❌ TERDETEKSI SPOOFING!", f"Skor Palsu: {sp_score:.2f}", score_txt, "SPOOFING", face.bbox, config.COLOR_RED)
                    print(f"\r[{current_instruction}] {score_txt}          ", end="", flush=True)
                    with self.frame_lock:
                        self.display_frame = display
                    continue 

                if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                    self._record_data_buffers(face, pose) 
                    result = self.liveness.update_register(face, self.detector)
                    self._commit_stage_data(result["step"])
                    
                    current_instruction = result["instruction"] 

                    color = config.COLOR_GREEN if result["step"] == "DONE" else config.COLOR_CYAN
                    RegistrationUI.draw_hud(display, self.stage, current_instruction, result.get("progress",""), score_txt, "VALIDATING", face.bbox, color)

                elif self.stage == RegistrationStage.EXTRACTION and not self.in_ext:
                    current_instruction = "4. Ekstraksi Fitur Database"
                    print(f"\r[{current_instruction}] {score_txt}          ", end="", flush=True)
                    self._process_extraction(frame, face, display, pose, score_txt)

                if self.stage != RegistrationStage.EXTRACTION:
                    self.print_counter += 1
                    if self.print_counter % 3 == 0:
                        print(f"\r[{current_instruction}] {score_txt}          ", end="", flush=True)

                if old_stage != self.stage:
                    print() 
                    if old_stage == RegistrationStage.FACEMESH:
                        _log(f"✅ FACEMESH 3D Terekam -> (Wajah Posisi Lurus/Netral)", "SUCCESS")
                    elif old_stage == RegistrationStage.YAW:
                        snaps = self.captured_data.get("yaw_snapshots", [])
                        L = next((s["yaw"] for s in snaps if s["tag"] == "yaw_left"), 0)
                        R = next((s["yaw"] for s in snaps if s["tag"] == "yaw_right"), 0)
                        _log(f"✅ YAW Selesai -> Toleh Kiri (Max Score): {L:.1f}° | Toleh Kanan (Max Score): {R:.1f}°", "SUCCESS")
                    elif old_stage == RegistrationStage.PITCH:
                        snaps = self.captured_data.get("pitch_snapshots", [])
                        U = next((s["pitch"] for s in snaps if s["tag"] == "pitch_up"), 0)
                        D = next((s["pitch"] for s in snaps if s["tag"] == "pitch_down"), 0)
                        _log(f"✅ PITCH Selesai -> Dongak Atas (Max Score): {U:.1f}° | Tunduk Bawah (Max Score): {D:.1f}°", "SUCCESS")
                    elif old_stage == RegistrationStage.ROLL:
                        snaps = self.captured_data.get("roll_snapshots", [])
                        L = next((s["roll"] for s in snaps if s["tag"] == "roll_left"), 0)
                        R = next((s["roll"] for s in snaps if s["tag"] == "roll_right"), 0)
                        _log(f"✅ ROLL Selesai -> Miring Kiri (Max Score): {L:.1f}° | Miring Kanan (Max Score): {R:.1f}°", "SUCCESS")
                    elif old_stage == RegistrationStage.BLINK:
                        c = self.captured_data.get("blink_closed", {}).get("avg_ear", 0)
                        o = self.captured_data.get("blink_open", {}).get("avg_ear", 0)
                        _log(f"✅ BLINK Selesai -> Mata Terbuka (Max EAR): {o:.2f} | Mata Tertutup (Min EAR): {c:.2f}", "SUCCESS")
                    elif old_stage == RegistrationStage.EXTRACTION:
                        _log(f"✅ Ekstraksi MobileFaceNet Selesai -> Skor Kemiripan Database: {self.last_match_score:.4f}", "SUCCESS")
                        if self.last_match_score < 0.50:
                            _log(f"ℹ️  Info: Wajah sangat unik dan aman (belum ada yang mirip).", "INFO")

                # Update frame untuk ditampilkan oleh main thread
                with self.frame_lock:
                    self.display_frame = display

        except Exception as e:
            _log(f"❌ Terjadi Crash di Thread: {e}", "ERROR")
        finally:
            self.is_running = False

    def run(self):
        if self.stage == RegistrationStage.COMPLETE: return
        _log(f"🎬 Memulai registrasi: {self.name}", "SYSTEM")
        
        # Memulai Worker Thread
        processing_thread = threading.Thread(target=self._process_thread, daemon=True)
        processing_thread.start()

        try:
            # Main Thread UI Loop
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                frame_to_show = None
                with self.frame_lock:
                    if self.display_frame is not None:
                        frame_to_show = self.display_frame.copy()
                
                if frame_to_show is not None:
                    cv2.imshow("Register", frame_to_show)
                
                # Menangani input keyboard di main thread
                if cv2.waitKey(1) & 0xFF == ord("q"): 
                    self.is_running = False
                    break
        finally:
            print() 
            self.is_running = False
            processing_thread.join(timeout=1.0) # Tunggu thread tertutup aman
            if GPIO_AVAILABLE: GPIO.cleanup()
            self.cam.stop(); self.detector.close(); cv2.destroyAllWindows()

    def _process_extraction(self, frame, face, display, pose, score_txt):
        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
        if max(abs(yaw), abs(pitch), abs(roll)) >= getattr(config, 'EXTRACTION_MAX_YAW', 12.0):
            RegistrationUI.draw_hud(display, self.stage, "Tatap LURUS ke kamera", "Menunggu posisi netral...", score_txt, "TAHAN LURUS", face.bbox, config.COLOR_YELLOW)
            return

        self.in_ext = True
        
        missing = [k for k, is_valid in [
            ("FaceMesh", self.captured_data["facemesh_vector"] is not None), 
            ("Yaw", len(self.captured_data["yaw_snapshots"]) > 1), 
            ("Pitch", len(self.captured_data["pitch_snapshots"]) > 1), 
            ("Roll", len(self.captured_data["roll_snapshots"]) > 1), 
            ("Blink", self.captured_data["blink_closed"] is not None)
        ] if not is_valid]
        
        if missing:
            RegistrationUI.show_msg(display, "❌ GAGAL!", f"Data Kurang: {', '.join(missing)}", config.COLOR_RED)
            with self.frame_lock: self.display_frame = display.copy()
            time.sleep(4) # Mengganti cv2.waitKey(4000)
            self.stage = RegistrationStage.COMPLETE
            return

        emb = self.model.get_embedding(self.model.crop_face(frame, face.bbox))
        self.captured_data["headpose_vector"] = [float(yaw), float(pitch), float(roll)]
        
        match = self.matcher.match(emb)
        self.last_match_score = match.get("score", 0.0) 
        
        if match["matched"] and os.getenv("REGISTER_ALLOW_DUPLICATE", "false").lower() != "true":
            _log(f"⚠️ WAJAH DUPLIKASI: Score: {match['score']:.4f}", "WARNING")
            RegistrationUI.show_msg(display, "❌ SUDAH TERDAFTAR!", f"Mirip dengan {match['name']} ({match['score']:.2f})", config.COLOR_RED)
            with self.frame_lock: self.display_frame = display.copy()
            time.sleep(4) # Mengganti cv2.waitKey(4000)
            self.stage = RegistrationStage.COMPLETE
            return

        if self.db.save_face(self.name, emb, self.captured_data):
            RegistrationUI.show_msg(display, "✅ BERHASIL!", f"Nama: {self.name}", config.COLOR_GREEN)
            _log(f"✅ Registrasi BERHASIL: {self.name}", "SUCCESS")
        else: RegistrationUI.show_msg(display, "❌ GAGAL!", "Database Error", config.COLOR_RED)
        
        with self.frame_lock: self.display_frame = display.copy()
        time.sleep(3) # Mengganti cv2.waitKey(3000)
        self.stage = RegistrationStage.COMPLETE

if __name__ == "__main__":
    name = input("\nMasukkan Nama Panggilan: ").strip()
    if name: FaceRegistrationApp(name).run()