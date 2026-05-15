import cv2, os, time, threading, numpy as np
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
STEP_TO_STAGE = {"FACEMESH": RegistrationStage.FACEMESH, "YAW": RegistrationStage.YAW, "PITCH": RegistrationStage.PITCH, "ROLL": RegistrationStage.ROLL, "BLINK": RegistrationStage.BLINK, "DONE": RegistrationStage.EXTRACTION}

def _log(msg, level="INFO"): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")

class Helpers:
    @staticmethod
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        if mean_val > 160.0: return frame 
        limit = getattr(config, 'CLAHE_CLIP_LIMIT', 2.5) if mean_val < 85.0 else 1.2
        l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
        return cv2.cvtColor(cv2.merge((cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8)).apply(l), a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def capture_blink(face):
        if not getattr(face, 'landmarks', None) or len(face.landmarks) < 400: return None
        p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        la = (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6)
        ra = (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6)
        return {"left_ear": la, "right_ear": ra, "avg_ear": (la+ra)/2.0}

    @staticmethod
    def draw_hud(frame, stage, instr, prog, score_txt, status, bbox, color):
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.rectangle(frame, (x, y-35), (x+180, y-5), color, -1)
            cv2.putText(frame, status, (x+5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 185), (20,20,20), -1)
        cv2.rectangle(frame, (0, 50), (config.FRAME_WIDTH, 185), config.COLOR_CYAN, 2)
        pts = [(STAGE_NAMES.get(stage, "Proses..."), 75, config.COLOR_GREEN, 0.85, 2), 
               (instr, 105, config.COLOR_YELLOW, 0.65, 2), 
               (prog, 130, config.COLOR_CYAN, 0.6, 1), 
               (score_txt, 160, config.COLOR_WHITE, 0.55, 1)]
        for txt, y_pos, col, sz, thk in pts:
            if txt: cv2.putText(frame, txt, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, sz, col, thk)
        bx, by, bw, bh, sv = (config.FRAME_WIDTH-350)//2, 15, 350, 25, min(stage.value, 6)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (30,30,30), -1)
        if sv > 0: cv2.rectangle(frame, (bx, by), (bx + int(bw*(sv-1)/6), by+bh), config.COLOR_GREEN, -1)
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
    
    def __init__(self, name):
        self.name, self.stage, self.in_ext, self.hold_frames, self.print_counter = name, RegistrationStage.FACEMESH, False, 0, 0
        self.last_match_score, self.fake_frames, self.reg_accuracy = 0.0, 0, 0.0 
        self.is_running, self.display_frame, self.frame_lock = True, None, threading.Lock()
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None}
        self._pose_buf, self._blink_buf, self._prev_step = {"yaw": {}, "pitch": {}, "roll": {}}, {"closed": None, "open": None}, "FACEMESH"
        
        self.db = FaceDatabase()
        if self.db.check_user_exists(self.name): 
            _log(f"❌ '{self.name}' sudah terdaftar!", "ERROR")
            self.stage = RegistrationStage.COMPLETE
            return

        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.liveness = LivenessManager()
        self.model = MobileFaceNet()
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.85))
        self.matcher = FaceMatcher(getattr(config, 'MATCH_THRESHOLD', 0.68))
        
        try:
            raw = self.db.load_all_faces()
            if raw:
                faces = {k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if isinstance(v, dict) and v.get('embedding')}
                if hasattr(self.matcher, 'load_faces'): self.matcher.load_faces(faces)
                else: self.matcher.known_faces = faces
        except Exception as e: 
            _log(f"Warning: {e}", "WARNING")
        
        self.liveness.start_register()
        _log(f"✅ Inisialisasi: {self.name}", "SYSTEM")

    def _record_data_buffers(self, face, pose):
        if self.stage == RegistrationStage.FACEMESH and self.cap_data["facemesh_vector"] is None:
            if face.landmarks:
                fm = np.array([[l.x, l.y, l.z] for l in face.landmarks], dtype=np.float32).flatten()
                self.hold_frames += 1
                if self.hold_frames >= 10: 
                    self.cap_data["facemesh_vector"], self.hold_frames = fm, 0

        if self.stage in self.POSE_CFG:
            _, tag_neg, tag_pos, axis, thr = self.POSE_CFG[self.stage]
            val = pose.get(axis, 0.0)
            buf = self._pose_buf[axis]
            cap_thr = thr * 0.7  
            
            if val < -cap_thr or val > cap_thr:
                self.hold_frames += 1
                tag = tag_neg if val < -cap_thr else tag_pos
                if self.hold_frames >= 3 and (tag not in buf or abs(val) > abs(buf[tag][axis])): 
                    buf[tag] = {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")}
                    buf[tag]["tag"] = tag
            else: 
                self.hold_frames = 0

        if self.stage == RegistrationStage.BLINK:
            bv = Helpers.capture_blink(face)
            if bv:
                if not self._blink_buf["closed"] or bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]: 
                    self._blink_buf["closed"] = bv
                if not self._blink_buf["open"] or bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]: 
                    self._blink_buf["open"] = bv

    def _generate_metric_text(self, pose, ear_val, sp_score):
        stg, y, p, r = self.stage, pose.get('yaw', 0.0), pose.get('pitch', 0.0), pose.get('roll', 0.0)
        return {
            RegistrationStage.FACEMESH: f"Posisi Lurus | Y:{y:.1f}° P:{p:.1f}° R:{r:.1f}° (Batas: < {getattr(config, 'MAX_YAW', 12):.0f}°)",
            RegistrationStage.YAW: f"Yaw: {y:.1f}° | Target: > ±{getattr(config, 'YAW_THRESHOLD', 25):.1f}° | Real: {sp_score:.2f}",
            RegistrationStage.PITCH: f"Pitch: {p:.1f}° | Target: > ±{getattr(config, 'PITCH_THRESHOLD', 20):.1f}° | Real: {sp_score:.2f}",
            RegistrationStage.ROLL: f"Roll: {r:.1f}° | Target: > ±{getattr(config, 'ROLL_THRESHOLD', 25):.1f}° | Real: {sp_score:.2f}",
            RegistrationStage.BLINK: f"EAR: {ear_val:.2f} | Merekam dinamika mata | Real: {sp_score:.2f}"
        }.get(stg, f"Tahan Lurus | Y:{y:.1f}° P:{p:.1f}° R:{r:.1f}°")

    def _commit_stage_data(self, cur_step):
        if cur_step in ("WAIT", self._prev_step): return
        if self._prev_step in {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}:
            axis = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}[self._prev_step]
            if len(self._pose_buf[axis]) < 2: 
                self.liveness._register_step -= 1
                return 
            self.cap_data[self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]] = list(self._pose_buf[axis].values())
            self._pose_buf[axis] = {}  
            
        if cur_step == "DONE" and self._prev_step == "BLINK":
            bc, bo = self._blink_buf["closed"], self._blink_buf["open"]
            if not bc or not bo or (bo["avg_ear"] - bc["avg_ear"]) < 0.03: 
                self.liveness._register_step = 7
                self.liveness._blink_state = 0
                self.liveness._hold_frames = 0
                self.liveness._blink_count = 0
                self._blink_buf = {"closed": None, "open": None}
                return
            self.cap_data.update({"blink_closed": bc, "blink_open": bo})
            
        self._prev_step, self.stage = cur_step, STEP_TO_STAGE.get(cur_step, self.stage)

    def _process_extraction(self, frame, face, display, pose, score_txt):
        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
        max_y = getattr(config, 'EXTRACTION_MAX_YAW', 12.0)
        max_dev = max(abs(yaw), abs(pitch), abs(roll))
        
        if max_dev >= max_y: 
            Helpers.draw_hud(display, self.stage, "Tatap LURUS", "Menunggu...", score_txt, "TAHAN", face.bbox, config.COLOR_YELLOW)
            return
            
        self.reg_accuracy = 100.0 - (max_dev / max_y) * 5.0
        self.in_ext = True
        
        missing = [k for k, v in [
            ("FaceMesh", self.cap_data["facemesh_vector"] is not None), 
            ("Yaw", len(self.cap_data["yaw_snapshots"])>1), 
            ("Pitch", len(self.cap_data["pitch_snapshots"])>1), 
            ("Roll", len(self.cap_data["roll_snapshots"])>1), 
            ("Blink", self.cap_data["blink_closed"] is not None)
        ] if not v]
        
        if missing:
            Helpers.show_msg(display, "❌ GAGAL!", f"Kurang: {','.join(missing)}", config.COLOR_RED)
            time.sleep(4)
            self.stage = RegistrationStage.COMPLETE
            return

        emb = self.model.get_embedding(self.model.crop_face(frame, face.bbox))
        self.cap_data["headpose_vector"] = [float(yaw), float(pitch), float(roll)]
        
        match = self.matcher.match(emb)
        self.last_match_score = match.get("score", 0.0)
        
        if match["matched"] and os.getenv("ALLOW_DUPLICATE", "false").lower() != "true":
            Helpers.show_msg(display, "❌ SUDAH TERDAFTAR!", f"Mirip {match['name']} ({match['score']:.2f})", config.COLOR_RED)
        elif self.db.save_face(self.name, emb, self.cap_data):
            Helpers.show_msg(display, "✅ BERHASIL!", f"Akurasi: {self.reg_accuracy:.2f}%", config.COLOR_GREEN)
            _log(f"SUKSES: {self.name}", "SUCCESS")
        else: 
            Helpers.show_msg(display, "❌ GAGAL!", "Database Error", config.COLOR_RED)
            
        with self.frame_lock: self.display_frame = display.copy()
        time.sleep(3)
        self.stage = RegistrationStage.COMPLETE

    def _log_transition(self, old_stage):
        print()
        def get_snap(key, tag): 
            return next((s.get(key, 0.0) for s in self.cap_data.get(f"{key}_snapshots", []) if s.get("tag") == tag), 0.0)
            
        if old_stage == RegistrationStage.FACEMESH:   
            _log("✅ TAHAP 1: Data FaceMesh 3D Berhasil Terekam (Posisi Lurus)", "SUCCESS")
        elif old_stage == RegistrationStage.YAW:      
            _log(f"✅ TAHAP 2a: Yaw Selesai -> Kiri: {get_snap('yaw','yaw_left'):.1f}° | Kanan: {get_snap('yaw','yaw_right'):.1f}°", "SUCCESS")
        elif old_stage == RegistrationStage.PITCH:    
            _log(f"✅ TAHAP 2b: Pitch Selesai -> Atas: {get_snap('pitch','pitch_up'):.1f}° | Bawah: {get_snap('pitch','pitch_down'):.1f}°", "SUCCESS")
        elif old_stage == RegistrationStage.ROLL:     
            _log(f"✅ TAHAP 2c: Roll Selesai -> Miring Kiri: {get_snap('roll','roll_left'):.1f}° | Miring Kanan: {get_snap('roll','roll_right'):.1f}°", "SUCCESS")
        elif old_stage == RegistrationStage.BLINK:
            c = self.cap_data.get("blink_closed") or {}
            o = self.cap_data.get("blink_open") or {}
            _log(f"✅ TAHAP 3: Blink Selesai -> EAR Buka: {o.get('avg_ear',0):.2f} | EAR Kedip: {c.get('avg_ear',0):.2f}", "SUCCESS")
        elif old_stage == RegistrationStage.EXTRACTION:
            _log("==================================================", "SYSTEM")
            _log("📊 RANGKUMAN ANALISIS BIOMETRIK REGISTRASI", "SYSTEM")
            _log(f"  • Kemiripan Tertinggi di DB (Toleransi max: {getattr(config, 'MATCH_THRESHOLD', 0.68)}) : {self.last_match_score:.2%}", "SUCCESS")
            _log(f"  • Kualitas Ekstraksi Wajah (Akurasi)           : {self.reg_accuracy:.2f}%", "SUCCESS")
            _log(f"  • Status Profil Keamanan Wajah                 : {'SANGAT UNIK & AMAN' if self.last_match_score < 0.50 else 'STANDAR'}", "SUCCESS")
            _log("==================================================", "SYSTEM")

    def _process_thread(self):
        try:
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                ret, frame = self.cam.read()
                if not ret: 
                    time.sleep(0.01)
                    continue
                    
                raw = frame.copy()
                enhanced = Helpers.enhance_frame(raw)
                display = enhanced.copy()
                
                faces = self.detector.detect(enhanced)
                if not faces: 
                    Helpers.draw_hud(display, self.stage, "Hadapkan wajah", "", "", "NO FACE", None, config.COLOR_RED)
                    with self.frame_lock: self.display_frame = display
                    continue
                    
                face = faces[0]
                display = self.detector.draw(display, face)
                
                if face.bbox[3] > int(config.FRAME_HEIGHT * 0.50): 
                    Helpers.draw_hud(display, self.stage, "Wajah Terlalu Dekat!", "Mundur sedikit", "", "TOO CLOSE", face.bbox, config.COLOR_YELLOW)
                    with self.frame_lock: self.display_frame = display
                    continue
                
                pose = self.liveness.pose_estimator.estimate(face, self.detector)
                e_data = Helpers.capture_blink(face)
                ear_val = e_data["avg_ear"] if e_data else 0.0
                
                check_spoof = self.stage in (RegistrationStage.FACEMESH, RegistrationStage.BLINK, RegistrationStage.EXTRACTION)
                if check_spoof:
                    sp = self.anti_spoof.is_real(raw, face.bbox)
                    sp_score = sp.get("score", 0.0)
                    sp_real = sp.get("real", True)
                else:
                    sp_score = 1.0
                    sp_real = True
                    
                score_txt = self._generate_metric_text(pose, ear_val, sp_score)
                
                if check_spoof and not sp_real:
                    self.fake_frames += 1
                    if self.fake_frames >= 3: 
                        Helpers.draw_hud(display, self.stage, "❌ SPOOFING!", f"Palsu: {sp_score:.2f}", score_txt, "SPOOFING", face.bbox, config.COLOR_RED)
                        print(f"\r[SPOOFING] {score_txt}   ", end="")
                        with self.frame_lock: self.display_frame = display
                        continue 
                else: 
                    self.fake_frames = 0

                old_stage = self.stage 
                instr = ""
                if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                    self._record_data_buffers(face, pose) 
                    res = self.liveness.update_register(face, self.detector)
                    self._commit_stage_data(res["step"])
                    hud_col = config.COLOR_GREEN if res["step"] == "DONE" else config.COLOR_CYAN
                    instr = res.get("instruction", "")
                    Helpers.draw_hud(display, self.stage, instr, res.get("progress",""), score_txt, "VALIDATING", face.bbox, hud_col)
                elif not self.in_ext: 
                    instr = "4. Ekstraksi Fitur Database"
                    self._process_extraction(enhanced, face, display, pose, score_txt)

                # ==============================================================
                # MENGEMBALIKAN FITUR SCORE YANG BERJALAN DI TERMINAL
                self.print_counter += 1
                if self.print_counter % 3 == 0 and instr:
                    print(f"\r[{instr}] {score_txt}           ", end="", flush=True)

                if old_stage != self.stage:
                    self._log_transition(old_stage)
                # ==============================================================

                with self.frame_lock: self.display_frame = display
        finally: 
            self.is_running = False

    def run(self):
        if self.stage == RegistrationStage.COMPLETE: return
        threading.Thread(target=self._process_thread, daemon=True).start()
        try:
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                with self.frame_lock: 
                    frame = self.display_frame.copy() if self.display_frame is not None else None
                if frame is not None: cv2.imshow("Register", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): 
                    self.is_running = False
                    break
        finally:
            self.is_running = False
            time.sleep(0.5)
            self.cam.stop()
            self.detector.close()
            cv2.destroyAllWindows()
            if GPIO_AVAILABLE: GPIO.cleanup()

if __name__ == "__main__":
    name = input("\nNama Panggilan: ").strip()
    if name: FaceRegistrationApp(name).run()