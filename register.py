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

class RegistrationStage(Enum): IDLE=0; FACEMESH=1; YAW=2; PITCH=3; ROLL=4; BLINK=5; EXTRACTION=6; COMPLETE=7
STAGE_NAMES = {RegistrationStage.FACEMESH: "1. FaceMesh (3D)", RegistrationStage.YAW: "2a. Liveness (Yaw)", RegistrationStage.PITCH: "2b. Liveness (Pitch)", RegistrationStage.ROLL: "2c. Liveness (Roll)", RegistrationStage.BLINK: "3. Liveness (Blink)", RegistrationStage.EXTRACTION: "4. Ekstraksi Fitur"}
STEP_TO_STAGE = {"FACEMESH": RegistrationStage.FACEMESH, "YAW": RegistrationStage.YAW, "PITCH": RegistrationStage.PITCH, "ROLL": RegistrationStage.ROLL, "BLINK": RegistrationStage.BLINK, "DONE": RegistrationStage.EXTRACTION}

def _log(msg, level="INFO"): 
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")

class Helpers:
    @staticmethod
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        
        denoised = cv2.bilateralFilter(frame, d=3, sigmaColor=30, sigmaSpace=30)
        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def capture_blink(face):
        if not getattr(face, 'landmarks', None) or len(face.landmarks) < 400: return None
        p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        la = (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6)
        ra = (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6)
        return {"left_ear": la, "right_ear": ra, "avg_ear": (la+ra)/2.0}

    @staticmethod
    def draw_hud(f, stg, instr, prog, score_txt, status, bbox, col):
        if bbox:
            bx, by, bw, bh = bbox
            bx = config.FRAME_WIDTH - bx - bw
            cv2.rectangle(f, (bx, by), (bx+bw, by+bh), col, 3)
            cv2.rectangle(f, (bx, by-35), (bx+200, by-5), col, -1)
            cv2.putText(f, status, (bx+5, by-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.rectangle(f, (0, 50), (config.FRAME_WIDTH, 185), (20,20,20), -1); cv2.rectangle(f, (0, 50), (config.FRAME_WIDTH, 185), config.COLOR_CYAN, 2)
        for txt, yp, c, sz, t in [(STAGE_NAMES.get(stg, "Proses..."), 75, config.COLOR_GREEN, 0.85, 2), (instr, 105, config.COLOR_YELLOW, 0.65, 2), (prog, 130, config.COLOR_CYAN, 0.6, 1), (score_txt, 160, config.COLOR_WHITE, 0.55, 1)]:
            if txt: cv2.putText(f, txt, (20, yp), cv2.FONT_HERSHEY_SIMPLEX, sz, c, t)
        bx_bar, by_bar, bw_bar, bh_bar, sv = (config.FRAME_WIDTH-350)//2, 15, 350, 25, min(stg.value, 6)
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (30,30,30), -1)
        if sv > 0: cv2.rectangle(f, (bx_bar, by_bar), (bx_bar + int(bw_bar*(sv-1)/6), by_bar+bh_bar), config.COLOR_GREEN, -1)
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), config.COLOR_WHITE, 2)
        cv2.putText(f, f"Tahap {sv if sv<=5 else 6}/6", (bx_bar+130, by_bar+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    @staticmethod
    def show_msg(f, t_title, t_sub, col):
        cv2.rectangle(f, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), (15, 15, 15), -1)
        cv2.rectangle(f, (15, 15), (config.FRAME_WIDTH - 15, config.FRAME_HEIGHT - 15), col, 8)
        cv2.putText(f, t_title, (40, config.FRAME_HEIGHT // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        lines = t_sub.split(" | ")
        y_offset = config.FRAME_HEIGHT // 2 + 10
        for line in lines:
            cv2.putText(f, line, (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
            y_offset += 35

class FaceRegistrationApp:
    POSE_CFG = {RegistrationStage.YAW: ("yaw_snapshots", "yaw_left", "yaw_right", "yaw", getattr(config, 'YAW_THRESHOLD', 25.0)), RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up", "pitch_down", "pitch", getattr(config, 'PITCH_THRESHOLD', 20.0)), RegistrationStage.ROLL: ("roll_snapshots", "roll_left", "roll_right", "roll", getattr(config, 'ROLL_THRESHOLD', 25.0))}
    
    def __init__(self, name):
        self.name, self.stage, self.in_ext, self.hold_frames, self.missed_frames = name, RegistrationStage.FACEMESH, False, 0, 0
        self.last_match_score, self.fake_frames, self.latency = 0.0, 0, 0.0 
        self.ext_embs = [] 
        
        self.is_running, self.display_frame, self.frame_lock = True, None, threading.Lock()
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None}
        self._pose_buf, self._blink_buf, self._prev_step = {"yaw": {}, "pitch": {}, "roll": {}}, {"closed": None, "open": None}, "FACEMESH"
        
        self.db = FaceDatabase()
        
        check_nim = self.name.split(" - ")[0] if " - " in self.name else self.name
        if self.db.check_user_exists(check_nim): 
            _log(f"❌ '{self.name}' sudah terdaftar!", "ERROR")
            self.stage = RegistrationStage.COMPLETE
            return

        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.liveness, self.model = LivenessManager(), MobileFaceNet()
        
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70))
        self.matcher = FaceMatcher(0.35) 
        
        try:
            raw = self.db.load_all_faces()
            if raw:
                faces = {k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if isinstance(v, dict) and v.get('embedding') is not None}
                self.matcher.load_faces(faces) if hasattr(self.matcher, 'load_faces') else setattr(self.matcher, 'known_faces', faces)
        except Exception as e: _log(f"Warning: {e}", "WARNING")
        
        self.liveness.start_register()
        
        # --- TRACKING WAKTU REGISTRASI ---
        self.reg_start_time = time.time()
        self.stage_start_time = time.time()  
        self.stage_durations = {}            
        
        _log(f"✅ Inisialisasi: {self.name} dimulai. Silakan tatap layar.", "SYSTEM")

    def _record_data_buffers(self, face, pose):
        if self.stage == RegistrationStage.FACEMESH and self.cap_data["facemesh_vector"] is None and face.landmarks:
            self.hold_frames += 1
            if self.hold_frames >= 5: self.cap_data["facemesh_vector"], self.hold_frames = np.array([[l.x, l.y, l.z] for l in face.landmarks], dtype=np.float32).flatten(), 0

        if self.stage in self.POSE_CFG:
            _, t_neg, t_pos, axis, thr = self.POSE_CFG[self.stage]
            val = pose.get(axis, 0.0)
            buf = self._pose_buf[axis]
            cap_thr = thr * 0.7  
            
            if val < -cap_thr or val > cap_thr:
                self.hold_frames += 1
                tag = t_neg if val < -cap_thr else t_pos
                if self.hold_frames >= 6 and (tag not in buf or abs(val) > abs(buf[tag][axis])): 
                    buf[tag] = {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")}
                    buf[tag]["tag"] = tag
                    buf[tag]["latency_ms"] = self.latency 
            else: self.hold_frames = 0

        if self.stage == RegistrationStage.BLINK:
            bv = Helpers.capture_blink(face)
            if bv:
                bv["latency_ms"] = self.latency 
                if not self._blink_buf["closed"] or bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]: self._blink_buf["closed"] = bv
                if not self._blink_buf["open"] or bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]: self._blink_buf["open"] = bv

    def _generate_metric_text(self, pose, ear_val, sp_score, light_cond):
        stg, y, p, r = self.stage, pose.get('yaw', 0.0), pose.get('pitch', 0.0), pose.get('roll', 0.0)
        
        hud_txt = {
            RegistrationStage.FACEMESH: f"Lurus | Y:{y:.1f}° P:{p:.1f}° R:{r:.1f}° | {light_cond}",
            RegistrationStage.YAW: f"Yaw: {y:.1f}° | Tgt: > ±{getattr(config, 'YAW_THRESHOLD', 25):.0f}° | {light_cond}",
            RegistrationStage.PITCH: f"Pitch: {p:.1f}° | Tgt: > ±{getattr(config, 'PITCH_THRESHOLD', 20):.0f}° | {light_cond}",
            RegistrationStage.ROLL: f"Roll: {r:.1f}° | Tgt: > ±{getattr(config, 'ROLL_THRESHOLD', 25):.0f}° | {light_cond}",
            RegistrationStage.BLINK: f"EAR: {ear_val:.2f} | Deteksi Mata | {light_cond}"
        }.get(stg, f"Tahan Lurus | Y:{y:.1f}° P:{p:.1f}° R:{r:.1f}° | {light_cond}")
        
        term_txt = f"{hud_txt} | Spf: {sp_score:.2f}"
        return hud_txt, term_txt

    def _commit_stage_data(self, cur_step):
        if cur_step in ("WAIT", self._prev_step): return
        if self._prev_step in {"YAW", "PITCH", "ROLL"}:
            axis = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}[self._prev_step]
            if len(self._pose_buf[axis]) < 2: self.liveness._register_step -= 1; return 
            self.cap_data[self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]], self._pose_buf[axis] = list(self._pose_buf[axis].values()), {}  
            
        if cur_step == "DONE" and self._prev_step == "BLINK":
            bc, bo = self._blink_buf["closed"], self._blink_buf["open"]
            min_open   = getattr(config, 'MIN_BLINK_OPEN_EAR', 0.22)
            max_closed = getattr(config, 'MAX_BLINK_CLOSED_EAR', 0.20)
            min_delta  = getattr(config, 'MIN_BLINK_DELTA', 0.04)
            
            if not bc or not bo or (bo["avg_ear"] < min_open) or (bc["avg_ear"] > max_closed) or (bo["avg_ear"] - bc["avg_ear"] < min_delta): 
                _log(f"⚠️ Kualitas Kedipan Ditolak! (Melek: {(bo or {}).get('avg_ear', 0):.2f}, Merem: {(bc or {}).get('avg_ear', 0):.2f}). Silakan ulangi.", "WARNING")
                self.liveness._register_step, self.liveness._blink_state, self.liveness._hold_frames, self.liveness._blink_count, self._blink_buf = 7, 0, 0, 0, {"closed": None, "open": None}; return
            
            self.cap_data.update({"blink_closed": bc, "blink_open": bo})
        self._prev_step, self.stage = cur_step, STEP_TO_STAGE.get(cur_step, self.stage)

    def _process_extraction(self, raw_frame, frame, face, display, pose, score_txt, sp_score):
        missing = [k for k, v in [("FaceMesh", self.cap_data["facemesh_vector"] is not None), ("Yaw", len(self.cap_data["yaw_snapshots"])>1), ("Pitch", len(self.cap_data["pitch_snapshots"])>1), ("Roll", len(self.cap_data["roll_snapshots"])>1), ("Blink", self.cap_data["blink_closed"] is not None)] if not v]
        if missing: Helpers.show_msg(display, "❌ GAGAL!", f"Kurang: {','.join(missing)}", config.COLOR_RED); time.sleep(4); self.stage = RegistrationStage.COMPLETE; return

        bx, by, bw, bh = face.bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(fw, bx + bw), min(fh, by + bh)
        safe_bbox = [x1, y1, x2 - x1, y2 - y1]

        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y1:y2, x1:x2]
        L = np.mean(face_roi) if face_roi.size > 0 else 100.0
        
        mask = np.ones((fh, fw), dtype=bool)
        mask[y1:y2, x1:x2] = False
        L_bg = np.mean(gray[mask]) if np.any(mask) else L
        
        if (L_bg - L) > 40 and L_bg > 120: light_cond = f"Backlight (B:{L_bg:.0f})"
        elif L_bg < 85 or L < 85: light_cond = f"Low Light (B:{L_bg:.0f})"
        else: light_cond = f"Normal (B:{L_bg:.0f})"

        raw_emb = self.model.get_embedding(self.model.crop_face(frame, safe_bbox))
        
        if raw_emb is not None:
            emb_flat = np.array(raw_emb, dtype=np.float32).flatten()
            emb_norm = emb_flat / (np.linalg.norm(emb_flat) + 1e-6)
            self.ext_embs.append(emb_norm)
        
        self.in_ext = True
        avg_emb = np.mean(self.ext_embs, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-6) 
        
        self.cap_data.update({"headpose_vector": [float(pose["yaw"]), float(pose["pitch"]), float(pose["roll"])], "registration_accuracy": 100.0, "light_condition": light_cond})
        
        match = self.matcher.match(avg_emb)
        self.last_match_score = match.get("score", 0.0)
        
        anti_dup_thr = getattr(config, 'ANTI_DUPLICATE_THRESHOLD', 0.48)
        nim_user, nama_user = self.name.split(" - ", 1) if " - " in self.name else ("0000", self.name)
        
        if match.get("name") and self.last_match_score >= anti_dup_thr and os.getenv("ALLOW_DUPLICATE", "false").lower() != "true": 
            msg_sub = f"User: {match['name']} (Sim: {match['score']:.4f}) | Kondisi: {light_cond}"
            Helpers.show_msg(display, "❌ WAJAH SUDAH TERDAFTAR!", msg_sub, config.COLOR_RED)
            _log(f"GAGAL: Terdeteksi duplikat dgn {match['name']} (Sim: {match['score']:.4f}) | Kondisi: {light_cond}", "ERROR")
            
        else:
            # --- WAKTU DIBULATKAN KE 2 DESIMAL ---
            total_time = round(time.time() - self.reg_start_time, 2)
            self.cap_data["reg_latency_sec"] = total_time
            
            ext_duration = round(time.time() - self.stage_start_time, 2)
            self.stage_durations[RegistrationStage.EXTRACTION.name] = ext_duration
            self.cap_data["stage_durations_exact"] = self.stage_durations 
            
            if self.db.save_face(nama_user, nim_user, avg_emb.tolist(), self.cap_data): 
                msg_sub = f"User: {nama_user} | Vektor Tersimpan | Kondisi: {light_cond}"
                Helpers.show_msg(display, "✅ REGISTRASI BERHASIL!", msg_sub, config.COLOR_GREEN)
                _log(f"SUKSES: Data untuk '{nama_user}' ({nim_user}) berhasil disimpan! (Total Waktu: {total_time}s)", "SUCCESS")
            else: 
                Helpers.show_msg(display, "❌ GAGAL!", "DB Error", config.COLOR_RED)
            
        with self.frame_lock: self.display_frame = display.copy()
        time.sleep(1.5); self.stage = RegistrationStage.COMPLETE 

    def _log_transition(self, old_stage):
        # --- PERHITUNGAN WAKTU PER TAHAP DIBULATKAN (2 DESIMAL) ---
        stage_duration = round(time.time() - self.stage_start_time, 2)
        self.stage_durations[old_stage.name] = stage_duration
        self.stage_start_time = time.time() # Reset stopwatch
        
        if old_stage == RegistrationStage.FACEMESH: 
            _log(f"✅ TAHAP 1 (FaceMesh): {stage_duration} detik", "SUCCESS")
        elif old_stage == RegistrationStage.YAW: 
            _log(f"✅ TAHAP 2a (Yaw): {stage_duration} detik", "SUCCESS")
        elif old_stage == RegistrationStage.PITCH: 
            _log(f"✅ TAHAP 2b (Pitch): {stage_duration} detik", "SUCCESS")
        elif old_stage == RegistrationStage.ROLL: 
            _log(f"✅ TAHAP 2c (Roll): {stage_duration} detik", "SUCCESS")
        elif old_stage == RegistrationStage.BLINK: 
            _log(f"✅ TAHAP 3 (Blink): {stage_duration} detik", "SUCCESS")
        elif old_stage == RegistrationStage.EXTRACTION:
            _log("📊 RANGKUMAN REGISTRASI KOMPREHENSIF", "SYSTEM")
            _log(f"   • Waktu Total Registrasi: {self.cap_data.get('reg_latency_sec', 0.0)} detik", "SUCCESS")

    def _process_thread(self):
        try:
            bbox_memory = None
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                t_start = time.time()
                
                ret, frame = self.cam.read()
                if not ret: time.sleep(0.01); continue
                
                raw = frame.copy()
                enhanced = Helpers.enhance_frame(raw)
                display = raw.copy()
                
                faces = self.detector.detect(enhanced)
                
                if not faces: 
                    self.missed_frames += 1
                    if self.missed_frames >= 5: 
                        bbox_memory = None
                        display = cv2.flip(display, 1) 
                        Helpers.draw_hud(display, self.stage, "Hadapkan wajah", "", "", "NO FACE", None, config.COLOR_RED)
                    elif bbox_memory: 
                        display = cv2.flip(display, 1) 
                        Helpers.draw_hud(display, self.stage, "Menganalisa...", "", "", "TRACKING", bbox_memory, config.COLOR_YELLOW)
                    with self.frame_lock: self.display_frame = display
                    continue
                
                self.missed_frames = 0
                face = faces[0]
                bbox_memory = face.bbox
                
                display = self.detector.draw(display, face)
                display = cv2.flip(display, 1)
                
                if face.bbox[3] > int(config.FRAME_HEIGHT * 0.50): 
                    Helpers.draw_hud(display, self.stage, "Wajah Terlalu Dekat!", "Mundur", "", "TOO CLOSE", face.bbox, config.COLOR_YELLOW)
                    with self.frame_lock: self.display_frame = display
                    continue
                
                pose = self.liveness.pose_estimator.estimate(face, self.detector)
                ear_val = (Helpers.capture_blink(face) or {}).get("avg_ear", 0.0)
                
                chk_spf = self.stage in (RegistrationStage.FACEMESH, RegistrationStage.BLINK, RegistrationStage.EXTRACTION)
                
                if chk_spf:
                    sp = self.anti_spoof.is_real(raw, face.bbox)
                    sp_score = float(sp.get("score_real", sp.get("score", 1.0)))
                    sp_photo = float(sp.get("score_photo", 0.0))
                    sp_video = float(sp.get("score_video", 0.0))
                    sp_real = sp.get("real", True)
                    sp_label = sp.get("label_name", "FOTO/VIDEO").upper() 
                    sp_lat = float(sp.get("latency_ms", 0.0))
                else:
                    sp_score, sp_real, sp_label, sp_lat = 1.0, True, "ASLI", 0.0
                
                bx, by, bw, bh = face.bbox
                fh_l, fw_l = raw.shape[:2]
                x1_l, y1_l, x2_l, y2_l = max(0, bx), max(0, by), min(fw_l, bx+bw), min(fh_l, by+bh)
                
                gray_live = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                face_roi_live = gray_live[y1_l:y2_l, x1_l:x2_l]
                L_live = np.mean(face_roi_live) if face_roi_live.size > 0 else 100.0
                
                mask_live = np.ones((fh_l, fw_l), dtype=bool)
                mask_live[y1_l:y2_l, x1_l:x2_l] = False
                L_bg_live = np.mean(gray_live[mask_live]) if np.any(mask_live) else L_live

                if (L_bg_live - L_live) > 40 and L_bg_live > 120: l_str = f"Backlight (B:{L_bg_live:.0f})"
                elif L_bg_live < 85 or L_live < 85: l_str = f"Low Light (B:{L_bg_live:.0f})"
                else: l_str = f"Normal (B:{L_bg_live:.0f})"

                hud_txt, term_txt = self._generate_metric_text(pose, ear_val, sp_score, l_str)
                
                if chk_spf and not sp_real:
                    self.fake_frames += 1
                    Helpers.draw_hud(display, self.stage, "❌ DETEKSI SPOOFING!", f"Palsu: {sp_score:.2f} | Latensi: {sp_lat}ms", hud_txt, f"{sp_label} (Spoof: {sp_score:.2f})", face.bbox, config.COLOR_RED)
                    
                    if self.fake_frames >= 10:
                        current_time = time.time()
                        if current_time - getattr(self, 'last_spoof_log_time', 0) > 5.0:
                            _log(f"❌ SPOOFING TERDETEKSI: {sp_label} (Latensi: {sp_lat}ms)", "ERROR")
                            if hasattr(self.db, 'log_spoofing_async'):
                                self.db.log_spoofing_async(sp_score, sp_photo, sp_video, sp_label, sp_lat)
                            self.last_spoof_log_time = current_time

                    with self.frame_lock: self.display_frame = display
                    continue 
                else: 
                    self.fake_frames = 0

                old_stage, instr = self.stage, ""
                if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                    self._record_data_buffers(face, pose) 
                    res = self.liveness.update_register(face, self.detector)
                    self._commit_stage_data(res["step"])
                    instr = res.get("instruction", "")
                    hud_col = config.COLOR_GREEN if res["step"] == "DONE" else config.COLOR_CYAN
                    hud_status = "VALIDATING" if not chk_spf else f"Spoof: {sp_score:.2f}"
                    Helpers.draw_hud(display, self.stage, instr, res.get("progress",""), hud_txt, hud_status, face.bbox, hud_col)
                elif not self.in_ext: 
                    instr = "4. Ekstraksi Fitur"
                    self._process_extraction(raw, enhanced, face, display, pose, hud_txt, sp_score)

                self.latency = (time.time() - t_start) * 1000.0

                if old_stage != self.stage: 
                    self._log_transition(old_stage)
                    
                with self.frame_lock: self.display_frame = display
        finally: 
            self.is_running = False

    def run(self):
        if self.stage == RegistrationStage.COMPLETE: return
        threading.Thread(target=self._process_thread, daemon=True).start()
        try:
            cv2.namedWindow("Register", cv2.WINDOW_AUTOSIZE)

            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                with self.frame_lock: 
                    frame = self.display_frame.copy() if self.display_frame is not None else None
                if frame is not None: cv2.imshow("Register", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): 
                    self.is_running = False; break
        finally: 
            self.is_running = False
            time.sleep(0.5)
            self.cam.stop()
            self.detector.close()
            cv2.destroyAllWindows()
            try:
                if GPIO_AVAILABLE: GPIO.cleanup()
            except RuntimeWarning:
                pass 

if __name__ == "__main__":
    print("\n" + "="*40)
    print("   SISTEM REGISTRASI WAJAH")
    print("="*40)
    
    nama_input = input("Masukan Nama : ").strip()
    
    if nama_input:
        nim_input = input("Masukan NIM  : ").strip()
        
        if nim_input:
            full_identity = f"{nim_input} - {nama_input}"
            FaceRegistrationApp(full_identity).run()
        else:
            print("❌ Registrasi Dibatalkan: NIM tidak boleh kosong!")
    else:
        print("❌ Registrasi Dibatalkan: Nama tidak boleh kosong!")