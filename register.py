import cv2, os, time, threading, numpy as np, uuid
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
    lvl_str = f"[{level}]".ljust(10)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {lvl_str} {msg}")

class Helpers:
    @staticmethod
    def enhance_adaptive(frame, bbox=None, l_str="Normal"):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        
        matrix_clip = {"Normal": 1.5, "Low Light": 2.0, "Backlight": 1.8}
        clip_limit = matrix_clip.get(l_str, 1.5)
        
        denoised = cv2.bilateralFilter(frame, d=3, sigmaColor=30, sigmaSpace=30)
        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def clone_low_light(img):
        low_light = cv2.convertScaleAbs(img, alpha=0.6, beta=-40)
        return cv2.GaussianBlur(low_light, (3, 3), 0)

    @staticmethod
    def clone_backlight(img):
        backlight = cv2.convertScaleAbs(img, alpha=0.4, beta=-10)
        img_yuv = cv2.cvtColor(backlight, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def capture_blink(face):
        if not getattr(face, 'landmarks', None) or len(face.landmarks) < 400: return None
        p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        la = (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6)
        ra = (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6)
        return {"left_ear": la, "right_ear": ra, "avg_ear": (la+ra)/2.0}

    @staticmethod
    def get_light_condition(raw_frame, bbox):
        if not bbox: return "Normal"
        bx, by, bw, bh = bbox
        fh, fw = raw_frame.shape[:2]
        x1, y1, x2, y2 = max(0, bx), max(0, by), min(fw, bx + bw), min(fh, by + bh)
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        
        L = np.mean(gray[y1:y2, x1:x2]) if gray[y1:y2, x1:x2].size > 0 else 100.0
        top_bg = gray[0:max(0, y1-10), max(0, x1-30):min(fw, x2+30)]
        L_bg_atas = np.mean(top_bg) if top_bg.size > 0 else L
        
        if (L_bg_atas - L) > 50 and L_bg_atas > 160 and L < 110: 
            return "Backlight"
        return "Low Light" if (L_bg_atas < 95 or L < 95) else "Normal"

    @staticmethod
    def is_image_quality_good(frame, bbox):
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0: return False, 0.0, 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return (50 < brightness < 200) and (blur_score > 80), brightness, blur_score

    @staticmethod
    def draw_hud(f, stg, instr, prog, score_txt, status, bbox, col):
        h, w = f.shape[:2]
        if bbox:
            bx, by, bw, bh = bbox
            bx = w - bx - bw
            cv2.rectangle(f, (bx, by), (bx+bw, by+bh), col, 2)
            lbl_h = 20
            lbl_y = max(lbl_h, by)
            cv2.rectangle(f, (bx, lbl_y - lbl_h), (bx + 110, lbl_y), col, -1)
            cv2.putText(f, status, (bx + 5, lbl_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            
        cv2.rectangle(f, (0, 0), (w, 32), (25, 25, 25), -1)
        stage_txt = STAGE_NAMES.get(stg, "Proses...")
        cv2.putText(f, stage_txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.COLOR_GREEN, 1, cv2.LINE_AA)
        
        sv = min(stg.value, 6)
        bw_bar, bh_bar = 160, 14
        bx_bar, by_bar = w - bw_bar - 10, 9
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (45, 45, 45), -1)
        if sv > 0:
            cv2.rectangle(f, (bx_bar, by_bar), (bx_bar + int(bw_bar*(sv-1)/6), by_bar+bh_bar), config.COLOR_GREEN, -1)
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (255,255,255), 1)
        cv2.putText(f, f"Tahap {sv if sv<=5 else 6}/6", (bx_bar + 45, by_bar + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

        cv2.rectangle(f, (0, h - 70), (w, h), (15, 15, 15), -1)
        cv2.line(f, (0, h - 70), (w, h - 70), config.COLOR_CYAN, 1)
        
        y_pos = h - 52
        if instr:
            cv2.putText(f, instr, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA)
            y_pos += 18
        if prog:
            cv2.putText(f, prog, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, config.COLOR_CYAN, 1, cv2.LINE_AA)
            y_pos += 16
        if score_txt:
            cv2.putText(f, score_txt, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def show_msg(f, t_title, t_sub, col):
        h, w = f.shape[:2]
        cv2.rectangle(f, (0, 0), (w, h), (15, 15, 15), -1)
        cv2.rectangle(f, (10, 10), (w - 10, h - 10), col, 4)
        cv2.putText(f, t_title, (25, h // 2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
        lines = t_sub.split(" | ")
        y_offset = h // 2 + 10
        for line in lines:
            cv2.putText(f, line, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            y_offset += 22

class FaceRegistrationApp:
    POSE_CFG = {RegistrationStage.YAW: ("yaw_snapshots", "yaw_left", "yaw_right", "yaw", getattr(config, 'YAW_THRESHOLD', 25.0)), RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up", "pitch_down", "pitch", getattr(config, 'PITCH_THRESHOLD', 20.0)), RegistrationStage.ROLL: ("roll_snapshots", "roll_left", "roll_right", "roll", getattr(config, 'ROLL_THRESHOLD', 25.0))}
    
    def __init__(self, name):
        self.name = name
        self.user_id = str(uuid.uuid4())
        self.stage = RegistrationStage.FACEMESH
        self.in_ext, self.hold_frames, self.missed_frames = False, 0, 0
        self.fake_frames = 0
        self.is_running, self.display_frame, self.frame_lock = True, None, threading.Lock()
        self.locked_light_cond = None  # Variabel Pengunci Cahaya
        
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None, "face_crops": []}
        self._pose_buf, self._blink_buf, self._prev_step = {"yaw": {}, "pitch": {}, "roll": {}}, {"closed": None, "open": None, "logged_closed": False, "logged_open": False}, "FACEMESH"
        
        self.db = FaceDatabase()
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
        except Exception as e: pass
        
        self.liveness.start_register()
        self.action_start_time = None
        self._timer_started = False
        self.prev_instruction = None
        self.prev_tag = None
        self.individual_latencies = {}  
        _log(f"✅ Memulai Registrasi untuk {self.name}...", "SYSTEM")

    def _reset_registration(self, display, t_title, t_sub):
        Helpers.show_msg(display, t_title, f"{t_sub} | Mengulang dari awal...", config.COLOR_RED)
        with self.frame_lock: self.display_frame = display.copy()
        _log(f"🔄 Registrasi diulang karena: {t_sub}", "WARNING")
        time.sleep(3.0)
        
        self.stage = RegistrationStage.FACEMESH
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None, "face_crops": []}
        self._pose_buf = {"yaw": {}, "pitch": {}, "roll": {}}
        self._blink_buf = {"closed": None, "open": None, "logged_closed": False, "logged_open": False}
        self._prev_step = "FACEMESH"
        
        self.hold_frames = 0
        self.missed_frames = 0
        self.fake_frames = 0
        self.individual_latencies = {}
        self.action_start_time = time.time()
        self.prev_instruction = None
        self.prev_tag = None
        self.locked_light_cond = None
        
        self.liveness.start_register()

    def _record_data_buffers(self, face, pose, enhanced):
        if not self.action_start_time: return
        
        if self.stage == RegistrationStage.FACEMESH and face.landmarks:
            if self.cap_data["facemesh_vector"] is None:
                self.hold_frames += 1
                
                if self.hold_frames % 2 == 0 and len(self.cap_data["face_crops"]) < 5:
                    crop = self.model.crop_face(enhanced, face.bbox)
                    if crop is not None and crop.size > 0:
                        idx = len(self.cap_data["face_crops"]) + 1
                        self.cap_data["face_crops"].append((f"Frontal_{idx}", crop))
                
                if self.hold_frames >= 10: 
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    self.cap_data["facemesh_vector"] = np.array([[l.x, l.y, l.z] for l in face.landmarks], dtype=np.float32).flatten()
                    
                    if not self.cap_data["face_crops"]:
                        crop = self.model.crop_face(enhanced, face.bbox)
                        if crop is not None and crop.size > 0: 
                            self.cap_data["face_crops"].append(("Frontal_Fallback", crop))

                    self.hold_frames = 0
                    self.individual_latencies["FaceMesh (3D)"] = lat_ms
                    _log(f"   -> [Wajah 3D Terekam (5 Sampel)] | Latensi: {lat_ms:>8.2f} ms", "SUCCESS")
                    self.action_start_time = time.time() 

        if self.stage in self.POSE_CFG:
            _, t_neg, t_pos, axis, thr = self.POSE_CFG[self.stage]
            val = pose.get(axis, 0.0)
            buf = self._pose_buf[axis]
            if val < -thr or val > thr:
                tag = t_neg if val < -thr else t_pos
                if getattr(self, 'prev_tag', None) != tag:
                    self.hold_frames = 0
                    self.prev_tag = tag
                self.hold_frames += 1
                if self.hold_frames >= 6:
                    if tag not in buf:
                        lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                        buf[tag] = {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")}
                        buf[tag]["tag"], buf[tag]["latency_ms"] = tag, lat_ms 
                        friendly_name = {"yaw_left": "Toleh Kiri", "yaw_right": "Toleh Kanan", "pitch_up": "Angguk Atas", "pitch_down": "Tunduk Bawah", "roll_left": "Miring Kiri", "roll_right": "Miring Kanan"}.get(tag, tag)
                        self.individual_latencies[friendly_name] = lat_ms
                        
                        _log(f"   -> [Berhasil {friendly_name:<12}] | Latensi: {lat_ms:>8.2f} ms", "SUCCESS")
                        self.action_start_time = time.time() 
                        self.hold_frames = 0
            else: 
                self.hold_frames = 0
                self.prev_tag = None

        if self.stage == RegistrationStage.BLINK:
            bv = Helpers.capture_blink(face)
            if bv:
                min_open, max_closed = getattr(config, 'MIN_BLINK_OPEN_EAR', 0.22), getattr(config, 'MAX_BLINK_CLOSED_EAR', 0.20)
                if bv["avg_ear"] <= max_closed and not self._blink_buf.get("logged_closed"):
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    bv["latency_ms"] = lat_ms
                    self._blink_buf["closed"], self._blink_buf["logged_closed"] = bv, True
                    self.individual_latencies["Mata Menutup"] = lat_ms
                    _log(f"   -> [Berhasil Kedip Tutup]  | Latensi: {lat_ms:>8.2f} ms", "SUCCESS")
                    self.action_start_time = time.time() 
                elif bv["avg_ear"] >= min_open and self._blink_buf.get("logged_closed") and not self._blink_buf.get("logged_open"):
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    bv["latency_ms"] = lat_ms
                    self._blink_buf["open"], self._blink_buf["logged_open"] = bv, True
                    self.individual_latencies["Mata Membuka"] = lat_ms
                    _log(f"   -> [Berhasil Kedip Buka]   | Latensi: {lat_ms:>8.2f} ms", "SUCCESS")
                    self.action_start_time = time.time() 
                if self._blink_buf.get("closed") and bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]: self._blink_buf["closed"].update(bv)
                if self._blink_buf.get("open") and bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]: self._blink_buf["open"].update(bv)

    def _generate_metric_text(self, pose, ear_val, sp_score, light_cond):
        stg = self.stage
        hud_txt = {
            RegistrationStage.FACEMESH: f"Menganalisa 3D | Cahaya: {light_cond}",
            RegistrationStage.YAW: f"Aksi: Toleh Kanan/Kiri | Cahaya: {light_cond}",
            RegistrationStage.PITCH: f"Aksi: Angguk Atas/Bawah | Cahaya: {light_cond}",
            RegistrationStage.ROLL: f"Aksi: Miring Kanan/Kiri | Cahaya: {light_cond}",
            RegistrationStage.BLINK: f"Aksi: Kedipkan Mata | Cahaya: {light_cond}"
        }.get(stg, f"Tahan Lurus | {light_cond}")
        return hud_txt, f"{hud_txt} | Spf: {sp_score:.2f}"

    def _commit_stage_data(self, cur_step):
        if cur_step in ("WAIT", self._prev_step): return
        if self._prev_step in {"YAW", "PITCH", "ROLL"}:
            axis = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}[self._prev_step]
            if len(self._pose_buf[axis]) < 2: self.liveness._register_step -= 1; return 
            self.cap_data[self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]], self._pose_buf[axis] = list(self._pose_buf[axis].values()), {}  
            
        if cur_step == "DONE" and self._prev_step == "BLINK":
            bc, bo = self._blink_buf["closed"], self._blink_buf["open"]
            min_open, max_closed, min_delta = getattr(config, 'MIN_BLINK_OPEN_EAR', 0.22), getattr(config, 'MAX_BLINK_CLOSED_EAR', 0.20), getattr(config, 'MIN_BLINK_DELTA', 0.04)
            if not bc or not bo or (bo["avg_ear"] < min_open) or (bc["avg_ear"] > max_closed) or (bo["avg_ear"] - bc["avg_ear"] < min_delta): 
                self.liveness._register_step, self.liveness._blink_state, self.liveness._hold_frames, self.liveness._blink_count, self._blink_buf = 7, 0, 0, 0, {"closed": None, "open": None, "logged_closed": False, "logged_open": False}; return
            self.cap_data.update({"blink_closed": bc, "blink_open": bo})
        
        self._prev_step, self.stage = cur_step, STEP_TO_STAGE.get(cur_step, self.stage)
        
        if self.stage != RegistrationStage.COMPLETE:
            self.action_start_time = time.time()
            self.hold_frames = 0

    def _process_extraction(self, raw_frame, frame, face, display, pose, score_txt, sp_score, sp_label):
        missing = [k for k, v in [("FaceMesh", self.cap_data["facemesh_vector"] is not None), ("Yaw", len(self.cap_data["yaw_snapshots"])>1), ("Pitch", len(self.cap_data["pitch_snapshots"])>1), ("Roll", len(self.cap_data["roll_snapshots"])>1), ("Blink", self.cap_data["blink_closed"] is not None)] if not v]
        if missing: 
            self._reset_registration(display, "❌ GAGAL EKTRAKSI!", f"Data Kurang: {','.join(missing)}")
            return

        quality_ok, brightness, blur_score = Helpers.is_image_quality_good(raw_frame, face.bbox)
        if not quality_ok:
            reason = "Cahaya Buruk" if not (50 < brightness < 200) else "Kamera Blur"
            Helpers.show_msg(display, "⚠️ KUALITAS GAMBAR BURUK", f"{reason} | Harap diam", config.COLOR_YELLOW)
            with self.frame_lock: self.display_frame = display.copy()
            time.sleep(1.5); return

        light_cond = getattr(self, 'locked_light_cond', "Normal")
        latensi_respon_subjek = round(sum(self.individual_latencies.values()), 2)
        t_mfn_start = time.time()
        
        if not self.cap_data.get("face_crops"):
            crop_cadangan = self.model.crop_face(frame, face.bbox)
            if crop_cadangan is not None: self.cap_data["face_crops"].append(("Frontal Akhir", crop_cadangan))
            
        best_crop = None
        max_blur = -1.0
        valid_embeddings = []

        for tag, crop in self.cap_data["face_crops"]:
            if crop is not None and crop.size > 0:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                b_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                if b_score > max_blur:
                    max_blur = b_score
                    best_crop = crop
                
                emb = self.model.get_embedding(crop)
                if emb is not None:
                    valid_embeddings.append(np.array(emb).flatten())

        if best_crop is None:
            _, best_crop = self.cap_data["face_crops"][0]

        if len(valid_embeddings) > 0:
            emb_normal = np.mean(valid_embeddings, axis=0) 
        else:
            emb_normal = self.model.get_embedding(best_crop)
            
        crop_low_light = Helpers.clone_low_light(best_crop)
        crop_backlight = Helpers.clone_backlight(best_crop)
        
        emb_low_light = self.model.get_embedding(crop_low_light)
        emb_backlight = self.model.get_embedding(crop_backlight)
        
        if emb_normal is None or emb_low_light is None or emb_backlight is None:
            self._reset_registration(display, "❌ GAGAL!", "Ekstraksi AI Gagal")
            return
            
        def norm_emb(e):
            e_arr = np.array(e).flatten()
            return (e_arr / (np.linalg.norm(e_arr) + 1e-6)).tolist()
            
        multi_master_embs = [norm_emb(emb_normal), norm_emb(emb_low_light), norm_emb(emb_backlight)]

        mfn_latency = round((time.time() - t_mfn_start) * 1000, 2)
        total_waktu_sistem = latensi_respon_subjek + mfn_latency
        
        # --- PERBAIKAN UTAMA: SISTEM ANTI-DUPLIKAT MULTI-VECTOR CROSS MATCH MENYELURUH ---
        is_duplicate = False
        duplicate_name = ""
        anti_dup_thr = getattr(config, 'ANTI_DUPLICATE_THRESHOLD', 0.48)
        best_sim_score = 0.0
        
        all_faces_raw = self.db.load_all_faces()
        if all_faces_raw:
            for name, data in all_faces_raw.items():
                if isinstance(data, dict) and 'embedding' in data:
                    emb_list = data['embedding']
                    if isinstance(emb_list, list) and len(emb_list) > 0:
                        if not isinstance(emb_list[0], (list, np.ndarray)):
                            emb_list = [emb_list]
                        
                        # Mengecek SEMUA vektor hasil ekstraksi vs SEMUA vektor di database
                        for q_emb in multi_master_embs:
                            q_vec = np.array(q_emb, dtype=np.float32)
                            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-6)
                            
                            for db_emb in emb_list:
                                db_vec = np.array(db_emb, dtype=np.float32)
                                db_vec = db_vec / (np.linalg.norm(db_vec) + 1e-6)
                                sim = np.dot(q_vec, db_vec)
                                
                                if sim > best_sim_score:
                                    best_sim_score = sim
                                    duplicate_name = name.split(" - ", 1)[-1]
                                    
                                if sim >= anti_dup_thr:
                                    is_duplicate = True
                                    break
                            if is_duplicate: break
                if is_duplicate: break

        if is_duplicate and os.getenv("ALLOW_DUPLICATE", "false").lower() != "true": 
            sim_percent = best_sim_score * 100
            Helpers.show_msg(display, "❌ WAJAH SUDAH TERDAFTAR!", f"Mirip {sim_percent:.1f}% dgn {duplicate_name}", config.COLOR_RED)
            with self.frame_lock: self.display_frame = display.copy()
            _log(f"Registrasi DIBATALKAN! Wajah mirip {sim_percent:.1f}% dengan user: {duplicate_name}", "WARNING")
            time.sleep(4.0)
            self.stage = RegistrationStage.COMPLETE
            return
        else:
            self.cap_data["reg_latency_ms"] = total_waktu_sistem
            self.cap_data["individual_latencies"] = self.individual_latencies
            self.cap_data.update({"headpose_vector": [float(pose["yaw"]), float(pose["pitch"]), float(pose["roll"])], "registration_accuracy": 100.0, "light_condition": light_cond})
            if "face_crops" in self.cap_data: del self.cap_data["face_crops"]
            
            success_master = self.db.save_face(self.name, self.user_id, multi_master_embs, self.cap_data)
            
            if success_master: 
                Helpers.show_msg(display, "✅ REGISTRASI BERHASIL!", f"User: {self.name} | Multi-Vector", config.COLOR_GREEN)
                
                print("\n" + "="*65)
                print(" 🎉 REGISTRASI MULTI-VECTOR BERHASIL 🎉".center(65))
                print("="*65)
                print(f" 👤 Nama User             : {self.name}")
                print(f" 🔑 User ID (UUID)        : {self.user_id}")
                print(f" 💡 Kondisi Live          : {light_cond}")
                print(f" 🧠 Strategi              : Centroid Averaging ({len(valid_embeddings)} Sampel) + Kloning Matriks 2D")
                print("-" * 65)
                print(f" ⏱️ Latensi Respon Subjek : {latensi_respon_subjek:.2f} ms")
                print(f" ⏱️ Ekstraksi AI          : {mfn_latency:.2f} ms")
                print("="*65 + "\n")
                with self.frame_lock: self.display_frame = display.copy()
                time.sleep(1.5)
                self.stage = RegistrationStage.COMPLETE 
            else:
                self._reset_registration(display, "❌ GAGAL!", "Database Error")

    def _process_thread(self):
        try:
            bbox_memory = None
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                ret, frame = self.cam.read()
                if not ret: 
                    time.sleep(0.01)
                    continue
                    
                raw = frame.copy()
                display = raw.copy()
                faces = self.detector.detect(raw)
                
                if not faces: 
                    self.missed_frames += 1
                    if self.missed_frames >= 5: 
                        bbox_memory = None
                        display = cv2.flip(display, 1)
                        Helpers.draw_hud(display, self.stage, "Hadapkan wajah", "", "", "NO FACE", None, config.COLOR_RED)
                        self._timer_started = False
                    elif bbox_memory: 
                        display = cv2.flip(display, 1)
                        Helpers.draw_hud(display, self.stage, "Menganalisa...", "", "", "TRACKING", bbox_memory, config.COLOR_YELLOW)
                    with self.frame_lock: 
                        self.display_frame = display
                    continue
                    
                self.missed_frames, face = 0, faces[0]
                bbox_memory = face.bbox
                
                # --- PERBAIKAN: LIGHT LOCKING REGISTRASI ---
                current_light = Helpers.get_light_condition(raw, face.bbox)
                if self.stage == RegistrationStage.FACEMESH or getattr(self, 'locked_light_cond', None) is None:
                    self.locked_light_cond = current_light
                light_cond = self.locked_light_cond
                
                enhanced = Helpers.enhance_adaptive(raw, face.bbox, light_cond)
                
                if not self._timer_started: 
                    self.action_start_time, self._timer_started = time.time(), True
                    
                display = cv2.flip(self.detector.draw(display, face), 1)
                
                if face.bbox[3] > int(config.FRAME_HEIGHT * 0.50): 
                    Helpers.draw_hud(display, self.stage, "Wajah Terlalu Dekat!", "Mundur", "", "TOO CLOSE", face.bbox, config.COLOR_YELLOW)
                    with self.frame_lock: 
                        self.display_frame = display
                    continue
                    
                pose, ear_val = self.liveness.pose_estimator.estimate(face, self.detector), (Helpers.capture_blink(face) or {}).get("avg_ear", 0.0)
                sp = self.anti_spoof.is_real(raw, face.bbox)
                sp_score, sp_real, sp_label = float(sp.get("score_real", sp.get("score", 1.0))), sp.get("real", True), sp.get("label_name", "FOTO").upper()
                hud_txt, term_txt = self._generate_metric_text(pose, ear_val, sp_score, light_cond)
                
                if not sp_real:
                    self.fake_frames += 1
                    if self.fake_frames >= 4: 
                        Helpers.draw_hud(display, self.stage, "❌ DETEKSI SPOOFING!", f"Palsu: {sp_score:.2f}", hud_txt, f"{sp_label}", face.bbox, config.COLOR_RED)
                        with self.frame_lock: 
                            self.display_frame = display
                        continue 
                else: 
                    self.fake_frames = 0
                    
                if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                    self._record_data_buffers(face, pose, enhanced) 
                    res = self.liveness.update_register(face, self.detector)
                    if res["step"] != "WAIT" and res["step"] != self._prev_step and self.stage in self.POSE_CFG and len(self._pose_buf[self.POSE_CFG[self.stage][3]]) < 2: 
                        res["step"] = self._prev_step 
                    self._commit_stage_data(res["step"])
                    if self.prev_instruction is None or res.get("instruction", "") != self.prev_instruction: 
                        self.prev_instruction, self.action_start_time, self.hold_frames = res.get("instruction", ""), time.time(), 0
                    Helpers.draw_hud(display, self.stage, res.get("instruction", ""), res.get("progress",""), hud_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_GREEN if res["step"] == "DONE" else config.COLOR_CYAN)
                elif not self.in_ext: 
                    self._process_extraction(raw, enhanced, face, display, pose, hud_txt, sp_score, sp_label)
                    
                with self.frame_lock: 
                    self.display_frame = display
        finally: 
            self.is_running = False
            
    def run(self):
        if self.stage == RegistrationStage.COMPLETE: return
        threading.Thread(target=self._process_thread, daemon=True).start()
        try:
            cv2.namedWindow("Register", cv2.WINDOW_AUTOSIZE)
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                with self.frame_lock: frame = self.display_frame.copy() if self.display_frame is not None else None
                if frame is not None: cv2.imshow("Register", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): self.is_running = False; break
        finally: self.is_running = False; time.sleep(0.5); self.cam.stop(); self.detector.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"\n{'='*45}\n   SISTEM REGISTRASI WAJAH (MULTI-VECTOR)\n{'='*45}")
    if nama_input := input("Masukan Nama Lengkap : ").strip(): FaceRegistrationApp(nama_input).run()