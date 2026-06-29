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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {f'[{level}]'.ljust(10)} {msg}")

class Helpers:
    @staticmethod
    def enhance_adaptive(frame, bbox=None, l_str="Normal"):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        d, sig_color, sig_space = {"Normal": (3, 30, 30), "Low Light": (5, 60, 60), "Backlight": (5, 45, 45)}.get(l_str, (3, 30, 30))
        yuv = cv2.cvtColor(cv2.bilateralFilter(frame, d, sig_color, sig_space), cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.createCLAHE(clipLimit={"Normal":1.5, "Low Light":2.0, "Backlight":1.8}.get(l_str, 1.5), tileGridSize=(8, 8)).apply(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def simulate_lowlight_raw(crop, level="medium"):
        if level == "medium":
            dark = cv2.convertScaleAbs(crop, alpha=0.45, beta=15)
            sigma = 8
        else:
            dark = cv2.convertScaleAbs(crop, alpha=0.25, beta=5)
            sigma = 15
        row, col, ch = dark.shape
        mean = 0
        gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.int16)
        noisy = np.clip(dark.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
        return noisy

    @staticmethod
    def simulate_backlight_raw(crop, level="medium"):
        if level == "medium":
            dark = cv2.convertScaleAbs(crop, alpha=0.35, beta=10)
            sat_factor = 0.75
        else:
            dark = cv2.convertScaleAbs(crop, alpha=0.20, beta=5)
            sat_factor = 0.55
        hsv = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= sat_factor
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def enhance_crop(crop, l_str="Normal"):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return crop
        d, sig_color, sig_space = {"Normal": (3, 30, 30), "Low Light": (5, 60, 60), "Backlight": (5, 45, 45)}.get(l_str, (3, 30, 30))
        filtered = cv2.bilateralFilter(crop, d, sig_color, sig_space)
        yuv = cv2.cvtColor(filtered, cv2.COLOR_BGR2YUV)
        clip_limit = {"Normal": 1.5, "Low Light": 2.0, "Backlight": 1.8}.get(l_str, 1.5)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_aligned_crop(frame, face, target_size=(112, 112)):
        lm = getattr(face, 'landmarks', [])
        fh, fw = frame.shape[:2]
        if not lm or len(lm) < 363: 
            bx, by, bw, bh = face.bbox
            return frame[max(0, by):min(fh, by+bh), max(0, bx):min(fw, bx+bw)]
        le = np.array([(lm[33].x + lm[133].x) * fw / 2, (lm[33].y + lm[133].y) * fh / 2])
        re = np.array([(lm[263].x + lm[362].x) * fw / 2, (lm[263].y + lm[362].y) * fh / 2])
        angle = np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))
        scale = 0.3 * target_size[0] / (np.linalg.norm(re - le) + 1e-6)
        M = cv2.getRotationMatrix2D(((le[0] + re[0]) / 2, (le[1] + re[1]) / 2), angle, scale)
        M[0, 2] += (target_size[0] * 0.5 - ((le[0] + re[0]) / 2))
        M[1, 2] += (target_size[1] * 0.40 - ((le[1] + re[1]) / 2))
        return cv2.warpAffine(frame, M, target_size, flags=cv2.INTER_CUBIC)

    @staticmethod
    def capture_blink(face):
        if not getattr(face, 'landmarks', None) or len(face.landmarks) < 400: return None
        p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        la = (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6)
        ra = (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6)
        return {"left_ear": la, "right_ear": ra, "avg_ear": (la+ra)/2.0}

    @staticmethod
    def get_light_condition_dynamic(raw, bbox=None):
        fh, fw = raw.shape[:2]; gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY); ambient = np.mean(gray)
        if bbox:
            bx, by, bw, bh = bbox; x1, y1, x2, y2 = max(0, bx), max(0, by), min(fw, bx+bw), min(fh, by+bh)
            face_bright = np.mean(gray[y1:y2, x1:x2]) if gray[y1:y2, x1:x2].size > 0 else ambient
            bg_top = gray[max(0, by-80):max(0, by-5), max(0, bx-30):min(fw, bx+bw+30)]
            if (bg_bright := np.mean(bg_top) if bg_top.size > 0 else ambient) > 150 and (bg_bright - face_bright) > 45 and face_bright < 120: return "Backlight"
        return "Low Light" if ambient < 65 else "Normal"

    @staticmethod
    def is_image_quality_good(frame, bbox):
        x, y, w, h = bbox; fh, fw = frame.shape[:2]; x1, y1, x2, y2 = max(0, x), max(0, y), min(fw, x+w), min(fh, y+h)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0: return False, 0.0, 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray); blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return (40 < brightness < 210) and (blur_score > 60), brightness, blur_score 

    @staticmethod
    def draw_hud(f, stg, instr, prog, score_txt, status, bbox, col):
        h, w = f.shape[:2]
        if bbox:
            bx, by, bw, bh = bbox; bx = w - bx - bw
            cv2.rectangle(f, (bx, by), (bx+bw, by+bh), col, 2)
            cv2.rectangle(f, (bx, max(20, by) - 20), (bx + 110, max(20, by)), col, -1)
            cv2.putText(f, status, (bx + 5, max(20, by) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(f, (0, 0), (w, 32), (25, 25, 25), -1)
        cv2.putText(f, STAGE_NAMES.get(stg, "Proses..."), (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.COLOR_GREEN, 1, cv2.LINE_AA)
        sv = min(stg.value, 6); bw_bar, bh_bar = 160, 14; bx_bar, by_bar = w - bw_bar - 10, 9
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (45, 45, 45), -1)
        if sv > 0: cv2.rectangle(f, (bx_bar, by_bar), (bx_bar + int(bw_bar*(sv-1)/6), by_bar+bh_bar), config.COLOR_GREEN, -1)
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (255,255,255), 1)
        cv2.putText(f, f"Tahap {sv if sv<=5 else 6}/6", (bx_bar + 45, by_bar + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(f, (0, h - 70), (w, h), (15, 15, 15), -1)
        cv2.line(f, (0, h - 70), (w, h - 70), config.COLOR_CYAN, 1)
        y_pos = h - 52
        if instr: cv2.putText(f, instr, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA); y_pos += 18
        if prog: cv2.putText(f, prog, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, config.COLOR_CYAN, 1, cv2.LINE_AA); y_pos += 16
        if score_txt: cv2.putText(f, score_txt, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def show_msg(f, t_title, t_sub, col):
        h, w = f.shape[:2]; y_offset = h // 2 + 10
        cv2.rectangle(f, (0, 0), (w, h), (15, 15, 15), -1)
        cv2.rectangle(f, (10, 10), (w - 10, h - 10), col, 4)
        cv2.putText(f, t_title, (25, h // 2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
        for line in t_sub.split(" | "):
            cv2.putText(f, line, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            y_offset += 22

class FaceRegistrationApp:
    POSE_CFG = {RegistrationStage.YAW: ("yaw_snapshots", "yaw_left", "yaw_right", "yaw", getattr(config, 'YAW_THRESHOLD', 25.0)), RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up", "pitch_down", "pitch", getattr(config, 'PITCH_THRESHOLD', 20.0)), RegistrationStage.ROLL: ("roll_snapshots", "roll_left", "roll_right", "roll", getattr(config, 'ROLL_THRESHOLD', 25.0))}
    
    def __init__(self, name):
        self.name, self.user_id = name, str(uuid.uuid4())
        self.stage = RegistrationStage.FACEMESH
        self.in_ext, self.hold_frames, self.missed_frames = False, 0, 0
        self.fake_frames = 0
        self.is_running, self.display_frame, self.frame_lock = True, None, threading.Lock()
        self.locked_light_cond = None 
        
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None}
        self._pose_buf, self._blink_buf, self._prev_step = {"yaw": {}, "pitch": {}, "roll": {}}, {"closed": None, "open": None, "logged_closed": False, "logged_open": False}, "FACEMESH"
        self.extraction_embeddings, self.last_extraction_time = [], 0
        
        self.db = FaceDatabase()
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        self.detector = FaceMeshDetector(min_detection_confidence=0.35, min_tracking_confidence=0.35)
        self.liveness, self.model = LivenessManager(), MobileFaceNet()
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), min(getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.80), 0.80))
        self.matcher = FaceMatcher(0.35) 
        
        try:
            if (raw := self.db.load_all_faces()):
                faces = {k: v.get('embedding') for k, v in raw.items() if isinstance(v, dict) and v.get('embedding') is not None}
                self.matcher.load_faces(faces) if hasattr(self.matcher, 'load_faces') else setattr(self.matcher, 'known_faces', faces)
        except: pass
        
        self.liveness.start_register()
        self.action_start_time, self._timer_started, self.app_start_time = None, False, time.time()
        self.prev_instruction, self.prev_tag, self.individual_latencies = None, None, {}
        _log(f"✅ Memulai Registrasi untuk {self.name}...", "SYSTEM")

    def _reset_registration(self, display, t_title, t_sub):
        Helpers.show_msg(display, t_title, f"{t_sub} | Mengulang dari awal...", config.COLOR_RED)
        with self.frame_lock: self.display_frame = display.copy()
        time.sleep(3.0)
        self.stage = RegistrationStage.FACEMESH
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None}
        self._pose_buf = {"yaw": {}, "pitch": {}, "roll": {}}
        self._blink_buf = {"closed": None, "open": None, "logged_closed": False, "logged_open": False}
        self._prev_step = "FACEMESH"
        self.extraction_embeddings, self.last_extraction_time = [], 0
        self.hold_frames = self.missed_frames = self.fake_frames = 0
        self.individual_latencies, self.action_start_time = {}, time.time()
        self.prev_instruction, self.prev_tag, self.locked_light_cond = None, None, None
        self.liveness.start_register()

    def _record_data_buffers(self, face, pose, enhanced):
        if not self.action_start_time: return
        if self.stage == RegistrationStage.FACEMESH and face.landmarks:
            if self.cap_data["facemesh_vector"] is None:
                self.hold_frames += 1
                if self.hold_frames >= 4: 
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    self.cap_data["facemesh_vector"] = np.array([[l.x, l.y, l.z] for l in face.landmarks], dtype=np.float32).flatten()
                    self.hold_frames = 0; self.individual_latencies["FaceMesh (3D)"] = lat_ms
                    _log(f"⏱️ Selesai Tahap FaceMesh (3D) | Latensi: {lat_ms:.2f} ms", "METRIK")
                    self.action_start_time = time.time() 

        if self.stage in self.POSE_CFG:
            _, t_neg, t_pos, axis, thr = self.POSE_CFG[self.stage]
            val, buf = pose.get(axis, 0.0), self._pose_buf[axis]
            if val < -thr or val > thr:
                tag = t_neg if val < -thr else t_pos
                if getattr(self, 'prev_tag', None) != tag:
                    self.hold_frames = 0; self.prev_tag = tag
                self.hold_frames += 1
                if self.hold_frames >= 3:
                    if tag not in buf:
                        lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                        buf[tag] = {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")}
                        buf[tag]["tag"], buf[tag]["latency_ms"] = tag, lat_ms 
                        friendly = {"yaw_left": "Toleh Kiri", "yaw_right": "Toleh Kanan", "pitch_up": "Angguk Atas", "pitch_down": "Tunduk Bawah", "roll_left": "Miring Kiri", "roll_right": "Miring Kanan"}.get(tag, tag)
                        self.individual_latencies[friendly] = lat_ms
                        _log(f"⏱️ Selesai Tahap Pose ({friendly}) | Latensi: {lat_ms:.2f} ms", "METRIK")
                        self.action_start_time, self.hold_frames = time.time(), 0
            else: self.hold_frames, self.prev_tag = 0, None

        if self.stage == RegistrationStage.BLINK:
            if (bv := Helpers.capture_blink(face)):
                min_open, max_closed = getattr(config, 'MIN_BLINK_OPEN_EAR', 0.22), getattr(config, 'MAX_BLINK_CLOSED_EAR', 0.20)
                if bv["avg_ear"] <= max_closed and not self._blink_buf.get("logged_closed"):
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    bv["latency_ms"] = lat_ms
                    self._blink_buf["closed"], self._blink_buf["logged_closed"] = bv, True
                    self.individual_latencies["Mata Menutup"] = lat_ms
                    _log(f"⏱️ Selesai Tahap Blink (Mata Menutup) | Latensi: {lat_ms:.2f} ms", "METRIK")
                    self.action_start_time = time.time() 
                elif bv["avg_ear"] >= min_open and self._blink_buf.get("logged_closed") and not self._blink_buf.get("logged_open"):
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    bv["latency_ms"] = lat_ms
                    self._blink_buf["open"], self._blink_buf["logged_open"] = bv, True
                    self.individual_latencies["Mata Membuka"] = lat_ms
                    _log(f"⏱️ Selesai Tahap Blink (Mata Membuka) | Latensi: {lat_ms:.2f} ms", "METRIK")
                    self.action_start_time = time.time() 
                if self._blink_buf.get("closed") and bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]: self._blink_buf["closed"].update(bv)
                if self._blink_buf.get("open") and bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]: self._blink_buf["open"].update(bv)

    def _generate_metric_text(self, pose, ear_val, sp_score, light_cond):
        stg = self.stage
        hud_txt = {RegistrationStage.FACEMESH: f"Menganalisa 3D | Cahaya: {light_cond}", RegistrationStage.YAW: f"Aksi: Toleh Kanan/Kiri | Cahaya: {light_cond}", RegistrationStage.PITCH: f"Aksi: Angguk Atas/Bawah | Cahaya: {light_cond}", RegistrationStage.ROLL: f"Aksi: Miring Kanan/Kiri | Cahaya: {light_cond}", RegistrationStage.BLINK: f"Aksi: Kedipkan Mata | Cahaya: {light_cond}"}.get(stg, f"Tahan Lurus | {light_cond}")
        return hud_txt, f"{hud_txt} | Spf: {sp_score:.2f}"

    def _commit_stage_data(self, cur_step, display):
        if cur_step in ("WAIT", self._prev_step): return False
        if self._prev_step in {"YAW", "PITCH", "ROLL"}:
            axis = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}[self._prev_step]
            if len(self._pose_buf[axis]) < 2:
                stage_name = {"YAW": "Toleh Kiri/Kanan", "PITCH": "Angguk Atas/Bawah", "ROLL": "Miring Kiri/Kanan"}[self._prev_step]
                self._reset_registration(display, f"❌ GAGAL {self._prev_step}!", f"Deteksi {stage_name} Kurang/Terlalu Cepat")
                return True
            self.cap_data[self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]], self._pose_buf[axis] = list(self._pose_buf[axis].values()), {}  
            
        if cur_step == "DONE" and self._prev_step == "BLINK":
            bc, bo = self._blink_buf["closed"], self._blink_buf["open"]
            min_open, max_closed, min_delta = getattr(config, 'MIN_BLINK_OPEN_EAR', 0.22), getattr(config, 'MAX_BLINK_CLOSED_EAR', 0.20), getattr(config, 'MIN_BLINK_DELTA', 0.04)
            if not bc or not bo or (bo["avg_ear"] < min_open) or (bc["avg_ear"] > max_closed) or (bo["avg_ear"] - bc["avg_ear"] < min_delta): 
                self.liveness._register_step, self.liveness._blink_state, self.liveness._hold_frames, self.liveness._blink_count, self._blink_buf = 7, 0, 0, 0, {"closed": None, "open": None, "logged_closed": False, "logged_open": False}; return False
            self.cap_data.update({"blink_closed": bc, "blink_open": bo})
        
        self._prev_step, self.stage = cur_step, STEP_TO_STAGE.get(cur_step, self.stage)
        if self.stage != RegistrationStage.COMPLETE:
            self.action_start_time, self.hold_frames = time.time(), 0

    def _process_extraction(self, raw_frame, frame, face, display, pose, score_txt, sp_score, sp_label):
        missing = [k for k, v in [("FaceMesh", self.cap_data["facemesh_vector"] is not None), ("Yaw", len(self.cap_data["yaw_snapshots"])>1), ("Pitch", len(self.cap_data["pitch_snapshots"])>1), ("Roll", len(self.cap_data["roll_snapshots"])>1), ("Blink", self.cap_data["blink_closed"] is not None)] if not v]
        if missing: 
            self._reset_registration(display, "❌ GAGAL EKTRAKSI!", f"Data Kurang: {','.join(missing)}")
            return
        quality_ok, brightness, blur_score = Helpers.is_image_quality_good(raw_frame, face.bbox)
        if not quality_ok:
            reason = "Cahaya Buruk" if not (40 < brightness < 210) else "Kamera Blur"
            Helpers.draw_hud(display, self.stage, f"⚠️ {reason}!", "Mohon tetap diam...", score_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_YELLOW)
            with self.frame_lock: self.display_frame = display.copy()
            return
        current_time = time.time()
        if current_time - self.last_extraction_time < 0.1:
            collected = len(self.extraction_embeddings)
            Helpers.draw_hud(display, self.stage, f"🧠 EKSTRAKSI FITUR WAJAH ({int((collected / 5) * 100)}%)", f"Mengambil data {collected}/5. Tahan posisi...", score_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_CYAN)
            with self.frame_lock: self.display_frame = display.copy()
            return
        self.last_extraction_time = current_time

        if (raw_crop := Helpers.get_aligned_crop(raw_frame, face, target_size=(112, 112))) is not None and raw_crop.size > 0:
            crop_normal = Helpers.enhance_crop(raw_crop, "Normal")
            emb_normal = self.model.get_embedding(crop_normal)

            raw_low_med = Helpers.simulate_lowlight_raw(raw_crop, "medium")
            crop_low_med = Helpers.enhance_crop(raw_low_med, "Low Light")
            emb_low_med = self.model.get_embedding(crop_low_med)

            raw_low_ext = Helpers.simulate_lowlight_raw(raw_crop, "extreme")
            crop_low_ext = Helpers.enhance_crop(raw_low_ext, "Low Light")
            emb_low_ext = self.model.get_embedding(crop_low_ext)

            raw_back_med = Helpers.simulate_backlight_raw(raw_crop, "medium")
            crop_back_med = Helpers.enhance_crop(raw_back_med, "Backlight")
            emb_back_med = self.model.get_embedding(crop_back_med)

            raw_back_str = Helpers.simulate_backlight_raw(raw_crop, "strong")
            crop_back_str = Helpers.enhance_crop(raw_back_str, "Backlight")
            emb_back_str = self.model.get_embedding(crop_back_str)

            if all(v is not None for v in (emb_normal, emb_low_med, emb_low_ext, emb_back_med, emb_back_str)):
                self.extraction_embeddings.append({
                    "normal": np.array(emb_normal).flatten(),
                    "low_med": np.array(emb_low_med).flatten(),
                    "low_ext": np.array(emb_low_ext).flatten(),
                    "back_med": np.array(emb_back_med).flatten(),
                    "back_str": np.array(emb_back_str).flatten()
                })

        total_frames_needed = 5
        collected = len(self.extraction_embeddings)
        if collected < total_frames_needed:
            Helpers.draw_hud(display, self.stage, f"🧠 EKSTRAKSI FITUR WAJAH ({int((collected / total_frames_needed) * 100)}%)", f"Mengambil data {collected}/{total_frames_needed}. Tahan posisi...", score_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_CYAN)
            with self.frame_lock: self.display_frame = display.copy()
            return

        norm_list = [e["normal"] for e in self.extraction_embeddings]
        low_med_list = [e["low_med"] for e in self.extraction_embeddings]
        low_ext_list = [e["low_ext"] for e in self.extraction_embeddings]
        back_med_list = [e["back_med"] for e in self.extraction_embeddings]
        back_str_list = [e["back_str"] for e in self.extraction_embeddings]

        avg_norm = np.mean(norm_list, axis=0)
        avg_low_med = np.mean(low_med_list, axis=0)
        avg_low_ext = np.mean(low_ext_list, axis=0)
        avg_back_med = np.mean(back_med_list, axis=0)
        avg_back_str = np.mean(back_str_list, axis=0)

        vec_norm = (avg_norm / (np.linalg.norm(avg_norm) + 1e-6)).tolist()
        vec_low_med = (avg_low_med / (np.linalg.norm(avg_low_med) + 1e-6)).tolist()
        vec_low_ext = (avg_low_ext / (np.linalg.norm(avg_low_ext) + 1e-6)).tolist()
        vec_back_med = (avg_back_med / (np.linalg.norm(avg_back_med) + 1e-6)).tolist()
        vec_back_str = (avg_back_str / (np.linalg.norm(avg_back_str) + 1e-6)).tolist()

        final_emb_vectors = [vec_norm, vec_low_med, vec_low_ext, vec_back_med, vec_back_str]
        light_cond = getattr(self, 'locked_light_cond', "Normal")
        latensi_respon_subjek = round(sum(self.individual_latencies.values()), 2)
        mfn_latency = round((time.time() - self.action_start_time) * 1000, 2)
        total_waktu_sistem = latensi_respon_subjek + mfn_latency
        
        is_duplicate = False
        duplicate_name = ""
        anti_dup_thr = getattr(config, 'ANTI_DUPLICATE_THRESHOLD', getattr(config, 'MATCH_THRESHOLD', 0.65))
        best_sim_score = 0.0
        all_faces_raw = self.db.load_all_faces()
        existing_user_id = self.user_id
        single_master_emb = final_emb_vectors

        if all_faces_raw:
            user_exists = any(db_key.split(" - ", 1)[-1].lower() == self.name.lower() if " - " in db_key else db_key.lower() == self.name.lower() for db_key in all_faces_raw)
            if user_exists and os.getenv("ALLOW_DUPLICATE", "false").lower() != "true":
                Helpers.show_msg(display, "❌ NAMA SUDAH TERDAFTAR!", f"Nama '{self.name}' sudah digunakan", config.COLOR_RED)
                with self.frame_lock: self.display_frame = display.copy()
                time.sleep(4.0); self.stage = RegistrationStage.COMPLETE; return

            if not user_exists:
                emb_dim = getattr(self.model, 'embedding_size', 128)
                for name, data in all_faces_raw.items():
                    if isinstance(data, dict) and 'embedding' in data:
                        emb_list = data['embedding']
                        if isinstance(emb_list, list) and len(emb_list) > 0:
                            if not isinstance(emb_list[0], (list, np.ndarray)): 
                                emb_list = [emb_list[0:emb_dim], emb_list[emb_dim:emb_dim*2], emb_list[emb_dim*2:emb_dim*3]] if len(emb_list) == (emb_dim * 3) else [emb_list]
                            for q_emb in [vec_norm]: 
                                q_vec = np.array(q_emb, dtype=np.float32); q_vec /= (np.linalg.norm(q_vec) + 1e-6)
                                for db_emb in emb_list:
                                    if len(db_emb) != emb_dim: continue
                                    db_vec = np.array(db_emb, dtype=np.float32); db_vec /= (np.linalg.norm(db_vec) + 1e-6)
                                    sim = np.dot(q_vec, db_vec)
                                    if sim > best_sim_score: best_sim_score, duplicate_name = sim, name.split(" - ", 1)[-1]
                                    if sim >= anti_dup_thr: is_duplicate = True; break
                                if is_duplicate: break
                    if is_duplicate: break

        if is_duplicate and os.getenv("ALLOW_DUPLICATE", "false").lower() != "true": 
            Helpers.show_msg(display, "❌ WAJAH SUDAH TERDAFTAR!", f"Mirip {best_sim_score * 100:.1f}% dgn {duplicate_name}", config.COLOR_RED)
            with self.frame_lock: self.display_frame = display.copy()
            time.sleep(4.0); self.stage = RegistrationStage.COMPLETE; return
        else:
            self.cap_data.update({"reg_latency_ms": total_waktu_sistem, "individual_latencies": self.individual_latencies, "headpose_vector": [float(pose["yaw"]), float(pose["pitch"]), float(pose["roll"])], "registration_accuracy": 100.0, "light_condition": light_cond})
            if "face_crops" in self.cap_data: del self.cap_data["face_crops"]
            if self.db.save_face(self.name, existing_user_id, single_master_emb, self.cap_data): 
                _log(f"⏱️ Ekstraksi AI & Simpan DB Selesai | Latensi AI: {mfn_latency:.2f} ms", "METRIK")
                _log(f"✅ Total Waktu Seluruh Registrasi | Latensi Sistem Total: {total_waktu_sistem:.2f} ms", "METRIK")
                Helpers.show_msg(display, "✅ REGISTRASI BERHASIL!", f"User: {self.name} | 3 Vektor Cahaya", config.COLOR_GREEN)
                with self.frame_lock: self.display_frame = display.copy()
                time.sleep(1.5); self.stage = RegistrationStage.COMPLETE 
            else: self._reset_registration(display, "❌ GAGAL!", "Database Error")

    def _process_thread(self):
        try:
            bbox_memory = None
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                ret, frame = self.cam.read()
                if not ret: time.sleep(0.01); continue
                raw, display = frame.copy(), frame.copy()
                faces = self.detector.detect(raw)
                if not faces: 
                    self.missed_frames += 1
                    if self.missed_frames >= 15: 
                        bbox_memory, display = None, cv2.flip(display, 1)
                        Helpers.draw_hud(display, self.stage, "Hadapkan wajah", "", "", "NO FACE", None, config.COLOR_RED)
                        self._timer_started = False
                    elif bbox_memory: 
                        display = cv2.flip(display, 1)
                        Helpers.draw_hud(display, self.stage, "Menganalisa...", "", "", "TRACKING", bbox_memory, config.COLOR_YELLOW)
                    with self.frame_lock: self.display_frame = display
                    continue
                self.missed_frames, face = 0, faces[0]
                bbox_memory = face.bbox
                current_light = Helpers.get_light_condition_dynamic(raw, face.bbox)
                if self.stage == RegistrationStage.FACEMESH: self.locked_light_cond = current_light
                light_cond = getattr(self, 'locked_light_cond', "Normal")
                if time.time() - getattr(self, 'app_start_time', time.time()) < 2.5:
                    light_cond = "Normal"; self.locked_light_cond = current_light 
                enhanced = Helpers.enhance_adaptive(raw, face.bbox, light_cond)
                if not self._timer_started: self.action_start_time, self._timer_started = time.time(), True
                display = cv2.flip(self.detector.draw(display, face), 1)
                if face.bbox[3] > int(config.FRAME_HEIGHT * 0.50): 
                    Helpers.draw_hud(display, self.stage, "Wajah Terlalu Dekat!", "Mundur", "", "TOO CLOSE", face.bbox, config.COLOR_YELLOW)
                    with self.frame_lock: self.display_frame = display
                    continue
                pose, ear_val = self.liveness.pose_estimator.estimate(face, self.detector), (Helpers.capture_blink(face) or {}).get("avg_ear", 0.0)
                if self.stage in (RegistrationStage.FACEMESH, RegistrationStage.EXTRACTION):
                    sp = self.anti_spoof.is_real(raw, face.bbox)
                    sp_score, sp_real, sp_label = float(sp.get("score_real", sp.get("score", 1.0))), sp.get("real", True), sp.get("label_name", "FOTO").upper()
                else:
                    sp_score, sp_real, sp_label = 1.0, True, "REAL"
                hud_txt, term_txt = self._generate_metric_text(pose, ear_val, sp_score, light_cond)
                if not sp_real:
                    self.fake_frames += 1
                    if self.fake_frames >= 8: 
                        Helpers.draw_hud(display, self.stage, "❌ DETEKSI SPOOFING!", f"Palsu: {sp_score:.2f}", hud_txt, f"{sp_label}", face.bbox, config.COLOR_RED)
                        with self.frame_lock: self.display_frame = display
                        continue 
                else: self.fake_frames = 0
                if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                    self._record_data_buffers(face, pose, enhanced) 
                    res = self.liveness.update_register(face, self.detector)
                    if self._commit_stage_data(res["step"], display): continue
                    if self.prev_instruction is None or res.get("instruction", "") != self.prev_instruction: 
                        self.prev_instruction, self.action_start_time, self.hold_frames = res.get("instruction", ""), time.time(), 0
                    Helpers.draw_hud(display, self.stage, res.get("instruction", ""), res.get("progress",""), hud_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_GREEN if res["step"] == "DONE" else config.COLOR_CYAN)
                elif not self.in_ext: 
                    self._process_extraction(raw, enhanced, face, display, pose, hud_txt, sp_score, sp_label)
                with self.frame_lock: self.display_frame = display
        finally: self.is_running = False
            
    def run(self):
        if self.stage == RegistrationStage.COMPLETE: return
        threading.Thread(target=self._process_thread, daemon=True).start()
        try:
            if getattr(config, 'USE_FULLSCREEN', False):
                cv2.namedWindow("Register", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Register", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow("Register", cv2.WINDOW_AUTOSIZE)
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                with self.frame_lock: frame = self.display_frame.copy() if self.display_frame is not None else None
                if frame is not None:
                    if not getattr(config, 'USE_FULLSCREEN', False) and getattr(config, 'DISPLAY_SCALE', 1.0) != 1.0:
                        h, w = frame.shape[:2]
                        frame = cv2.resize(frame, (int(w * config.DISPLAY_SCALE), int(h * config.DISPLAY_SCALE)))
                    cv2.imshow("Register", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): self.is_running = False; break
        finally: self.is_running = False; time.sleep(0.5); self.cam.stop(); self.detector.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"\n{'='*45}\n   SISTEM REGISTRASI WAJAH (MULTI-FRAME V2)\n{'='*45}")
    if nama_input := input("Masukan Nama Lengkap : ").strip(): FaceRegistrationApp(nama_input).run()