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

class RegistrationStage(Enum): IDLE=0; FACEMESH=1; POSE=2; BLINK=3; EXTRACTION=4; COMPLETE=5
STAGE_NAMES = {RegistrationStage.FACEMESH: "1. FaceMesh (3D)", RegistrationStage.POSE: "2. Liveness (Pose)", RegistrationStage.BLINK: "3. Liveness (Blink)", RegistrationStage.EXTRACTION: "4. Ekstraksi Fitur"}
STEP_TO_STAGE = {"FACEMESH": RegistrationStage.FACEMESH, "POSE": RegistrationStage.POSE, "BLINK": RegistrationStage.BLINK, "DONE": RegistrationStage.EXTRACTION}

def _log(msg, level="INFO"): 
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {f'[{level}]'.ljust(10)} {msg}")

class Helpers:
    @staticmethod
    def enhance_adaptive(frame, bbox=None, l_str="Normal"):
        if l_str == "Normal" or not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        d, sig_color, sig_space = {"Low Light": (5, 60, 60), "Backlight": (5, 45, 45)}.get(l_str, (5, 40, 40))
        filtered = cv2.bilateralFilter(frame, d, sig_color, sig_space)
        gamma = {"Low Light": 0.6, "Backlight": 0.5}.get(l_str, 1.0)
        if gamma != 1.0:
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            filtered = cv2.LUT(filtered, table)
        yuv = cv2.cvtColor(filtered, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.createCLAHE(clipLimit={"Low Light":2.0, "Backlight":1.8}.get(l_str, 1.5), tileGridSize=(8, 8)).apply(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def simulate_lowlight_raw(crop, level="medium"):
        alpha, beta, sigma = (0.45, 15, 8) if level == "medium" else (0.25, 5, 15)
        dark = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        gauss = np.random.normal(0, sigma, dark.shape).astype(np.int16)
        return np.clip(dark.astype(np.int16) + gauss, 0, 255).astype(np.uint8)

    @staticmethod
    def simulate_backlight_raw(crop, level="medium"):
        alpha, beta, sat = (0.35, 10, 0.75) if level == "medium" else (0.20, 5, 0.55)
        dark = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        hsv = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= sat
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def enhance_crop(crop, l_str="Normal"):
        if l_str == "Normal" or not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return crop
        d, sig_col, sig_sp = {"Low Light": (5, 60, 60), "Backlight": (5, 45, 45)}.get(l_str, (5, 40, 40))
        filtered = cv2.bilateralFilter(crop, d, sig_col, sig_sp)
        gamma = {"Low Light": 0.6, "Backlight": 0.5}.get(l_str, 1.0)
        if gamma != 1.0:
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            filtered = cv2.LUT(filtered, table)
        yuv = cv2.cvtColor(filtered, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit={"Low Light": 2.0, "Backlight": 1.8}.get(l_str, 1.5), tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_aligned_crop(frame, face, target_size=(112, 112)):
        lm, (fh, fw) = getattr(face, 'landmarks', []), frame.shape[:2]
        if not lm or len(lm) < 300: 
            bx, by, bw, bh = face.bbox
            return frame[max(0, by):min(fh, by+bh), max(0, bx):min(fw, bx+bw)]
        le = np.array([(lm[33].x + lm[133].x) * fw / 2, (lm[33].y + lm[133].y) * fh / 2])
        re = np.array([(lm[263].x + lm[362].x) * fw / 2, (lm[263].y + lm[362].y) * fh / 2])
        nose = np.array([lm[4].x * fw, lm[4].y * fh])
        lmouth, rmouth = np.array([lm[61].x * fw, lm[61].y * fh]), np.array([lm[291].x * fw, lm[291].y * fh])
        src_pts = np.array([le, re, nose, lmouth, rmouth], dtype=np.float32)
        dst_pts = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
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
        brightness, blur_score = np.mean(gray), cv2.Laplacian(gray, cv2.CV_64F).var()
        min_b = getattr(config, 'REG_MIN_BRIGHTNESS', 15.0)
        max_b = getattr(config, 'REG_MAX_BRIGHTNESS', 245.0)
        min_blur = getattr(config, 'REG_MIN_BLUR_SCORE', 35.0)
        return (min_b < brightness < max_b) and (blur_score > min_blur), brightness, blur_score 

    @staticmethod
    def detect_glasses(frame, face):
        if not getattr(face, 'landmarks_px', None) is not None or len(face.landmarks_px) < 400: return False
        try:
            lm = face.landmarks_px
            p_left, p_right, p_bridge = lm[133], lm[362], lm[168]
            eye_dist = np.linalg.norm(p_right - p_left)
            if eye_dist < 10: return False
            w_roi, h_roi = int(eye_dist * 0.6), int(eye_dist * 0.3)
            x1, y1 = max(0, int(p_bridge[0] - w_roi // 2)), max(0, int(p_bridge[1] - h_roi // 2))
            x2, y2 = min(frame.shape[1], x1 + w_roi), min(frame.shape[0], y1 + h_roi)
            if (x2 - x1) < 5 or (y2 - y1) < 5: return False
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.bilateralFilter(gray, 5, 50, 50), 20, 80)
            return np.mean(edges > 0) > 0.030
        except: return False

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
        sv = min(stg.value, 4); bw_bar, bh_bar = 160, 14; bx_bar, by_bar = w - bw_bar - 10, 9
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (45, 45, 45), -1)
        if sv > 0: cv2.rectangle(f, (bx_bar, by_bar), (bx_bar + int(bw_bar*(sv-1)/4), by_bar+bh_bar), config.COLOR_GREEN, -1)
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (255,255,255), 1)
        cv2.putText(f, f"Tahap {sv if sv<=3 else 4}/4", (bx_bar + 45, by_bar + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
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
    
    def __init__(self, name):
        self.name, self.user_id = name, str(uuid.uuid4())
        self.stage = RegistrationStage.FACEMESH
        self.in_ext, self.hold_frames, self.missed_frames = False, 0, 0
        self.fake_frames = 0
        self.is_running, self.display_frame, self.frame_lock = True, None, threading.Lock()
        self.locked_light_cond = None 
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None}
        self._pose_buf, self._blink_buf, self._prev_step = {"yaw": {}, "pitch": {}, "roll": {}}, {"closed": None, "open": None, "logged_closed": False, "logged_open": False}, "FACEMESH"
        self.extraction_embeddings = []
        self.extraction_candidates = []
        self._motion_history = []
        self.last_extraction_time = 0
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
        self._pose_emb = None
        self._pose_emb_captured = False
        _log(f"✅ Memulai Registrasi untuk {self.name}...", "SYSTEM")

    def trigger_buzzer_async(self, pin, duration=0.1, count=1, gap=0.05):
        if not GPIO_AVAILABLE or pin is None: return
        def _run():
            try:
                try:
                    GPIO.setup(pin, GPIO.OUT)
                except:
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(pin, GPIO.OUT)
                
                for _ in range(count):
                    GPIO.output(pin, GPIO.HIGH)
                    time.sleep(duration)
                    GPIO.output(pin, GPIO.LOW)
                    if count > 1:
                        time.sleep(gap)
            except Exception as e:
                _log(f"Gagal membunyikan buzzer: {e}", "WARNING")
        threading.Thread(target=_run, daemon=True).start()

    def _reset_registration(self, display, t_title, t_sub):
        buzzer_pin = getattr(config, 'BUZZER_PIN', 23)
        self.trigger_buzzer_async(buzzer_pin, duration=0.4, count=1)
        Helpers.show_msg(display, t_title, f"{t_sub} | Mengulang dari awal...", config.COLOR_RED)
        with self.frame_lock: self.display_frame = display.copy()
        time.sleep(3.0)
        self.stage = RegistrationStage.FACEMESH
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None}
        self._pose_buf = {"yaw": {}, "pitch": {}, "roll": {}}
        self._blink_buf = {"closed": None, "open": None, "logged_closed": False, "logged_open": False}
        self._prev_step = "FACEMESH"
        self.extraction_embeddings = []
        self.extraction_candidates = []
        self._motion_history = []
        self.last_extraction_time = 0
        self.hold_frames = self.missed_frames = self.fake_frames = 0
        self.individual_latencies, self.action_start_time = {}, time.time()
        self.prev_instruction, self.prev_tag, self.locked_light_cond = None, None, None
        self._pose_emb = None
        self._pose_emb_captured = False
        self.liveness.start_register()

    def _record_data_buffers(self, face, pose, enhanced, raw):
        if not self.action_start_time: return
        if self.stage == RegistrationStage.FACEMESH and face.landmarks:
            if self.cap_data["facemesh_vector"] is None:
                self.hold_frames += 1
                if self.hold_frames >= 4: 
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    key_indices = [33, 263, 4, 61, 291]  # Left eye, right eye, nose, mouth corners
                    self.cap_data["facemesh_vector"] = np.array([[face.landmarks[i].x, face.landmarks[i].y, face.landmarks[i].z] for i in key_indices], dtype=np.float32).flatten()
                    self.hold_frames = 0; self.individual_latencies["FaceMesh (3D)"] = lat_ms
                    _log(f"⏱️ Selesai Tahap FaceMesh (3D) | Latensi: {lat_ms:.2f} ms", "METRIK")
                    self.action_start_time = time.time() 

        if self.stage == RegistrationStage.POSE:
            axis = self.liveness.chosen_params["axis"]
            tag = self.liveness.chosen_params["tag"]
            friendly = self.liveness.chosen_params["friendly"]
            buf = self._pose_buf[axis]
            
            # Ambil embedding variasi sudut ringan (10 - 15 derajat) secara dinamis
            baseline_val = getattr(self.liveness, '_baseline_pose', {}).get(axis, 0.0) if getattr(self.liveness, '_baseline_pose', None) else 0.0
            val = abs(pose.get(axis, 0.0) - baseline_val)
            target_min = 8.0 if axis == "pitch" else 10.0
            target_max = 13.0 if axis == "pitch" else 15.0
            
            if target_min <= val <= target_max and not getattr(self, '_pose_emb_captured', False):
                quality_ok, _, _ = Helpers.is_image_quality_good(raw, face.bbox)
                if quality_ok:
                    if (raw_crop := Helpers.get_aligned_crop(raw, face, target_size=(112, 112))) is not None and raw_crop.size > 0:
                        enhanced_c = Helpers.enhance_crop(raw_crop, "Normal")
                        emb = self.model.get_embedding(enhanced_c)
                        if emb is not None:
                            self._pose_emb = np.array(emb).flatten().tolist()
                            self._pose_emb_captured = True
                            _log(f"📸 Berhasil merekam 1 variasi sudut master ({friendly})", "SYSTEM")

            # Record pose snapshot when liveness manager has verified the extreme pose hold
            if self.liveness._pose_state == "WAITING_CENTER":
                if tag not in buf:
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    buf[tag] = {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")}
                    buf[tag]["tag"], buf[tag]["latency_ms"] = tag, lat_ms 
                    self.individual_latencies[friendly] = lat_ms
                    _log(f"⏱️ Selesai Tahap Pose ({friendly}) | Latensi: {lat_ms:.2f} ms", "METRIK")
                    self.action_start_time = time.time()

        if self.stage == RegistrationStage.BLINK and (bv := Helpers.capture_blink(face)):
            # Log closed when liveness manager detects closed eye state
            if self.liveness._blink_state == 2 and not self._blink_buf.get("logged_closed"):
                lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                bv["latency_ms"] = lat_ms
                self._blink_buf["closed"], self._blink_buf["logged_closed"] = bv, True
                self.individual_latencies["Mata Menutup"] = lat_ms
                _log(f"⏱️ Selesai Tahap Blink (Mata Menutup) | Latensi: {lat_ms:.2f} ms", "METRIK")
                self.action_start_time = time.time() 
            # Log open when liveness manager detects open eye state again after closed
            elif self.liveness._blink_state != 2 and self._blink_buf.get("logged_closed") and not self._blink_buf.get("logged_open"):
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
        if stg == RegistrationStage.FACEMESH:
            hud_txt = f"Menganalisa 3D | Cahaya: {light_cond}"
        elif stg == RegistrationStage.POSE:
            friendly = self.liveness.chosen_params["friendly"]
            hud_txt = f"Aksi: {friendly} | Cahaya: {light_cond}"
        elif stg == RegistrationStage.BLINK:
            hud_txt = f"Aksi: Kedipkan Mata | Cahaya: {light_cond}"
        else:
            hud_txt = f"Tahan Lurus | {light_cond}"
        return hud_txt, f"{hud_txt} | Spf: {sp_score:.2f}"

    def _commit_stage_data(self, cur_step, display):
        if cur_step in ("WAIT", self._prev_step): return False
        if self._prev_step == "POSE":
            axis = self.liveness.chosen_params["axis"]
            snap_key = self.liveness.chosen_params["snap_key"]
            if len(self._pose_buf[axis]) < 1:
                friendly = self.liveness.chosen_params["friendly"]
                self._reset_registration(display, "❌ GAGAL POSE!", f"Deteksi {friendly} Kurang/Terlalu Cepat")
                return True
            self.cap_data[snap_key], self._pose_buf[axis] = list(self._pose_buf[axis].values()), {}  
            
        if cur_step == "DONE" and self._prev_step == "BLINK":
            bc, bo = self._blink_buf["closed"], self._blink_buf["open"]
            min_open, max_closed, min_delta = getattr(config, 'MIN_BLINK_OPEN_EAR', 0.22), getattr(config, 'MAX_BLINK_CLOSED_EAR', 0.20), getattr(config, 'MIN_BLINK_DELTA', 0.04)
            if not bc or not bo or (bo["avg_ear"] < min_open) or (bc["avg_ear"] > max_closed) or (bo["avg_ear"] - bc["avg_ear"] < min_delta): 
                self.liveness._register_step, self.liveness._blink_state, self.liveness._hold_frames, self.liveness._blink_count, self._blink_buf = 2, 0, 0, 0, {"closed": None, "open": None, "logged_closed": False, "logged_open": False}; return False
            self.cap_data.update({"blink_closed": bc, "blink_open": bo})
        self._prev_step, self.stage = cur_step, STEP_TO_STAGE.get(cur_step, self.stage)
        buzzer_pin = getattr(config, 'BUZZER_PIN', 23)
        self.trigger_buzzer_async(buzzer_pin, duration=0.1, count=1)
        if self.stage != RegistrationStage.COMPLETE: self.action_start_time, self.hold_frames = time.time(), 0

    def _process_extraction(self, raw_frame, frame, face, display, pose, score_txt, sp_score, sp_label):
        chosen = self.liveness.chosen_challenge
        pose_ok = False
        if chosen in ("yaw_left", "yaw_right"):
            pose_ok = len(self.cap_data["yaw_snapshots"]) >= 1
        elif chosen in ("pitch_up", "pitch_down"):
            pose_ok = len(self.cap_data["pitch_snapshots"]) >= 1
        elif chosen in ("roll_left", "roll_right"):
            pose_ok = len(self.cap_data["roll_snapshots"]) >= 1

        missing = [k for k, v in [
            ("FaceMesh", self.cap_data["facemesh_vector"] is not None), 
            ("Pose", pose_ok), 
            ("Blink", self.cap_data["blink_closed"] is not None)
        ] if not v]
        if missing: 
            self._reset_registration(display, "❌ GAGAL EKSTRAKSI!", f"Data Kurang: {','.join(missing)}")
            return

        # 1. Pengecekan stabilitas gerakan wajah (Motion Stability)
        curr_center = (face.bbox[0] + face.bbox[2]/2.0, face.bbox[1] + face.bbox[3]/2.0)
        self._motion_history.append(curr_center)
        if len(self._motion_history) > 4:
            self._motion_history.pop(0)

        if len(self._motion_history) >= 3:
            diffs = [np.linalg.norm(np.array(self._motion_history[i]) - np.array(self._motion_history[i-1])) for i in range(1, len(self._motion_history))]
            avg_movement = np.mean(diffs)
            if avg_movement > 7.0:
                Helpers.draw_hud(display, self.stage, "⚠️ Harap Diam, Jangan Bergerak!", "Menstabilkan posisi wajah...", score_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_YELLOW)
                with self.frame_lock: self.display_frame = display.copy()
                return

        # 2. Pengecekan kemiringan wajah (Pose Frontal) secara ketat
        yaw, pitch, roll = pose.get("yaw", 0.0), pose.get("pitch", 0.0), pose.get("roll", 0.0)
        is_straight = abs(yaw) < 7.0 and abs(pitch) < 7.0 and abs(roll) < 7.0
        if not is_straight:
            Helpers.draw_hud(display, self.stage, "⚠️ Posisikan Wajah Lurus ke Depan!", "Harap tatap kamera secara lurus...", score_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_YELLOW)
            with self.frame_lock: self.display_frame = display.copy()
            return

        # 3. Kriteria kecerahan dan ketajaman dasar
        quality_ok, brightness, blur_score = Helpers.is_image_quality_good(raw_frame, face.bbox)
        if not quality_ok:
            reason = "Cahaya Buruk" if not (config.REG_MIN_BRIGHTNESS < brightness < config.REG_MAX_BRIGHTNESS) else "Kamera Blur"
            Helpers.draw_hud(display, self.stage, f"⚠️ {reason}!", "Mohon tetap diam...", score_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_YELLOW)
            with self.frame_lock: self.display_frame = display.copy()
            return

        # 4. Pengumpulan Kandidat (Best-of-10)
        current_time = time.time()
        if current_time - self.last_extraction_time < 0.08:
            collected = len(self.extraction_candidates)
            Helpers.draw_hud(display, self.stage, f"🧠 EKSTRAKSI FITUR WAJAH ({int((collected / 10) * 100)}%)", f"Mengambil data {collected}/10. Tahan posisi...", score_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_CYAN)
            with self.frame_lock: self.display_frame = display.copy()
            return

        if (raw_crop := Helpers.get_aligned_crop(raw_frame, face, target_size=(112, 112))) is not None and raw_crop.size > 0:
            self.last_extraction_time = current_time
            # Hitung skor kualitas komposit
            # Frontalness score (kemiringan optimal = 0)
            frontal_dev = abs(yaw) + abs(pitch) + abs(roll)
            frontal_score = 1.0 - (frontal_dev / 21.0)
            
            # Sharpness/Blur normalization (semakin tajam semakin baik, target 150)
            blur_norm = min(1.0, blur_score / 150.0)
            
            # Brightness balance score (ideal = 120)
            brightness_score = 1.0 - (abs(brightness - 120.0) / 120.0)
            
            # Liveness score
            liveness_score = sp_score
            
            # Gabungkan skor dengan bobot
            composite_score = (blur_norm * 0.45) + (frontal_score * 0.35) + (brightness_score * 0.10) + (liveness_score * 0.10)
            
            self.extraction_candidates.append({
                "raw_crop": raw_crop.copy(),
                "score": composite_score,
                "blur": blur_score,
                "frontal": frontal_score,
                "brightness": brightness
            })

        total_candidates_needed = 10
        collected = len(self.extraction_candidates)
        if collected < total_candidates_needed:
            Helpers.draw_hud(display, self.stage, f"🧠 EKSTRAKSI FITUR WAJAH ({int((collected / total_candidates_needed) * 100)}%)", f"Mengambil data {collected}/{total_candidates_needed}. Tahan posisi...", score_txt, f"Real: {sp_score:.2f}", face.bbox, config.COLOR_CYAN)
            with self.frame_lock: self.display_frame = display.copy()
            return

        # 5. Seleksi 5 Kandidat Terbaik (Best of 10)
        self.extraction_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = self.extraction_candidates[:5]
        
        _log("📊 Memilih 5 frame terbaik untuk ekstraksi:", "SYSTEM")
        for idx, cand in enumerate(top_candidates):
            _log(f"   Frame #{idx+1}: Skor={cand['score']:.3f} | Blur={cand['blur']:.1f} | Brightness={cand['brightness']:.1f}", "SYSTEM")
            
        # Ekstraksi embedding multi-kondisi dari 5 kandidat terbaik
        self.extraction_embeddings = []
        for cand in top_candidates:
            c_crop = cand["raw_crop"]
            embs = {}
            for key, cond, sim_args in [
                ("normal", "Normal", None),
                ("low_med", "Low Light", ("low", "medium")),
                ("low_ext", "Low Light", ("low", "extreme")),
                ("back_med", "Backlight", ("back", "medium")),
                ("back_str", "Backlight", ("back", "strong"))
            ]:
                if not sim_args: c = c_crop
                elif sim_args[0] == "low": c = Helpers.simulate_lowlight_raw(c_crop, sim_args[1])
                else: c = Helpers.simulate_backlight_raw(c_crop, sim_args[1])
                
                enhanced_c = Helpers.enhance_crop(c, cond)
                emb = self.model.get_embedding(enhanced_c)
                if emb is not None: 
                    embs[key] = np.array(emb).flatten()
            if len(embs) == 5: 
                self.extraction_embeddings.append(embs)

        if len(self.extraction_embeddings) < 3:
            self._reset_registration(display, "❌ GAGAL EKSTRAKSI!", "Kualitas model ekstraksi buruk")
            return

        avg_embs = {k: np.mean([e[k] for e in self.extraction_embeddings], axis=0) for k in ["normal", "low_med", "low_ext", "back_med", "back_str"]}
        final_emb_vectors = [(v / (np.linalg.norm(v) + 1e-6)).tolist() for v in avg_embs.values()]
        if getattr(self, '_pose_emb', None) is not None:
            final_emb_vectors.append(self._pose_emb)
        light_cond = getattr(self, 'locked_light_cond', "Normal")
        latensi_respon_subjek = round(sum(self.individual_latencies.values()), 2)
        mfn_latency = round((time.time() - self.action_start_time) * 1000, 2)
        total_waktu_sistem = latensi_respon_subjek + mfn_latency
        
        is_duplicate, duplicate_name = False, ""
        anti_dup_thr = getattr(config, 'ANTI_DUPLICATE_THRESHOLD', getattr(config, 'MATCH_THRESHOLD', 0.65))
        best_sim_score = 0.0
        all_faces_raw = self.db.load_all_faces()
        existing_user_id = self.user_id

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
                            for q_emb in [final_emb_vectors[0]]: 
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
            buzzer_pin = getattr(config, 'BUZZER_PIN', 23)
            self.trigger_buzzer_async(buzzer_pin, duration=0.4, count=1)
            Helpers.show_msg(display, "❌ WAJAH SUDAH TERDAFTAR!", f"Mirip {best_sim_score * 100:.1f}% dgn {duplicate_name}", config.COLOR_RED)
            with self.frame_lock: self.display_frame = display.copy()
            time.sleep(4.0); self.stage = RegistrationStage.COMPLETE; return
        else:
            self.cap_data.update({"reg_latency_ms": total_waktu_sistem, "individual_latencies": self.individual_latencies, "headpose_vector": [float(pose["yaw"]), float(pose["pitch"]), float(pose["roll"])], "registration_accuracy": 100.0, "light_condition": light_cond})
            if "face_crops" in self.cap_data: del self.cap_data["face_crops"]
            if self.db.save_face(self.name, existing_user_id, final_emb_vectors, self.cap_data): 
                _log(f"⏱️ Ekstraksi AI & Simpan DB Selesai | Latensi AI: {mfn_latency:.2f} ms", "METRIK")
                _log(f"✅ Total Waktu Seluruh Registrasi | Latensi Sistem Total: {total_waktu_sistem:.2f} ms", "METRIK")
                Helpers.show_msg(display, "✅ REGISTRASI BERHASIL!", f"User: {self.name} | 3 Vektor Cahaya", config.COLOR_GREEN)
                buzzer_pin = getattr(config, 'BUZZER_PIN', 23)
                self.trigger_buzzer_async(buzzer_pin, duration=0.15, count=2, gap=0.08)
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
                    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                    if np.mean(gray) < 130:
                        gamma = 0.5
                        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                        faces = self.detector.detect(cv2.LUT(raw, table))
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
                
                # Fase kalibrasi sensor cahaya kamera saat aplikasi pertama menyala
                if time.time() - getattr(self, 'app_start_time', time.time()) < 2.2:
                    display = cv2.flip(self.detector.draw(display, face), 1)
                    Helpers.draw_hud(display, self.stage, "Mengkalibrasi Sensor...", "Tahan Posisi Wajah Anda", "", "CALIBRATING", face.bbox, config.COLOR_YELLOW)
                    with self.frame_lock: self.display_frame = display
                    continue

                current_light = Helpers.get_light_condition_dynamic(raw, face.bbox)
                if self.stage == RegistrationStage.FACEMESH and getattr(self, 'locked_light_cond', None) is None:
                    self.locked_light_cond = current_light
                light_cond = getattr(self, 'locked_light_cond', "Normal")
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
                    sp_score = float(sp.get("score_real", sp.get("score", 1.0)))
                    as_thr = 0.72 if Helpers.detect_glasses(raw, face) else 0.80
                    sp_real = sp_score >= as_thr
                    sp_label = sp.get("label_name", "FOTO").upper() if not sp_real else "REAL"
                else: sp_score, sp_real, sp_label = 1.0, True, "REAL"
                hud_txt, term_txt = self._generate_metric_text(pose, ear_val, sp_score, light_cond)
                if not sp_real:
                    self.fake_frames += 1
                    if self.fake_frames >= 8: 
                        Helpers.draw_hud(display, self.stage, "❌ DETEKSI SPOOFING!", f"Palsu: {sp_score:.2f}", hud_txt, f"{sp_label}", face.bbox, config.COLOR_RED)
                        with self.frame_lock: self.display_frame = display
                        continue 
                else: self.fake_frames = 0
                if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                    self._record_data_buffers(face, pose, enhanced, raw) 
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
            cv2.namedWindow("Register", cv2.WND_PROP_FULLSCREEN if getattr(config, 'USE_FULLSCREEN', False) else cv2.WINDOW_AUTOSIZE)
            if getattr(config, 'USE_FULLSCREEN', False): cv2.setWindowProperty("Register", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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