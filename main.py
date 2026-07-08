import cv2, random, time, threading, sys, numpy as np
from datetime import datetime
from enum import Enum
import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet  import MobileFaceNet
from door.door_lock             import DoorLock
from liveness.head_pose         import HeadPoseEstimator
from liveness.anti_spoofing     import SilentAntiSpoofing  
from database.face_db           import FaceDatabase

try: import RPi.GPIO as GPIO; GPIO_AVAILABLE = True
except ImportError: GPIO_AVAILABLE = False

class ValidationState(Enum): IDLE=0; RECOGNIZING=1; CHALLENGE=2; UNMATCHED=3; UNLOCKED=4

class UIHelper:
    @staticmethod
    def log(msg, lvl="INFO"): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    @staticmethod
    def print_inline(msg): sys.stdout.write(f"\r ⏳ {msg}\033[K"); sys.stdout.flush()

    @staticmethod
    def enhance_adaptive(frame, bbox, l_str="Normal"):
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
    def get_ear(f):
        lm = getattr(f, 'landmarks', [])
        if not lm or len(lm) < 400: return 0.0
        p = np.array([[lm[i].x, lm[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]]); n = np.linalg.norm 
        return ((n(p[1]-p[5])+n(p[2]-p[4]))/(2.0*n(p[0]-p[3])+1e-6) + (n(p[7]-p[11])+n(p[8]-p[10]))/(2.0*n(p[6]-p[9])+1e-6)) / 2.0

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
    def analyze_spoof_type(raw_frame, bbox):
        fh, fw = raw_frame.shape[:2]; bx, by, bw, bh = bbox
        pad_w, pad_h = int(bw * 0.15), int(bh * 0.15)
        x1, y1 = max(0, bx - pad_w), max(0, by - pad_h)
        x2, y2 = min(fw, bx + bw + pad_w), min(fh, by + bh + pad_h)
        crop = raw_frame[y1:y2, x1:x2]
        if crop.size == 0: return "MEDIA PALSU"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return "LAYAR VIDEO" if np.mean(hsv[:, :, 2]) > 160 and np.mean(hsv[:, :, 1]) < 90 else "FOTO CETAK"

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
    def draw_ui(d, ui, locked):
        fw, fh = d.shape[1], d.shape[0]; c_font = cv2.FONT_HERSHEY_SIMPLEX
        if ui.get("status") == "STARTING":
            cv2.rectangle(d, (0, 0), (fw, fh), (20, 20, 20), -1)
            return cv2.putText(d, "SISTEM SEDANG BERSIAP...", (fw//2 - 120, fh//2 - 10), c_font, 0.5, config.COLOR_YELLOW, 2, cv2.LINE_AA)
        if l_cond := ui.get("light_cond"): cv2.putText(d, f"Cahaya: {l_cond}", (15, 55), c_font, 0.45, config.COLOR_GREEN if l_cond == "Normal" else (config.COLOR_YELLOW if l_cond == "Low Light" else config.COLOR_RED), 1, cv2.LINE_AA)
        if ui.get("wait"): cv2.putText(d, "Mencari Wajah...", (15, 35), c_font, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA)
        elif ui.get("bbox"):
            x, y, w, h = ui["bbox"]; fx, c = fw - x - w, ui.get("color", config.COLOR_WHITE)
            cv2.rectangle(d, (fx, y), (fx+w, y+h), c, 2)
            if stat := ui.get("status", ""):
                tw, th = cv2.getTextSize(stat, c_font, 0.45, 1)[0]
                cv2.rectangle(d, (fx, max(y, th+10) - th - 8), (fx + tw + 10, max(y, th+10)), c, -1)
                cv2.putText(d, stat, (fx + 5, max(y, th+10) - 4), c_font, 0.45, (0,0,0) if c in (config.COLOR_WHITE, config.COLOR_YELLOW) else (255,255,255), 1, cv2.LINE_AA)
        if instr := ui.get("instr"):
            tw, th = cv2.getTextSize(instr, c_font, 0.45, 1)[0]
            cv2.rectangle(d, (5, 5), (tw + 25, th + 15), (25, 25, 25), -1)
            cv2.putText(d, instr, (15, th + 10), c_font, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA)
        cv2.rectangle(d, (0, fh - 28), (fw, fh), (20, 20, 20), -1)
        cv2.putText(d, f"STATUS PINTU: {'TERKUNCI' if locked else 'TERBUKA'}", (10, fh - 8), c_font, 0.45, config.COLOR_RED if locked else config.COLOR_GREEN, 1, cv2.LINE_AA)

class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH", "MIRING_KANAN": "Miring KANAN", "MIRING_KIRI": "Miring KIRI"}

    def __init__(self):
        print(f"\n{'='*50}\n[SYSTEM] SISTEM DOOR LOCK MULTI-VECTOR MATRIKS 2D\n{'='*50}\n")
        self._is_ready = False; self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui = {"wait": True, "bbox": None, "status": "STARTING", "color": config.COLOR_WHITE, "instr": "", "light_cond": None}
        self.missed_frames = self.fake_frames = self.last_spoof_log_time = self.pause_until = 0.0
        if GPIO_AVAILABLE: GPIO.setwarnings(False)
        self._reset_state(); self._init_heavy_models()

    def _init_heavy_models(self):
        self.db, self.model, self.pose_estimator = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator()
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.80))
        self.detector = FaceMeshDetector(min_detection_confidence=0.35, min_tracking_confidence=0.35)
        self.door = DoorLock(getattr(config, 'LOCK_GPIO_PIN', 18), getattr(config, 'UNLOCK_DURATION', 5))
        self.known_faces_2d = {}
        emb_dim = getattr(self.model, 'embedding_size', 128)
        for k, v in (self.db.load_all_faces() or {}).items():
            if isinstance(v, dict) and (emb_val := v.get('embedding')):
                sub_embs = emb_val if isinstance(emb_val, list) and len(emb_val) > 0 and isinstance(emb_val[0], list) else ([emb_val[0:emb_dim], emb_val[emb_dim:emb_dim*2], emb_val[emb_dim*2:emb_dim*3]] if isinstance(emb_val, list) and len(emb_val) == (emb_dim * 3) else [emb_val])
                valid_embs = [np.array(e, dtype=np.float32) for e in sub_embs if len(e) == emb_dim]
                if valid_embs: self.known_faces_2d[k] = valid_embs
        if GPIO_AVAILABLE:
            btn = getattr(config, 'BUTTON_PIN', 26)
            try:
                GPIO.setmode(GPIO.BCM); GPIO.setup(btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                GPIO.remove_event_detect(btn); GPIO.add_event_detect(btn, GPIO.FALLING, callback=self._manual_unlock, bouncetime=1000)
            except: threading.Thread(target=self._button_polling_worker, args=(btn,), daemon=True).start()
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start(); time.sleep(2.0)
        self.ui.update({"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": "", "light_cond": None})
        threading.Thread(target=self._ai_worker, daemon=True).start(); time.sleep(1.0); self._is_ready = True 

    def _button_polling_worker(self, pin):
        last_state = GPIO.HIGH
        while self.running:
            try:
                curr_state = GPIO.input(pin)
                if last_state == GPIO.HIGH and curr_state == GPIO.LOW: self._manual_unlock(pin); time.sleep(1.5)
                last_state = curr_state
            except: pass
            time.sleep(0.05) 

    def _manual_unlock(self, channel):
        if not getattr(self, '_is_ready', False): return 
        
        # 1. Abaikan jika pintu sudah terbuka (mencegah trigger akibat noise listrik saat relay aktif)
        if not self.door.locked:
            return

        # 2. Software debounce: memastikan tombol benar-benar ditekan (LOW) selama 50ms untuk memfilter noise/spikes
        if GPIO_AVAILABLE:
            if GPIO.input(channel) != GPIO.LOW:
                return
            for _ in range(5):
                time.sleep(0.01)
                if GPIO.input(channel) != GPIO.LOW:
                    return

        print(f"\n{'='*60}"); UIHelper.log("🔓 PINTU DIBUKA MANUAL VIA TOMBOL", "SUCCESS"); print(f"{'='*60}\n")
        self._reset_state()
        self.state = ValidationState.UNLOCKED  # Set state ke UNLOCKED agar _ai_worker mereset UI secara otomatis saat pintu mengunci kembali
        self.ui.update({"wait": False, "bbox": None, "status": "DIBUKA MANUAL", "color": config.COLOR_GREEN, "instr": "Tombol Ditekan", "light_cond": None})
        threading.Thread(target=self.door.unlock, daemon=True).start()

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.challenge_start_time, self.face_val_latency, self.final_display_acc = 0.0, 0.0, 0.0
        self.wait_center, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter, self.access_details = False, False, 0, [], 0, []
        self.fake_frames = self.recog_frames = self.spoof_start_time = 0; self.locked_light_cond = "Normal"; self.spoof_hist = []
        self.last_failed_cosine, self.last_failed_threshold, self.last_failed_l_str = 0.0, 0.0, "Normal"
        self.recog_embeddings, self.ear_history_open, self.wearing_glasses = [], [], False

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state(); self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr})
        self.pause_until = time.time() + 2.0  

    def _check_action(self, action, face):
        if action == "BLINK": 
            self.ear_hist = (self.ear_hist + [UIHelper.get_ear(face)])[-3:]
            base_open = np.mean(self.ear_history_open) if getattr(self, 'ear_history_open', None) else 0.30
            adaptive_thr = max(0.20, min(0.24, base_open * 0.72))
            if min(self.ear_hist) <= adaptive_thr: self.blink_hold += 1
            else: self.blink_passed, self.blink_hold = self.blink_hold >= 1, 0
            return self.blink_passed, 1.0, 1.0, False  
        est = self.pose_estimator.estimate(face, self.detector)
        dy, dp, dr = est.get("yaw", 0) - self.reg_pose[0], est.get("pitch", 0) - self.reg_pose[1], est.get("roll", 0) - self.reg_pose[2]
        ty, tp, tr = getattr(config, 'CHALLENGE_YAW', 15.0), getattr(config, 'CHALLENGE_PITCH', 12.0), getattr(config, 'CHALLENGE_ROLL', 12.0)
        val, thr, sign = {"KANAN": (dy, ty, 1), "KIRI": (dy, ty, -1), "ATAS": (dp, tp, -1), "BAWAH": (dp, tp, 1), "MIRING_KANAN": (dr, tr, -1), "MIRING_KIRI": (dr, tr, 1)}.get(action, (0, 1, 1))
        passed = (val > thr) if sign == 1 else (val < -thr)
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 3, abs(float(val)), thr, (val < -thr) if sign == 1 else (val > thr)

    def _check_identity(self, q_emb, l_str):
        d_thr = {"Normal": 0.70, "Backlight": 0.65, "Low Light": 0.67}.get(l_str, 0.70)
        if getattr(self, 'wearing_glasses', False): d_thr -= 0.04
        if not self.known_faces_2d or q_emb is None: return "TIDAK DIKENAL", 0.0, d_thr, 0.0, False
        b_name, b_score = max(((n, np.max([np.dot(q_emb, e) for e in el])) for n, el in self.known_faces_2d.items() if el), key=lambda x: x[1], default=("", 0.0))
        if b_score < d_thr:
            self.recog_frames = max(0, getattr(self, 'recog_frames', 0) - 1)
            return "TIDAK DIKENAL", b_score, d_thr, 0.0, False
        self.recog_frames += 1
        return b_name, b_score, d_thr, float(b_score * 100.0), (self.recog_frames >= getattr(config, 'REQUIRED_STABLE_FRAMES', 6))

    def _async_process_face(self, raw, enhanced, face, l_str):
        try: self._process_face(raw, enhanced, face, l_str)
        finally: self.is_processing = False

    def _ai_worker(self):
        self.is_processing = False
        last_reload_time = 0
        while self.running:
            # Pemuatan ulang data wajah berkala dari SQLite untuk sinkronisasi real-time
            current_time = time.time()
            if current_time - last_reload_time > 5.0:
                last_reload_time = current_time
                self._reload_known_faces()

            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None or time.time() < getattr(self, 'pause_until', 0): time.sleep(0.02); continue
            faces = self.detector.detect(frame)
            if not faces and self.state != ValidationState.UNLOCKED:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) < 130:
                    gamma = 0.5
                    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    faces = self.detector.detect(cv2.LUT(frame, table))
            if self.state == ValidationState.UNLOCKED:
                if getattr(self.door, 'locked', True): self._reset_state(); self.ui.update({"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": "", "light_cond": None})
                else: self.ui.update({"wait": False, "bbox": max(faces, key=lambda f: f.bbox[2]*f.bbox[3]).bbox if faces else None})
                time.sleep(0.02); continue
            if not faces:
                self.missed_frames += 1 
                if self.missed_frames >= 30: self._fail("", wait=True)
            else: 
                self.missed_frames = 0
                face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
                    self.locked_light_cond = UIHelper.get_light_condition_dynamic(frame, face.bbox)
                    self.wearing_glasses = UIHelper.detect_glasses(frame, face)
                l_str = self.locked_light_cond
                self.ui.update({"wait": False, "bbox": face.bbox, "light_cond": l_str}) 
                if not getattr(self, 'is_processing', False):
                    self.is_processing = True
                    threading.Thread(target=self._async_process_face, args=(frame.copy(), UIHelper.enhance_adaptive(frame.copy(), face.bbox, l_str), face, l_str), daemon=True).start()
            time.sleep(0.01)

    def _reload_known_faces(self):
        try:
            faces_raw = self.db.load_all_faces()
            new_known_faces = {}
            emb_dim = getattr(self.model, 'embedding_size', 128)
            for k, v in (faces_raw or {}).items():
                if isinstance(v, dict) and (emb_val := v.get('embedding')):
                    sub_embs = emb_val if isinstance(emb_val, list) and len(emb_val) > 0 and isinstance(emb_val[0], list) else ([emb_val[0:emb_dim], emb_val[emb_dim:emb_dim*2], emb_val[emb_dim*2:emb_dim*3]] if isinstance(emb_val, list) and len(emb_val) == (emb_dim * 3) else [emb_val])
                    valid_embs = [np.array(e, dtype=np.float32) for e in sub_embs if len(e) == emb_dim]
                    if valid_embs: 
                        new_known_faces[k] = valid_embs
            self.known_faces_2d = new_known_faces
        except:
            pass

    def _process_face(self, raw, enhanced, face, l_str):
        if self.ui.get("status") in ("DIBUKA MANUAL", "STARTING"): return
        cx, cy = face.bbox[0] + face.bbox[2]//2, face.bbox[1] + face.bbox[3]//2
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(face.bbox[2:])*2.5: return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 
        if face.bbox[3] > int(config.FRAME_HEIGHT * 0.70): return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur")
        if self.state == ValidationState.IDLE: self.state, self.auth_start = ValidationState.RECOGNIZING, time.time(); return
        liveness_info = self.anti_spoof.is_real(raw.copy(), face.bbox)
        m_conf = float(liveness_info.get("score_real", liveness_info.get("score", 0.0)))
        is_m_real, as_thr = liveness_info.get("real", True), {"Normal": 0.82, "Backlight": 0.80, "Low Light": 0.80}.get(l_str, 0.80)
        if getattr(self, 'wearing_glasses', False): as_thr -= 0.08
        is_actually_real = True if self.state == ValidationState.CHALLENGE else (m_conf >= as_thr)
        if self.state == ValidationState.CHALLENGE: self.fake_frames = 0
        if not is_actually_real:
            if self.fake_frames == 0: self.spoof_start_time, self.spoof_hist = time.time(), []
            self.fake_frames += 1; self.spoof_hist.append(m_conf)
            avg_score, max_spoof_thr = sum(self.spoof_hist) / len(self.spoof_hist), getattr(config, 'MAX_SPOOF_FRAMES', 8)
            if self.fake_frames >= max_spoof_thr: 
                lat_ms = (time.time() - self.spoof_start_time) * 1000
                if is_m_real: sp_type = "TIDAK YAKIN (SKOR RENDAH)"
                else:
                    m_lbl = liveness_info.get("label_name", "").upper()
                    sp_type = "FOTO CETAK" if ("FOTO" in m_lbl or "PRINT" in m_lbl) else ("LAYAR VIDEO" if ("LAYAR" in m_lbl or "VIDEO" in m_lbl or "SCREEN" in m_lbl) else UIHelper.analyze_spoof_type(raw, face.bbox))
                if time.time() - self.last_spoof_log_time > 4.0:
                    self.last_spoof_log_time = time.time()
                    print(f"\n{'='*60}\n⚠️ SECURITY BLOCK: {'Ditolak Skor Rendah' if is_m_real else 'Serangan '+sp_type}\n   AI Confidence: {avg_score:.4f} | Latensi: {lat_ms:.0f} ms\n{'='*60}")
                    if hasattr(self.db, 'log_spoofing_async'): self.db.log_spoofing_async(round(avg_score, 4), 0.0, 0.0, sp_type, round(lat_ms, 2))
                return self._fail(f"SPOOF: {sp_type}", config.COLOR_RED, "Akses Ditolak", wait=False)
            self.ui.update({"wait": False, "bbox": face.bbox, "status": "MEMINDAI LIVENESS...", "color": config.COLOR_YELLOW, "instr": f"Tahan Posisi ({self.fake_frames}/{max_spoof_thr})..."}); return
        else: self.fake_frames, self.spoof_hist = 0, []

        if self.state == ValidationState.RECOGNIZING:
            t_val = time.time()
            gray_face = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            x, y, w, h = face.bbox
            face_roi = gray_face[max(0, y):min(gray_face.shape[0], y+h), max(0, x):min(gray_face.shape[1], x+w)]
            if face_roi.size > 0 and cv2.Laplacian(face_roi, cv2.CV_64F).var() < 35:
                self.ui.update({"status": "MEMINDAI...", "color": config.COLOR_YELLOW, "instr": "Harap tenang, memfokuskan..."}); return

            avg_emb = None
            if (cropped_raw := UIHelper.get_aligned_crop(raw, face, target_size=(112, 112))) is not None and cropped_raw.size > 0:
                cropped = UIHelper.enhance_crop(cropped_raw, l_str)
                raw_emb = self.model.get_embedding(cropped)
                if raw_emb is not None:
                    q_emb = np.array(raw_emb, dtype=np.float32).flatten()
                    q_emb /= (np.linalg.norm(q_emb) + 1e-6)
                    self.recog_embeddings.append(q_emb)
                    self.recog_embeddings = self.recog_embeddings[-3:]
                    avg_emb = np.mean(self.recog_embeddings, axis=0)
                    avg_emb /= (np.linalg.norm(avg_emb) + 1e-6)
                    current_ear = UIHelper.get_ear(face)
                    if current_ear > 0.15:
                        self.ear_history_open.append(current_ear)
                        self.ear_history_open = self.ear_history_open[-15:]

            b_name, sm_score, d_thr, f_acc, is_recog = self._check_identity(avg_emb, l_str)
            disp_name = b_name.split(" - ", 1)[-1] if (b_name != "TIDAK DIKENAL") else "TIDAK DIKENAL"
            if b_name == "TIDAK DIKENAL":
                self.last_failed_cosine, self.last_failed_threshold, self.last_failed_l_str = sm_score, d_thr, l_str
                UIHelper.print_inline(f"Ditolak... Cosine: {sm_score:.3f} < Target Thr:{d_thr:.2f} ({l_str})")
                self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": f"Akses Ditolak (Wajah Belum Terdaftar)"})
                if time.time() - self.auth_start > 8.0:
                    sys.stdout.write("\n"); sys.stdout.flush()
                    return self._fail("TIDAK DIKENAL", config.COLOR_RED, "Mulai Ulang", wait=True)
                return
            if not is_recog: 
                req_f = getattr(config, 'REQUIRED_STABLE_FRAMES', 6)
                self.ui.update({"status": "VERIFIKASI...", "color": config.COLOR_YELLOW, "instr": f"Tahan Posisi Berdiri ({self.recog_frames}/{req_f})..."}); return
            self.print_counter += 1
            if self.print_counter % 2 == 0: UIHelper.print_inline(f"Memproses {disp_name}... Cosine: {sm_score:.3f}")
            self.face_val_latency = (time.time() - t_val) * 1000 
            UIHelper.log(f"\nTerverifikasi: {disp_name} | Cosine: {sm_score:.3f} (Target Thr: {d_thr:.2f}) | Cahaya: {l_str}", "SUCCESS")
            self.last_name, self.match_score, self.final_display_acc, self.state, self.step_idx, self.wait_center = b_name, sm_score, f_acc, ValidationState.CHALLENGE, 0, False
            self.seq, self.challenge_start_time = [random.choice(["KANAN", "KIRI", "ATAS", "BAWAH", "MIRING_KANAN", "MIRING_KIRI"]), "BLINK"], time.time()
            c_pose = self.pose_estimator.estimate(face, self.detector); self.reg_pose = [c_pose.get("yaw", 0.0), c_pose.get("pitch", 0.0), c_pose.get("roll", 0.0)]

        elif self.state == ValidationState.CHALLENGE:
            act = self.seq[self.step_idx]
            self.ui.update({"status": f"{self.last_name.split(' - ', 1)[-1]}", "color": config.COLOR_CYAN, "instr": f"Tantangan {self.step_idx+1}/{len(self.seq)}: {self.CHALLENGES[act]}"})
            passed, val, tgt, salah = self._check_action(act, face)
            if salah: return self._fail("GERAKAN SALAH", config.COLOR_RED, "Akses Ditolak", wait=True)
            if (time.time() - self.challenge_start_time) > 12.0: return self._fail("WAKTU HABIS", config.COLOR_RED, "Mulai Ulang", wait=True)
            if passed:
                self.access_details.append({"tantangan": self.CHALLENGES[act], "latensi_ms": (time.time() - self.challenge_start_time) * 1000})
                UIHelper.log(f"Berhasil {self.CHALLENGES[act]} | Lat: {(time.time() - self.challenge_start_time)*1000:.0f} ms", "SUCCESS")
                self.step_idx, self.pose_hold, self.blink_passed = self.step_idx + 1, 0, False; self.ear_hist.clear()
                if self.step_idx < len(self.seq): self.challenge_start_time = time.time()
                else:
                    self.state = ValidationState.UNLOCKED; threading.Thread(target=self.door.unlock, daemon=True).start()
                    pts = self.last_name.split(" - ", 1); user_name = pts[1] if len(pts) > 1 else self.last_name
                    self.ui.update({"status": f"{user_name}", "color": config.COLOR_GREEN, "instr": f"Akses Diterima ({l_str})"})
                    print(f"\n{'='*60}\n🔓 AKSES DIBUKA | User: {user_name} | Akurasi: {self.final_display_acc:.2f}%\n{'='*60}\n")
                    if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async(user_name, pts[0] if len(pts)>1 else None, "UNLOCKED", round(self.final_display_acc, 2), l_str, self.access_details, round(self.face_val_latency + sum([d["latensi_ms"] for d in self.access_details]), 2), round(self.face_val_latency, 2))

    def run(self):
        try:
            cv2.namedWindow("Smart Door Lock", cv2.WND_PROP_FULLSCREEN if getattr(config, 'USE_FULLSCREEN', False) else cv2.WINDOW_AUTOSIZE)
            if getattr(config, 'USE_FULLSCREEN', False): cv2.setWindowProperty("Smart Door Lock", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            while self.running:
                ret, frame = self.cam.read()
                if ret:
                    with self.lock: self.shared_frame = frame.copy()
                    display = cv2.flip(frame, 1)
                    UIHelper.draw_ui(display, self.ui, getattr(self.door, 'locked', True))
                    if not getattr(config, 'USE_FULLSCREEN', False) and getattr(config, 'DISPLAY_SCALE', 1.0) != 1.0:
                        h, w = display.shape[:2]
                        display = cv2.resize(display, (int(w * config.DISPLAY_SCALE), int(h * config.DISPLAY_SCALE)))
                    cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(10) & 0xFF == ord("q"): self.running = False
        finally: self.running = False; self.cam.stop(); cv2.destroyAllWindows(); GPIO.cleanup() if GPIO_AVAILABLE else None

if __name__ == "__main__": 
    SmartDoorApp().run()