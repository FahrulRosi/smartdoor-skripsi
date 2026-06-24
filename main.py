import cv2, random, time, threading, sys, numpy as np
from datetime import datetime
from enum import Enum
import config

from camera.camera_stream import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet import MobileFaceNet
from door.door_lock import DoorLock
from liveness.head_pose import HeadPoseEstimator
from liveness.anti_spoofing import SilentAntiSpoofing  
from database.face_db import FaceDatabase

try: import RPi.GPIO as GPIO; GPIO_AVAILABLE = True
except ImportError: GPIO_AVAILABLE = False

class ValidationState(Enum): IDLE=0; RECOGNIZING=1; CHALLENGE=2; UNMATCHED=3; UNLOCKED=4

class UIHelper:
    @staticmethod
    def log(msg, lvl="INFO"): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    @staticmethod
    def print_inline(msg): sys.stdout.write(f"\r ⏳ {msg}".ljust(120)); sys.stdout.flush()

    @staticmethod
    def enhance_adaptive(frame, bbox, l_str="Normal"):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        img_yuv = cv2.cvtColor(cv2.bilateralFilter(frame, 3, 30, 30), cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.createCLAHE(clipLimit={"Normal":1.5, "Low Light":2.2, "Backlight":1.8}.get(l_str, 1.5), tileGridSize=(8, 8)).apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_aligned_crop(frame, face, target_size=(112, 112)):
        lm = getattr(face, 'landmarks', [])
        fh, fw = frame.shape[:2]
        
        if not lm or len(lm) < 363: 
            bx, by, bw, bh = face.bbox
            return frame[max(0, by):min(fh, by+bh), max(0, bx):min(fw, bx+bw)]
        
        le = np.array([(lm[33].x + lm[133].x) * fw / 2, (lm[33].y + lm[133].y) * fh / 2])
        re = np.array([(lm[263].x + lm[362].x) * fw / 2, (lm[263].y + lm[362].y) * fh / 2])

        dy = re[1] - le[1]
        dx = re[0] - le[0]
        angle = np.degrees(np.arctan2(dy, dx))

        desired_left_eye_x = 0.35
        desired_eye_y = 0.40
        desired_right_eye_x = 0.65
        
        desired_dist = (desired_right_eye_x - desired_left_eye_x) * target_size[0]
        current_dist = np.linalg.norm(re - le)
        scale = desired_dist / (current_dist + 1e-6)
        
        eye_center = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)
        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        
        t_x = target_size[0] * 0.5
        t_y = target_size[1] * desired_eye_y
        M[0, 2] += (t_x - eye_center[0])
        M[1, 2] += (t_y - eye_center[1])
        
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
            if (bg_bright := np.mean(bg_top) if bg_top.size > 0 else ambient) > 150 and (bg_bright - face_bright) > 45 and face_bright < 120: 
                return "Backlight"
        return "Low Light" if ambient < 65 else "Normal"

    @staticmethod
    def analyze_spoof_type(raw_frame, bbox):
        fh, fw = raw_frame.shape[:2]
        bx, by, bw, bh = bbox
        
        pad_w, pad_h = int(bw * 0.15), int(bh * 0.15)
        x1, y1 = max(0, bx - pad_w), max(0, by - pad_h)
        x2, y2 = min(fw, bx + bw + pad_w), min(fh, by + bh + pad_h)
        
        crop = raw_frame[y1:y2, x1:x2]
        if crop.size == 0: return "MEDIA PALSU"
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_s = np.mean(hsv[:, :, 1])
        avg_v = np.mean(hsv[:, :, 2])
        
        if avg_v > 135 and avg_s < 110: 
            return "LAYAR VIDEO"
        return "FOTO CETAK"

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
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing_int8.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.85))
        self.detector = FaceMeshDetector(min_detection_confidence=0.35, min_tracking_confidence=0.35)
        self.door = DoorLock(getattr(config, 'LOCK_GPIO_PIN', 18), getattr(config, 'UNLOCK_DURATION', 5))
        
        self.known_faces_2d = {}
        raw_db_faces = self.db.load_all_faces() or {}
        for name, data in raw_db_faces.items():
            if isinstance(data, dict) and 'embedding' in data:
                emb_data = data['embedding']
                vectors = []
                
                if isinstance(emb_data, list) and len(emb_data) > 0 and isinstance(emb_data[0], list):
                    for sub_emb in emb_data:
                        if len(sub_emb) == 512:
                            vectors.append(np.array(sub_emb, dtype=np.float32))
                
                elif isinstance(emb_data, list) and len(emb_data) == 1536:
                    vectors.append(np.array(emb_data[0:512], dtype=np.float32))
                    vectors.append(np.array(emb_data[512:1024], dtype=np.float32))
                    vectors.append(np.array(emb_data[1024:1536], dtype=np.float32))
                    
                elif isinstance(emb_data, list) and len(emb_data) == 512:
                    vectors.append(np.array(emb_data, dtype=np.float32))
                
                if vectors:
                    self.known_faces_2d[name] = vectors

        if GPIO_AVAILABLE:
            btn = getattr(config, 'BUTTON_PIN', 26)
            try: GPIO.setmode(GPIO.BCM); GPIO.setup(btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            except: pass
            try: GPIO.remove_event_detect(btn); GPIO.add_event_detect(btn, GPIO.FALLING, callback=self._manual_unlock, bouncetime=1000)
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
        print(f"\n{'='*60}"); UIHelper.log("🔓 PINTU DIBUKA MANUAL VIA TOMBOL", "SUCCESS"); print(f"{'='*60}\n")
        self._reset_state(); self.ui.update({"wait": False, "bbox": None, "status": "DIBUKA MANUAL", "color": config.COLOR_GREEN, "instr": "Tombol Ditekan", "light_cond": None})
        threading.Thread(target=self.door.unlock, daemon=True).start()

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.challenge_start_time, self.face_val_latency, self.final_display_acc = 0.0, 0.0, 0.0
        self.wait_center, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter, self.access_details = False, False, 0, [], 0, []
        self.fake_frames = self.recog_frames = self.spoof_start_time = 0; self.locked_light_cond = "Normal"; self.spoof_hist = []

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state(); self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr})
        self.pause_until = time.time() + 2.0  

    def _check_action(self, action, face):
        if action == "BLINK": 
            self.ear_hist.append(UIHelper.get_ear(face)); self.ear_hist = self.ear_hist[-3:]
            if min(self.ear_hist) <= getattr(config, 'BLINK_EAR_THRESHOLD', 0.21): self.blink_hold += 1
            else: self.blink_passed, self.blink_hold = self.blink_hold >= 1, 0
            return self.blink_passed, 1.0, 1.0, False  

        est = self.pose_estimator.estimate(face, self.detector)
        dy, dp, dr = est.get("yaw", 0) - self.reg_pose[0], est.get("pitch", 0) - self.reg_pose[1], est.get("roll", 0) - self.reg_pose[2]
        ty, tp, tr = getattr(config, 'CHALLENGE_YAW', 15.0), getattr(config, 'CHALLENGE_PITCH', 12.0), getattr(config, 'CHALLENGE_ROLL', 12.0)
        
        cmap = {"KANAN": (dy, ty, 1), "KIRI": (dy, ty, -1), "ATAS": (dp, tp, -1), "BAWAH": (dp, tp, 1), "MIRING_KANAN": (dr, tr, -1), "MIRING_KIRI": (dr, tr, 1)}
        val, thr, sign = cmap.get(action, (0, 1, 1))
        
        passed = (val > thr) if sign == 1 else (val < -thr)
        salah = (val < -thr) if sign == 1 else (val > thr)

        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 3, abs(float(val)), thr, salah

    def _check_identity(self, raw, enhanced, face, l_str):
        d_thr = {"Normal": 0.70, "Backlight": 0.65, "Low Light": 0.67}.get(l_str, 0.70)
        if not self.known_faces_2d: return "TIDAK DIKENAL", 0.0, d_thr, 0.0, False
        
        cropped = UIHelper.get_aligned_crop(enhanced, face, target_size=(112, 112))
        if cropped is None or cropped.size == 0 or (raw_emb := self.model.get_embedding(cropped)) is None: 
            return "TIDAK DIKENAL", 0.0, d_thr, 0.0, False

        q_emb = np.array(raw_emb, dtype=np.float32).flatten(); q_emb /= (np.linalg.norm(q_emb) + 1e-6)
        
        b_name, b_score = max(((n, np.max([np.dot(q_emb, e) for e in el])) for n, el in self.known_faces_2d.items()), key=lambda x: x[1], default=("", 0.0))
        
        if b_score < d_thr: self.recog_frames = 0; return "TIDAK DIKENAL", b_score, d_thr, 0.0, False
        
        req_frames = getattr(config, 'REQUIRED_STABLE_FRAMES', 6)
        self.recog_frames += 1
        return b_name, b_score, d_thr, float(b_score * 100.0), (self.recog_frames >= req_frames)

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None or time.time() < getattr(self, 'pause_until', 0): time.sleep(0.02); continue
                
            faces = self.detector.detect(frame)
            if self.state == ValidationState.UNLOCKED:
                if getattr(self.door, 'locked', True): self._reset_state(); self.ui.update({"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": "", "light_cond": None})
                else: self.ui.update({"wait": False, "bbox": max(faces, key=lambda f: f.bbox[2]*f.bbox[3]).bbox if faces else None})
                time.sleep(0.02); continue
            
            if not faces:
                self.missed_frames += 1 
                if self.missed_frames >= 30: self._fail("", wait=True)
            else: 
                self.missed_frames, face = 0, max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING): self.locked_light_cond = UIHelper.get_light_condition_dynamic(frame, face.bbox)
                
                l_str = self.locked_light_cond
                self.ui.update({"wait": False, "bbox": face.bbox, "light_cond": l_str}) 
                self._process_face(frame.copy(), UIHelper.enhance_adaptive(frame.copy(), face.bbox, l_str), face, l_str)
            time.sleep(0.01)

    def _process_face(self, raw, enhanced, face, l_str):
        if self.ui.get("status") in ("DIBUKA MANUAL", "STARTING"): return
        cx, cy = face.bbox[0] + face.bbox[2]//2, face.bbox[1] + face.bbox[3]//2
        
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(face.bbox[2:])*2.5: return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 
        if face.bbox[3] > int(config.FRAME_HEIGHT * 0.70): return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur")
        if self.state == ValidationState.IDLE: self.state, self.auth_start = ValidationState.RECOGNIZING, time.time(); return

        liveness_info = self.anti_spoof.is_real(raw.copy(), face.bbox)
        m_conf, is_m_real = float(liveness_info.get("score", 0.0)), liveness_info.get("real", True)
        as_thr = {"Normal": 0.88, "Backlight": 0.85, "Low Light": 0.85}.get(l_str, 0.85)
        
        is_actually_real = True if self.state == ValidationState.CHALLENGE else (is_m_real and m_conf >= as_thr)
        if self.state == ValidationState.CHALLENGE: self.fake_frames = 0
            
        if not is_actually_real:
            if self.fake_frames == 0: self.spoof_start_time, self.spoof_hist = time.time(), []
            self.fake_frames += 1; self.spoof_hist.append(m_conf)
            avg_score = sum(self.spoof_hist) / len(self.spoof_hist)

            if self.fake_frames >= 2: 
                lat_ms = (time.time() - self.spoof_start_time) * 1000
                sp_type = "TIDAK YAKIN (SKOR RENDAH)" if is_m_real else UIHelper.analyze_spoof_type(raw, face.bbox)
                
                if time.time() - self.last_spoof_log_time > 4.0:
                    self.last_spoof_log_time = time.time()
                    print(f"\n{'='*60}\n⚠️ SECURITY BLOCK: {'Ditolak Skor Rendah' if is_m_real else 'Serangan '+sp_type}\n   AI Confidence: {avg_score:.4f} | Latensi: {lat_ms:.0f} ms\n{'='*60}")
                    if hasattr(self.db, 'log_spoofing_async'): self.db.log_spoofing_async(round(avg_score, 4), 0.0, 0.0, sp_type, round(lat_ms, 2))
                
                return self._fail(f"SPOOF: {sp_type}", config.COLOR_RED, "Akses Ditolak", wait=False)
                
            self.ui.update({"wait": False, "bbox": face.bbox, "status": "MEMINDAI LIVENESS...", "color": config.COLOR_YELLOW, "instr": f"Tahan Posisi ({self.fake_frames}/2)..."}); return
        else:
            self.fake_frames, self.spoof_hist = 0, []

        if self.state == ValidationState.RECOGNIZING:
            # === FILTER FRAME TERBAIK (ANTI MENUNDUK & BLUR) ===
            c_pose = self.pose_estimator.estimate(face, self.detector)
            yaw, pitch, roll = c_pose.get("yaw", 0), c_pose.get("pitch", 0), c_pose.get("roll", 0)
            
            if abs(yaw) > 15 or abs(pitch) > 15 or abs(roll) > 15:
                self.ui.update({"status": "MENGAMBIL FRAME...", "color": config.COLOR_YELLOW, "instr": "Tatap lurus ke kamera"})
                return  
            
            gray_roi = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)[max(0, face.bbox[1]):min(raw.shape[0], face.bbox[1]+face.bbox[3]), max(0, face.bbox[0]):min(raw.shape[1], face.bbox[0]+face.bbox[2])]
            if gray_roi.size > 0 and cv2.Laplacian(gray_roi, cv2.CV_64F).var() < 40:
                self.ui.update({"status": "KAMERA BLUR", "color": config.COLOR_YELLOW, "instr": "Tahan posisi kepala"})
                return
            # ===================================================

            t_val = time.time()
            b_name, sm_score, d_thr, f_acc, is_recog = self._check_identity(raw, enhanced, face, l_str)
            disp_name = b_name.split(" - ", 1)[-1] if (b_name != "TIDAK DIKENAL") else "TIDAK DIKENAL"

            if b_name == "TIDAK DIKENAL":
                UIHelper.print_inline(f"Ditolak... Cosine: {sm_score:.3f} < Target Thr:{d_thr:.2f} ({l_str})")
                self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": f"Akses Ditolak (Wajah Belum Terdaftar)"})
                if time.time() - self.auth_start > 5.0: return self._fail("TIDAK DIKENAL", config.COLOR_RED, "Mulai Ulang", wait=True)
                return

            if not is_recog: 
                req_f = getattr(config, 'REQUIRED_STABLE_FRAMES', 6)
                self.ui.update({"status": "VERIFIKASI...", "color": config.COLOR_YELLOW, "instr": f"Tahan Posisi Berdiri ({self.recog_frames}/{req_f})..."})
                return

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
            cv2.namedWindow("Smart Door Lock", cv2.WINDOW_AUTOSIZE)
            while self.running:
                ret, frame = self.cam.read()
                if ret:
                    with self.lock: self.shared_frame = frame.copy()
                    display = cv2.flip(frame, 1); UIHelper.draw_ui(display, self.ui, getattr(self.door, 'locked', True)); cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(10) & 0xFF == ord("q"): self.running = False
        finally: self.running = False; self.cam.stop(); cv2.destroyAllWindows(); GPIO.cleanup() if GPIO_AVAILABLE else None

if __name__ == "__main__": 
    SmartDoorApp().run()