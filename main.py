import cv2, random, time, threading, numpy as np
from datetime import datetime; from enum import Enum
import config
from camera.camera_stream import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet import MobileFaceNet
from recognition.face_matcher import FaceMatcher
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
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def create_ai_frame(raw_frame, bbox):
        """
        [SOLUSI FINAL LINTAS CAHAYA]
        Memastikan wajah SELALU memiliki kecerahan yang SAMA PERSIS (Target: 130),
        baik saat mendaftar maupun saat verifikasi, di kondisi apapun.
        """
        x, y, w, h = bbox
        fh, fw = raw_frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x+w), min(fh, y+h)
        
        # 1. Konversi ke YUV untuk mengatur saluran cahaya (Y)
        img_yuv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2YUV)
        y_ch, u_ch, v_ch = cv2.split(img_yuv)
        
        # 2. Ambil HANYA area wajah untuk diukur kecerahan aslinya
        face_y = y_ch[y1:y2, x1:x2]
        mean_y = np.mean(face_y) if face_y.size > 0 else 130.0
        
        # 3. Hitung pengali mutlak (Gain) agar wajah selalu menyentuh angka 130.0
        target_brightness = 130.0
        if mean_y > 5.0:
            alpha = target_brightness / mean_y
            # Batasi agar tidak terjadi over-exposure ekstrem (max 3.5x lebih terang, min 0.5x lebih gelap)
            alpha = min(max(alpha, 0.5), 3.5)
            
            # 4. Kalikan seluruh frame dengan pengali (wajah akan sempurna, background akan menyesuaikan)
            y_ch = cv2.convertScaleAbs(y_ch, alpha=alpha, beta=0)
            
        # 5. Terapkan CLAHE untuk mengunci detail kontras (hidung/mata tetap tajam)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_ch = clahe.apply(y_ch)
        
        return cv2.cvtColor(cv2.merge((y_ch, u_ch, v_ch)), cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_ear(f):
        lm = getattr(f, 'landmarks', [])
        if not lm or len(lm) < 400: return 0.0
        p = np.array([[lm[i].x, lm[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        n = np.linalg.norm 
        return ((n(p[1]-p[5])+n(p[2]-p[4]))/(2.0*n(p[0]-p[3])+1e-6) + (n(p[7]-p[11])+n(p[8]-p[10]))/(2.0*n(p[6]-p[9])+1e-6)) / 2.0

    @staticmethod
    def draw_ui(d, ui, locked):
        if ui.get("instr"):
            w, h = cv2.getTextSize(ui.get("instr"), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(d, (10, 10), (w+40, h+30), (0,0,0), -1)
            cv2.putText(d, ui.get("instr"), (20, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
        if ui.get("wait"): cv2.putText(d, "Menunggu Wajah...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
        elif ui.get("bbox"):
            x, y, w, h = ui["bbox"]
            fx, c = config.FRAME_WIDTH - x - w, ui.get("color", config.COLOR_WHITE)
            cv2.rectangle(d, (fx, y), (fx+w, y+h), c, 3)
            if stat := ui.get("status", ""):
                tw = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
                cv2.rectangle(d, (fx, y-35), (fx+tw+15, y-5), c, -1)
                cv2.putText(d, stat, (fx+8, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(d, f"PINTU: {'TERKUNCI' if locked else 'TERBUKA'}", (10, config.FRAME_HEIGHT-30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, config.COLOR_RED if locked else config.COLOR_GREEN, 2)

class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH"}

    def __init__(self):
        UIHelper.log("Sistem Smart Door Lock Aktif", "SYSTEM")
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui, self.missed_frames, self.spoof_score = {"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": ""}, 0, 1.0
        
        if GPIO_AVAILABLE:
            self.btn_pin = getattr(config, 'BUTTON_PIN', 23) 
            try:
                GPIO.setmode(GPIO.BCM); GPIO.setup(self.btn_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
                GPIO.add_event_detect(self.btn_pin, GPIO.FALLING, callback=self._manual_unlock, bouncetime=500)
            except Exception as e: UIHelper.log(f"Gagal init tombol: {e}", "WARNING")
        
        self._reset_state(); self.fake_frames = 0
        self._init_heavy_models()

    def _init_heavy_models(self):
        self.db, self.model, self.pose_estimator = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator()
        spoof_thr = getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70) 
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), spoof_thr)
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.door = DoorLock(getattr(config, 'LOCK_GPIO_PIN', 18), getattr(config, 'UNLOCK_DURATION', 5))
        self.pose_estimator = HeadPoseEstimator()
        self.matcher = FaceMatcher(getattr(config, 'MATCH_THRESHOLD', 0.42)) 
        
        try:
            if (raw := self.db.load_all_faces()):
                faces = {k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if isinstance(v, dict) and v.get('embedding') is not None}
                self.matcher.load_faces(faces) if hasattr(self.matcher, 'load_faces') else setattr(self.matcher, 'known_faces', faces)
                UIHelper.log(f"Memori database dimuat: {len(faces)} wajah siap.", "SUCCESS")
        except Exception as e: UIHelper.log(f"Gagal muat memori: {e}", "WARNING")
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _manual_unlock(self, channel):
        if hasattr(self, 'door') and self.door and getattr(self.door, 'locked', True):
            UIHelper.log("🔘 Tombol Manual Ditekan! Pintu Terbuka.", "SUCCESS")
            self.ui.update({"status": "DIBUKA MANUAL", "color": config.COLOR_GREEN, "instr": ""})
            threading.Thread(target=self.door.unlock, daemon=True).start()
            if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async("Manual", "UNLOCKED", 100.0)

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.wait_center, self.center_hold, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter, self.access_details = False, 0, False, 0, [], 0, []

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state(); self.fake_frames = 0
        self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr})

    def _check_action(self, action, face):
        if action == "BLINK": 
            ear = UIHelper.get_ear(face)
            self.ear_hist.append(ear)
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            tgt = getattr(config, 'BLINK_EAR_THRESHOLD', 0.21)
            if min(self.ear_hist) <= tgt: self.blink_hold += 1
            else:
                if self.blink_hold >= 1: self.blink_passed = True
                self.blink_hold = 0
            return self.blink_passed, ear, tgt
        p, ref = self.pose_estimator.estimate(face, self.detector), self.reg_pose
        dy, dp, dr = p.get("yaw", 0)-ref[0], p.get("pitch", 0)-ref[1], p.get("roll", 0)-ref[2]
        ty, tp, tr = getattr(config, 'CHALLENGE_YAW', 25.0), getattr(config, 'CHALLENGE_PITCH', 20.0), getattr(config, 'CHALLENGE_ROLL', 25.0)
        val, tgt, passed = {"KANAN": (dy, ty, dy>ty), "KIRI": (-dy, ty, -dy>ty), "ATAS": (-dp, tp, -dp>tp), "BAWAH": (dp, tp, dp>tp)}.get(action, (0.0, 1.0, False))
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 6, val, tgt

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None: time.sleep(0.01); continue
            t0, raw = time.time(), frame.copy()
            enhanced = UIHelper.enhance_frame(raw)
            faces = self.detector.detect(enhanced)
            if not faces: 
                self.missed_frames += 1 
                if self.missed_frames >= 5: self._fail("", wait=True)
            else: 
                self.missed_frames, target_face = 0, max(faces, key=lambda f: f.bbox[2] * f.bbox[3]) 
                self.ui.update({"wait": False, "bbox": target_face.bbox}) 
                self._process_face(raw, enhanced, target_face)
            self.latency = (time.time() - t0) * 1000; time.sleep(0.01) 

    def _process_face(self, raw, enhanced, face):
        x, y, w, h = face.bbox; cx, cy = x + w//2, y + h//2
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(w, h)*0.40: return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 

        if h > int(config.FRAME_HEIGHT * 0.50): return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur sedikit")
        
        sp = self.anti_spoof.is_real(raw, face.bbox)
        self.spoof_score = sp.get("score", 1.0)
        
        if not sp.get("real", True):
            self.fake_frames += 1 
            if self.fake_frames < 10: 
                print(f"\r\033[K[Memeriksa] INDIKASI PALSU! | Spoofing: {self.spoof_score:.2f}", end="", flush=True)
            elif self.fake_frames == 10: 
                print()
                detected_type = sp.get("label_name", "Spoofing Tidak Diketahui")
                UIHelper.log(f"⚠️ Serangan Spoofing Terdeteksi! Tipe: {detected_type} (Skor: {self.spoof_score:.2f})", "WARNING")
                if hasattr(self.db, 'log_spoofing_async'): self.db.log_spoofing_async(self.spoof_score, detected_type)
                self._fail(f"{detected_type.upper()} (Spoof: {self.spoof_score:.2f})")
                self.spoof_score = sp.get("score", 1.0) 
            return
        
        self.fake_frames = 0
        
        gray_live = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh_l, fw_l = gray_live.shape
        x1_l, y1_l = max(0, x), max(0, y)
        x2_l, y2_l = min(fw_l, x+w), min(fh_l, y+h)
        
        face_roi_live = gray_live[y1_l:y2_l, x1_l:x2_l]
        L_live = np.mean(face_roi_live) if face_roi_live.size > 0 else 100.0
        
        mask_live = np.ones((fh_l, fw_l), dtype=bool)
        mask_live[y1_l:y2_l, x1_l:x2_l] = False
        L_bg_live = np.mean(gray_live[mask_live]) if np.any(mask_live) else L_live

        if (L_bg_live - L_live) > 40 and L_bg_live > 120: 
            l_str = "Backlight"
        elif L_bg_live < 85 or L_live < 85: 
            l_str = "Low Light"
        else: 
            l_str = "Normal"

        if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
            if self.state == ValidationState.IDLE: self.state, self.auth_start = ValidationState.RECOGNIZING, time.time(); return
            fh, fw = raw.shape[:2]
            
            # ---> MENGEKSTRAK WAJAH MENGGUNAKAN STANDAR MUTLAK <---
            ai_frame = UIHelper.create_ai_frame(raw, face.bbox)
            
            if (raw_emb := self.model.get_embedding(self.model.crop_face(ai_frame, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)]))) is None: return
            emb = np.array(raw_emb, dtype=np.float32).flatten()
            match = self.matcher.match(emb / (np.linalg.norm(emb) + 1e-6))
            
            if match.get("matched", False):
                self.last_name, self.match_score, self.state, self.step_idx, self.wait_center, self.center_hold = match["name"], match.get("score", 0.0), ValidationState.CHALLENGE, 0, True, 0
                self.seq = [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"]
                print()
                UIHelper.log(f"Wajah {self.last_name} Dikenali ({l_str}). Memulai Liveness...", "SUCCESS")
            else: 
                self.print_counter += 1
                if self.print_counter % 20 == 0: 
                    UIHelper.log(f"Wajah TIDAK DIKENAL (Spoofing: {self.spoof_score:.2f})", "WARNING")
                self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": f"Cahaya: {l_str}"})

        elif self.state == ValidationState.CHALLENGE:
            curr = self.pose_estimator.estimate(face, self.detector)
            if self.wait_center:
                if abs(curr.get("yaw", 0)) < 15 and abs(curr.get("pitch", 0)) < 15 and abs(curr.get("roll", 0)) < 12:
                    self.wait_center, self.center_hold, self.reg_pose = False, 0, [curr.get(k, 0) for k in ("yaw", "pitch", "roll")]
                else: 
                    return self.ui.update({"status": f"{self.last_name}", "color": config.COLOR_CYAN, "instr": "Tatap lurus kamera..."})
                
            act, inst = self.seq[self.step_idx], self.CHALLENGES[self.seq[self.step_idx]]
            self.ui.update({"status": f"{self.last_name} ({l_str})", "color": config.COLOR_CYAN, "instr": f"Tahap {self.step_idx+1}/{len(self.seq)}: {inst}"})
            passed, val, tgt = self._check_action(act, face)
            
            self.print_counter += 1
            if self.print_counter % 3 == 0: 
                print(f"\r\033[K[{inst}] {'EAR Mata' if act=='BLINK' else 'Sudut'}: {val:.2f}{'' if act=='BLINK' else '°'} | Target: {tgt:.2f} | Lat: {getattr(self, 'latency', 0):.1f}ms   ", end="", flush=True)
            
            if passed:
                print()
                UIHelper.log(f"✅ {inst} SELESAI", "SUCCESS")
                self.access_details.append({"tantangan": inst, "skor_asli": round(val, 2), "target": round(tgt, 2), "latensi_ms": round(getattr(self, 'latency', 0), 1)})
                self.step_idx, self.pose_hold, self.blink_hold, self.blink_passed = self.step_idx + 1, 0, 0, False; self.ear_hist.clear()
                if self.step_idx < len(self.seq): self.reg_pose, self.wait_center = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")], False 
                else:
                    self.state = ValidationState.UNLOCKED
                    threading.Thread(target=self.door.unlock, daemon=True).start()
                    self._finalize_unlock(raw, face.bbox)

    def _finalize_unlock(self, raw, bbox):
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(fw, bbox[0]+bbox[2]), min(fh, bbox[1]+bbox[3])
        
        face_roi = gray[y1:y2, x1:x2]
        L_face = np.mean(face_roi) if face_roi.size > 0 else 100.0
        
        mask = np.ones((fh, fw), dtype=bool)
        mask[y1:y2, x1:x2] = False
        L_bg = np.mean(gray[mask]) if np.any(mask) else L_face

        if (L_bg - L_face) > 40 and L_bg > 120: 
            light_cond = f"Backlight (F:{L_face:.0f}/B:{L_bg:.0f})"
        elif L_bg < 85 or L_face < 85: 
            light_cond = f"Low Light (F:{L_face:.0f}/B:{L_bg:.0f})"
        else: 
            light_cond = f"Normal (F:{L_face:.0f}/B:{L_bg:.0f})"

        pure_similarity = self.match_score
        UIHelper.log(f"🧪 [DATA UJI PENGAKUAN] Cosine Similarity Murni: {pure_similarity:.4f}", "SYSTEM")
        
        final_acc = pure_similarity * 100.0
        final_acc = min(100.0, max(0.0, final_acc))
        
        self.ui.update({"status": f"SELAMAT DATANG", "color": config.COLOR_GREEN, "instr": ""})
        UIHelper.log(f"🔓 AKSES DIBERIKAN. Waktu: {time.time() - self.auth_start:.2f}s", "SUCCESS")
        UIHelper.log(f"📊 Skor Konfidensi: {final_acc:.2f}% | Kecerahan: {light_cond}", "SUCCESS")
        if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async(self.last_name, "UNLOCKED", final_acc, light_cond, self.access_details)

    def run(self):
        window_name = "Smart Door Lock"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            while self.running:
                ret, frame = self.cam.read()
                if ret:
                    with self.lock: self.shared_frame = frame.copy()
                    display = cv2.flip(frame, 1)
                    is_door_locked = getattr(self.door, 'locked', True) if hasattr(self, 'door') and self.door else True
                    UIHelper.draw_ui(display, self.ui, is_door_locked)
                    cv2.imshow(window_name, display)
                if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]: self.running = False; break
        finally: 
            self.running = False; time.sleep(0.5)
            if hasattr(self, 'cam') and self.cam: self.cam.stop()
            if hasattr(self, 'door') and self.door: self.door.cleanup()
            cv2.destroyAllWindows()
            if GPIO_AVAILABLE: 
                try: GPIO.remove_event_detect(getattr(config, 'BUTTON_PIN', 23))
                except: pass
                GPIO.cleanup()

if __name__ == "__main__": SmartDoorApp().run()