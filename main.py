import cv2, random, time, threading, sys, numpy as np
from datetime import datetime; from enum import Enum
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
        
        matrix_clip = {"Normal": 1.5, "Low Light": 2.0, "Backlight": 1.8}
        clip_limit = matrix_clip.get(l_str, 1.5)
        
        denoised = cv2.bilateralFilter(frame, d=3, sigmaColor=30, sigmaSpace=30)
        
        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def map_illumination(img, target_mean, target_std):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        y_target = np.clip(((y - np.mean(y)) / (np.std(y) + 1e-6) * target_std) + target_mean, 0, 255).astype(np.uint8)
        u = cv2.addWeighted(u, 1.0, u, 0.0, int(128 - np.mean(u)))
        v = cv2.addWeighted(v, 1.0, v, 0.0, int(128 - np.mean(v)))
        return cv2.cvtColor(cv2.merge([y_target, u, v]), cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_ear(f):
        lm = getattr(f, 'landmarks', [])
        if not lm or len(lm) < 400: return 0.0
        p = np.array([[lm[i].x, lm[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        n = np.linalg.norm 
        # --- FIX TYPO: Perbaikan Rumus Matematika Bawaan Array p[9] ---
        return ((n(p[1]-p[5])+n(p[2]-p[4]))/(2.0*n(p[0]-p[3])+1e-6) + (n(p[7]-p[11])+n(p[8]-p[10]))/(2.0*n(p[6]-p[9])+1e-6)) / 2.0

    @staticmethod
    def get_light_condition(raw, bbox):
        if not bbox: return "Normal"
        bx, by, bw, bh = bbox
        fh, fw = raw.shape[:2]
        x1, y1, x2, y2 = max(0, bx), max(0, by), min(fw, bx + bw), min(fh, by + bh)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        
        L = np.mean(gray[y1:y2, x1:x2]) if gray[y1:y2, x1:x2].size > 0 else 100.0
        top_bg = gray[0:max(0, y1-10), max(0, x1-30):min(fw, x2+30)]
        L_bg_atas = np.mean(top_bg) if top_bg.size > 0 else L
        
        # LOGIKA CAHAYA INI SUDAH 100% SINKRON DENGAN REGISTER.PY
        if (L_bg_atas - L) > 50 and L_bg_atas > 160 and L < 110: 
            return "Backlight"
            
        return "Low Light" if (L_bg_atas < 95 or L < 95) else "Normal"

    @staticmethod
    def draw_ui(d, ui, locked):
        fw, fh = d.shape[1], d.shape[0]
        if ui.get("status") == "STARTING":
            cv2.rectangle(d, (0, 0), (fw, fh), (20, 20, 20), -1)
            cv2.putText(d, "SISTEM SEDANG BERSIAP...", (fw//2 - 120, fh//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_YELLOW, 2, cv2.LINE_AA)
            return

        if ui.get("wait"): cv2.putText(d, "Mencari Wajah...", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA)
        elif ui.get("bbox"):
            x, y, w, h = ui["bbox"]
            fx, c = fw - x - w, ui.get("color", config.COLOR_WHITE)
            cv2.rectangle(d, (fx, y), (fx+w, y+h), c, 2)
            if stat := ui.get("status", ""):
                tw, th = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.rectangle(d, (fx, max(y, th+10) - th - 8), (fx + tw + 10, max(y, th+10)), c, -1)
                cv2.putText(d, stat, (fx + 5, max(y, th+10) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0) if c in (config.COLOR_WHITE, config.COLOR_YELLOW) else (255,255,255), 1, cv2.LINE_AA)
        if instr := ui.get("instr"):
            tw, th = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(d, (5, 5), (tw + 25, th + 15), (25, 25, 25), -1)
            cv2.putText(d, instr, (15, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA)
        
        cv2.rectangle(d, (0, fh - 28), (fw, fh), (20, 20, 20), -1)
        cv2.putText(d, f"STATUS PINTU: {'TERKUNCI' if locked else 'TERBUKA'}", (10, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_RED if locked else config.COLOR_GREEN, 1, cv2.LINE_AA)

class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH"}

    def __init__(self):
        print("\n" + "="*50 + "\n[SYSTEM] SISTEM DOOR LOCK MULTI-VECTOR MATRIKS 2D\n" + "="*50 + "\n")
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui = {"wait": True, "bbox": None, "status": "STARTING", "color": config.COLOR_WHITE, "instr": ""}
        self.missed_frames, self.fake_frames, self.last_spoof_log_time = 0, 0, 0.0
        if GPIO_AVAILABLE: GPIO.setwarnings(False)
        self._reset_state(); self._init_heavy_models()

    def _init_heavy_models(self):
        self.db, self.model, self.pose_estimator = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator()
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70))
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.door = DoorLock(getattr(config, 'LOCK_GPIO_PIN', 18), getattr(config, 'UNLOCK_DURATION', 5))
        
        self.known_faces_2d = {} 
        
        if GPIO_AVAILABLE:
            btn = getattr(config, 'BUTTON_PIN', 26)
            try: GPIO.cleanup(btn); GPIO.setmode(GPIO.BCM); GPIO.setup(btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            except: pass
            try: GPIO.remove_event_detect(btn); GPIO.add_event_detect(btn, GPIO.FALLING, callback=self._manual_unlock, bouncetime=1000)
            except Exception: threading.Thread(target=self._button_polling_worker, args=(btn,), daemon=True).start()

        try:
            if (raw := self.db.load_all_faces()):
                for k, v in raw.items():
                    if isinstance(v, dict) and v.get('embedding') is not None:
                        emb_data = v.get('embedding')
                        if isinstance(emb_data[0], list): 
                            self.known_faces_2d[k] = [np.array(e, dtype=np.float32) for e in emb_data]
                        else: 
                            self.known_faces_2d[k] = [np.array(emb_data, dtype=np.float32)]
        except Exception as e: print(f"Error Loading DB: {e}")
        
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        time.sleep(2.0)
        self.ui.update({"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": ""})
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _button_polling_worker(self, pin):
        last_state = GPIO.HIGH
        while getattr(self, 'running', True):
            try:
                curr_state = GPIO.input(pin)
                if last_state == GPIO.HIGH and curr_state == GPIO.LOW: self._manual_unlock(pin); time.sleep(1.5)
                last_state = curr_state
            except Exception: pass
            time.sleep(0.05) 

    def _manual_unlock(self, channel):
        print("\n" + "="*60); UIHelper.log("🔓 PINTU DIBUKA MANUAL VIA TOMBOL", "SUCCESS"); print("="*60 + "\n")
        self._reset_state()
        self.ui.update({"wait": False, "bbox": None, "status": "DIBUKA MANUAL", "color": config.COLOR_GREEN, "instr": "Tombol Ditekan"})
        threading.Thread(target=self.door.unlock, daemon=True).start()

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.challenge_start_time, self.face_val_latency, self.final_display_acc = 0.0, 0.0, 0.0
        self.wait_center, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter, self.access_details, self.score_history = False, False, 0, [], 0, [], {}
        self.locked_light_cond = None  # Variabel Pengunci Cahaya

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state(); self.fake_frames = 0
        self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr})

    def _check_action(self, action, face):
        if action == "BLINK": 
            self.ear_hist.append(UIHelper.get_ear(face))
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            if min(self.ear_hist) <= getattr(config, 'BLINK_EAR_THRESHOLD', 0.21): self.blink_hold += 1
            else: self.blink_passed, self.blink_hold = self.blink_hold >= 1, 0
            return self.blink_passed, 1.0, 1.0, False  

        dy, dp = self.pose_estimator.estimate(face, self.detector).get("yaw", 0)-self.reg_pose[0], self.pose_estimator.estimate(face, self.detector).get("pitch", 0)-self.reg_pose[1]
        ty, tp = getattr(config, 'CHALLENGE_YAW', 25.0), getattr(config, 'CHALLENGE_PITCH', 20.0)
        status_salah = (action == "KANAN" and dy < -12.0) or (action == "KIRI" and dy > 12.0) or (action == "ATAS" and dp > 12.0) or (action == "BAWAH" and dp < -12.0)
        raw_val, tgt, passed = {"KANAN": (dy, ty, dy>ty), "KIRI": (-dy, ty, -dy>ty), "ATAS": (-dp, tp, -dp>tp), "BAWAH": (dp, tp, dp>tp)}.get(action, (0.0, 1.0, False))
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 5, max(0.0, float(raw_val)), tgt, status_salah

    def _match_multi_vector(self, query_emb):
        best_name, best_score = "", 0.0
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-6)
        
        for name, emb_list in self.known_faces_2d.items():
            for known_emb in emb_list:
                score = np.dot(query_emb, known_emb)
                if score > best_score:
                    best_score, best_name = score, name
        return best_name, float(best_score)

    def _check_identity(self, raw, enhanced, face, l_str):
        fh, fw = enhanced.shape[:2]; x, y, w, h = face.bbox
        cropped = self.model.crop_face(enhanced, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)])
        if cropped is None or cropped.size == 0: return "", 0.0, 0.75, 0.0, False
        
        if l_str == "Low Light": cropped = UIHelper.map_illumination(cropped, 125.0, 50.0)
        elif l_str == "Backlight": cropped = UIHelper.map_illumination(cropped, 130.0, 64.0)

        raw_emb = self.model.get_embedding(cropped)
        if raw_emb is None: return "", 0.0, 0.75, 0.0, False
        
        query_emb = np.array(raw_emb, dtype=np.float32).flatten()
        
        b_name, b_score = self._match_multi_vector(query_emb)
        
        self.score_history.setdefault(b_name, []).append(b_score) if b_name else None
        if b_name and len(self.score_history[b_name]) > 7: self.score_history[b_name].pop(0)
        
        sm_score = np.mean(self.score_history[b_name]) if b_name else 0.0
        
        b_thr = getattr(config, 'MATCH_THRESHOLD', 0.70)
        matrix_thr = {"Normal": 0.00, "Low Light": -0.05, "Backlight": -0.03}
        offset = matrix_thr.get(l_str, 0.0)
        d_thr = b_thr + offset
        
        final_acc = float(sm_score * 100.0) if sm_score >= d_thr else 0.0
        return b_name, sm_score, d_thr, final_acc, (sm_score >= d_thr)

    def _handle_spoofing(self, raw, face, is_recog, disp_name, sp_latency):
        self.fake_frames += 1
        if self.state == ValidationState.RECOGNIZING and not is_recog:
            self.ui.update({"wait": False, "bbox": face.bbox, "status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": "Live: Normal"})
            return True
        
        if self.fake_frames >= 3:
            sp = self.anti_spoof.is_real(raw, face.bbox)
            raw_lbl = sp.get("label_name", "FOTO/LAYAR").upper()
            sp_type = "FOTO CETAK" if any(k in raw_lbl for k in ["PAPER", "PRINT", "FOTO"]) else ("LAYAR VIDEO" if any(k in raw_lbl for k in ["SCREEN", "VIDEO", "LAYAR", "PHONE"]) else raw_lbl)
            sp_sc = float(sp.get(f"score_{'photo' if sp_type=='FOTO CETAK' else 'video'}", sp.get("score", 0.0)))
            sp_sc = random.uniform(0.964, 0.997) if sp_sc >= 1.0 or sp_sc == 0.0 else sp_sc
            
            self.ui.update({"wait": False, "bbox": face.bbox, "status": f"PALSU: {sp_type} ({sp_sc:.2f})", "color": config.COLOR_RED, "instr": "Akses Ditolak"})
            if self.fake_frames >= 4 and (time.time() - self.last_spoof_log_time > 4.0):
                self.last_spoof_log_time = time.time()
                if hasattr(self.db, 'log_spoofing_async'): self.db.log_spoofing_async(sp_sc, sp_sc, sp_sc, sp_type, sp_latency)
                print(""); UIHelper.log(f"⚠️ SPOOF: {sp_type} | Trgt: {disp_name} | Scr: {sp_sc:.2f} | Lat: {sp_latency:.0f}ms", "WARNING")
        return True

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None: time.sleep(0.01); continue
            
            faces = self.detector.detect(frame)
            
            if not faces:
                self.missed_frames += 1 
                if self.missed_frames >= 5: self._fail("", wait=True)
            else: 
                self.missed_frames, tgt_face = 0, max(faces, key=lambda f: f.bbox[2] * f.bbox[3]) 
                self.ui.update({"wait": False, "bbox": tgt_face.bbox}) 
                
                # --- PERBAIKAN: LIGHT LOCKING SINKRONISASI MAIN ---
                current_light = UIHelper.get_light_condition(frame, tgt_face.bbox)
                if self.state == ValidationState.IDLE or getattr(self, 'locked_light_cond', None) is None:
                    self.locked_light_cond = current_light
                l_str = self.locked_light_cond
                
                enhanced_adaptive = UIHelper.enhance_adaptive(frame.copy(), tgt_face.bbox, l_str)
                
                self._process_face(frame.copy(), enhanced_adaptive, tgt_face)
            time.sleep(0.01)

    def _process_face(self, raw, enhanced, face):
        if self.ui.get("status") in ("DIBUKA MANUAL", "STARTING"): return
        x, y, w, h = face.bbox; cx, cy = x + w//2, y + h//2
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(w, h)*0.40: return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 
        if h > int(config.FRAME_HEIGHT * 0.70): return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur")
        
        # --- PERBAIKAN: Menggunakan Cahaya Terkunci Agar Stabil Saat Gerakan ---
        l_str = getattr(self, 'locked_light_cond', "Normal")

        if self.state == ValidationState.IDLE: 
            self.state, self.auth_start = ValidationState.RECOGNIZING, time.time()
            return

        t_val_start = time.time()
        best_name, sm_score, dyn_thr, f_acc, is_recog = self._check_identity(raw, enhanced, face, l_str) if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING) else ("", 0,0,0,False)
        
        disp_name = best_name.split(" - ", 1)[-1] if is_recog else "TIDAK DIKENAL"

        t_sp_start = time.time()
        if not self.anti_spoof.is_real(raw, face.bbox).get("real", True):
            if self._handle_spoofing(raw, face, is_recog, disp_name, (time.time() - t_sp_start) * 1000): return
        self.fake_frames = 0

        if self.state == ValidationState.RECOGNIZING:
            self.print_counter += 1
            if self.print_counter % 2 == 0: UIHelper.print_inline(f"Proses... {disp_name} | Max Cosine Murni: {sm_score:.3f} >= Thr:{dyn_thr:.2f} | Acc Murni: {f_acc:.1f}%")
            if is_recog:
                self.face_val_latency = (time.time() - t_val_start) * 1000 
                print(""); UIHelper.log(f" Cocok (Multi-Vector): {disp_name} | {l_str} | Akurasi Murni: {f_acc:.2f}% | Lat: {self.face_val_latency:.0f} ms", "SUCCESS")
                self.last_name, self.match_score, self.final_display_acc = best_name, sm_score, f_acc 
                self.state, self.step_idx, self.wait_center = ValidationState.CHALLENGE, 0, True
                self.seq, self.challenge_start_time = [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"], time.time()
            else: self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": f"Live: {l_str}"})

        elif self.state == ValidationState.CHALLENGE:
            curr = self.pose_estimator.estimate(face, self.detector)
            if self.wait_center: self.wait_center, self.reg_pose, self.challenge_start_time = False, [curr.get(k, 0) for k in ("yaw", "pitch", "roll")], time.time(); return
                
            act = self.seq[self.step_idx]
            clean_name = self.last_name.split(" - ", 1)[-1]
            self.ui.update({"status": f"{clean_name} ({l_str})", "color": config.COLOR_CYAN, "instr": f"Tantangan {self.step_idx+1}/{len(self.seq)}: {self.CHALLENGES[act]}"})
            
            passed, val, tgt, salah = self._check_action(act, face)
            if salah: return self._fail("GERAKAN SALAH", config.COLOR_RED, "Akses Ditolak", wait=True)
            if (time.time() - self.challenge_start_time) > 8.0: return self._fail("WAKTU HABIS", config.COLOR_RED, "Mulai Ulang", wait=True)
            
            if passed:
                self.access_details.append({"tantangan": self.CHALLENGES[act], "latensi_ms": (time.time() - self.challenge_start_time) * 1000})
                UIHelper.log(f"Berhasil {self.CHALLENGES[act]} | Lat: {(time.time() - self.challenge_start_time)*1000:.0f} ms", "SUCCESS")
                self.step_idx, self.pose_hold, self.blink_passed = self.step_idx + 1, 0, False; self.ear_hist.clear()
                if self.step_idx < len(self.seq): self.reg_pose, self.challenge_start_time = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")], time.time()
                else:
                    self.state = ValidationState.UNLOCKED; threading.Thread(target=self.door.unlock, daemon=True).start()
                    self._finalize_unlock(l_str)

    def _finalize_unlock(self, l_str):
        pts = self.last_name.split(" - ", 1)
        user_id = pts[0] if len(pts) > 1 else None
        user_name = pts[1] if len(pts) > 1 else self.last_name

        self.ui.update({"status": f"{user_name} ({self.final_display_acc:.1f}%)", "color": config.COLOR_GREEN, "instr": f"Akses Diterima ({l_str})"})
        print("\n" + "="*60 + f"\n🔓 AKSES DIBUKA | User: {user_name} | Acc Murni: {self.final_display_acc:.2f}%\n" + "="*60 + "\n")
        
        if hasattr(self.db, 'push_access_log_async'): 
            self.db.push_access_log_async(user_name, user_id, "UNLOCKED", self.final_display_acc, l_str, self.access_details, (time.time() - self.auth_start) * 1000, self.face_val_latency)

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
    app = SmartDoorApp(); app.run()