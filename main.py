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
        img_yuv = cv2.cvtColor(cv2.bilateralFilter(frame, 3, 30, 30), cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.createCLAHE(clipLimit={"Normal":1.5, "Low Light":2.0, "Backlight":1.8}.get(l_str, 1.5), tileGridSize=(8, 8)).apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def map_illumination(img, t_mean, t_std):
        y, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
        y = np.clip(((y - np.mean(y)) / (np.std(y) + 1e-6) * t_std) + t_mean, 0, 255).astype(np.uint8)
        u, v = cv2.addWeighted(u, 1.0, u, 0.0, int(128 - np.mean(u))), cv2.addWeighted(v, 1.0, v, 0.0, int(128 - np.mean(v)))
        return cv2.cvtColor(cv2.merge([y, u, v]), cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_ear(f):
        lm = getattr(f, 'landmarks', [])
        if not lm or len(lm) < 400: return 0.0
        p = np.array([[lm[i].x, lm[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        n = np.linalg.norm 
        return ((n(p[1]-p[5])+n(p[2]-p[4]))/(2.0*n(p[0]-p[3])+1e-6) + (n(p[7]-p[11])+n(p[8]-p[10]))/(2.0*n(p[6]-p[9])+1e-6)) / 2.0

    @staticmethod
    def get_light_condition(raw, bbox):
        if not bbox: return "Normal"
        bx, by, bw, bh = bbox; fh, fw = raw.shape[:2]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        
        x1, y1, x2, y2 = max(0, bx), max(0, by), min(fw, bx + bw), min(fh, by + bh)
        L = np.mean(gray[y1:y2, x1:x2]) if gray[y1:y2, x1:x2].size > 0 else 100.0
        oL = np.mean(gray) 
        
        top = gray[max(0, by-100):max(0, by-10), max(0, bx-50):min(fw, bx+bw+50)]
        left = gray[max(0, by-20):min(fh, by+bh+20), max(0, bx-100):max(0, bx-10)]
        right = gray[max(0, by-20):min(fh, by+bh+20), min(fw, bx+bw+10):min(fw, bx+bw+100)]
        
        max_bg = max(np.mean(top) if top.size else L, np.mean(left) if left.size else L, np.mean(right) if right.size else L)
        
        if (max_bg > 150 and (max_bg - L) > 30) or (oL > 150 and (oL - L) > 30): 
            return "Backlight"
        if L < 80 and max_bg < 120: 
            return "Low Light"
        return "Normal"

    @staticmethod
    def draw_ui(d, ui, locked):
        fw, fh = d.shape[1], d.shape[0]; c_font = cv2.FONT_HERSHEY_SIMPLEX
        if ui.get("status") == "STARTING":
            cv2.rectangle(d, (0, 0), (fw, fh), (20, 20, 20), -1)
            return cv2.putText(d, "SISTEM SEDANG BERSIAP...", (fw//2 - 120, fh//2 - 10), c_font, 0.5, config.COLOR_YELLOW, 2, cv2.LINE_AA)

        if l_cond := ui.get("light_cond"):
            cv2.putText(d, f"Cahaya: {l_cond}", (15, 55), c_font, 0.45, config.COLOR_GREEN if l_cond == "Normal" else (config.COLOR_YELLOW if l_cond == "Low Light" else config.COLOR_RED), 1, cv2.LINE_AA)

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
    CHALLENGES = {
        "BLINK": "Kedipkan Mata", 
        "KANAN": "Toleh KANAN", 
        "KIRI": "Toleh KIRI", 
        "ATAS": "Dongak ATAS", 
        "BAWAH": "Tunduk BAWAH", 
        "MIRING_KANAN": "Miring KANAN", 
        "MIRING_KIRI": "Miring KIRI"
    }

    def __init__(self):
        print(f"\n{'='*50}\n[SYSTEM] SISTEM DOOR LOCK MULTI-VECTOR MATRIKS 2D\n{'='*50}\n")
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui = {"wait": True, "bbox": None, "status": "STARTING", "color": config.COLOR_WHITE, "instr": "", "light_cond": None}
        self.missed_frames = self.fake_frames = self.last_spoof_log_time = 0
        if GPIO_AVAILABLE: GPIO.setwarnings(False)
        self._reset_state(); self._init_heavy_models()

    def _init_heavy_models(self):
        self.db, self.model, self.pose_estimator = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator()
        
        spoof_thr = getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.85)
        spoof_thr = min(spoof_thr, 0.75) 
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), spoof_thr)
        
        self.detector, self.door = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5), DoorLock(getattr(config, 'LOCK_GPIO_PIN', 18), getattr(config, 'UNLOCK_DURATION', 5))
        self.known_faces_2d = {k: [np.array(e, dtype=np.float32) for e in (v['embedding'] if isinstance(v['embedding'][0], list) else [v['embedding']])] for k, v in (self.db.load_all_faces() or {}).items() if isinstance(v, dict) and v.get('embedding')}
        
        if GPIO_AVAILABLE:
            btn = getattr(config, 'BUTTON_PIN', 26)
            try: GPIO.cleanup(btn); GPIO.setmode(GPIO.BCM); GPIO.setup(btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            except: pass
            try: GPIO.remove_event_detect(btn); GPIO.add_event_detect(btn, GPIO.FALLING, callback=self._manual_unlock, bouncetime=1000)
            except Exception: threading.Thread(target=self._button_polling_worker, args=(btn,), daemon=True).start()

        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        time.sleep(2.0)
        self.ui.update({"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": "", "light_cond": None})
        self.app_start_time = time.time()  
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _button_polling_worker(self, pin):
        last_state = GPIO.HIGH
        while self.running:
            try:
                curr_state = GPIO.input(pin)
                if last_state == GPIO.HIGH and curr_state == GPIO.LOW: self._manual_unlock(pin); time.sleep(1.5)
                last_state = curr_state
            except Exception: pass
            time.sleep(0.05) 

    def _manual_unlock(self, channel):
        print(f"\n{'='*60}"); UIHelper.log("🔓 PINTU DIBUKA MANUAL VIA TOMBOL", "SUCCESS"); print(f"{'='*60}\n")
        self._reset_state()
        self.ui.update({"wait": False, "bbox": None, "status": "DIBUKA MANUAL", "color": config.COLOR_GREEN, "instr": "Tombol Ditekan", "light_cond": None})
        threading.Thread(target=self.door.unlock, daemon=True).start()

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.challenge_start_time, self.face_val_latency, self.final_display_acc = 0.0, 0.0, 0.0
        self.wait_center, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter, self.access_details, self.score_history = False, False, 0, [], 0, [], {}
        self.locked_light_cond, self.fake_frames = None, 0  

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state()
        self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr, "light_cond": None})

    def _check_action(self, action, face):
        if action == "BLINK": 
            self.ear_hist.append(UIHelper.get_ear(face))
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            if min(self.ear_hist) <= getattr(config, 'BLINK_EAR_THRESHOLD', 0.21): self.blink_hold += 1
            else: self.blink_passed, self.blink_hold = self.blink_hold >= 1, 0
            return self.blink_passed, 1.0, 1.0, False  

        est = self.pose_estimator.estimate(face, self.detector)
        dy = est.get("yaw", 0) - self.reg_pose[0]
        dp = est.get("pitch", 0) - self.reg_pose[1]
        dr = est.get("roll", 0) - self.reg_pose[2] 
        
        ty = getattr(config, 'CHALLENGE_YAW', 25.0)
        tp = getattr(config, 'CHALLENGE_PITCH', 20.0)
        tr = getattr(config, 'CHALLENGE_ROLL', 20.0)
        
        salah = (action=="KANAN" and dy<-18.0) or (action=="KIRI" and dy>18.0) or \
                (action=="ATAS" and dp>18.0) or (action=="BAWAH" and dp<-18.0) or \
                (action=="MIRING_KANAN" and dr<-18.0) or (action=="MIRING_KIRI" and dr>18.0)
        
        raw_val, tgt, passed = {
            "KANAN": (dy, ty, dy>ty), "KIRI": (-dy, ty, -dy>ty), 
            "ATAS": (-dp, tp, -dp>tp), "BAWAH": (dp, tp, dp>tp),
            "MIRING_KANAN": (dr, tr, dr>tr), "MIRING_KIRI": (-dr, tr, -dr>tr)
        }.get(action, (0.0, 1.0, False))
        
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 5, max(0.0, float(raw_val)), tgt, salah

    def _check_identity(self, raw, enhanced, face, l_str):
        fh, fw = enhanced.shape[:2]; x, y, w, h = face.bbox
        cropped = self.model.crop_face(enhanced, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)])
        if cropped is None or cropped.size == 0: return "", 0.0, 0.75, 0.0, False
        
        if l_str == "Low Light": cropped = UIHelper.map_illumination(cropped, 125.0, 50.0)
        elif l_str == "Backlight": cropped = UIHelper.map_illumination(cropped, 130.0, 64.0)

        if (raw_emb := self.model.get_embedding(cropped)) is None: return "", 0.0, 0.75, 0.0, False
        q_emb = np.array(raw_emb, dtype=np.float32).flatten(); q_emb /= (np.linalg.norm(q_emb) + 1e-6)
        
        b_name, b_score = max(((n, np.max([np.dot(q_emb, e) for e in el])) for n, el in self.known_faces_2d.items()), key=lambda x: x[1], default=("", 0.0))
        
        if b_name:
            self.score_history.setdefault(b_name, []).append(b_score)
            if len(self.score_history[b_name]) > 7: self.score_history[b_name].pop(0)
        
        sm_score = np.mean(self.score_history[b_name]) if b_name else 0.0
        d_thr = getattr(config, 'MATCH_THRESHOLD', 0.70) + {"Normal": 0.0, "Low Light": -0.05, "Backlight": -0.03}.get(l_str, 0.0)
        
        return b_name, sm_score, d_thr, float(sm_score * 100.0) if sm_score >= d_thr else 0.0, sm_score >= d_thr

    def _handle_spoofing(self, raw, face, is_recog, disp_name, sp_lat, l_str, liveness_info=None):
        self.fake_frames += 1
        
        sp = liveness_info if liveness_info is not None else self.anti_spoof.is_real(raw, face.bbox)
        lbl = sp.get("label_name", "FOTO/LAYAR").upper()
        sp_type = "FOTO CETAK" if any(k in lbl for k in ["PAPER", "PRINT", "FOTO"]) else ("LAYAR VIDEO" if any(k in lbl for k in ["SCREEN", "VIDEO", "LAYAR", "PHONE"]) else lbl)
        
        sp_sc = float(sp.get("score", sp.get(f"score_{'photo' if sp_type=='FOTO CETAK' else 'video'}", 0.99)))
        
        if self.fake_frames < 8:
            self.ui.update({
                "wait": False, 
                "bbox": face.bbox, 
                "status": f"MEMERIKSA LIVENESS ({self.fake_frames}/8)", 
                "color": config.COLOR_RED, 
                "instr": f"Terindikasi {sp_type}"
            })
        else:
            self.ui.update({
                "wait": False, 
                "bbox": face.bbox, 
                "status": f"PALSU: {sp_type} ({sp_sc:.2f})", 
                "color": config.COLOR_RED, 
                "instr": "Akses Ditolak"
            })
            
            if time.time() - self.last_spoof_log_time > 4.0:
                self.last_spoof_log_time = time.time()
                if hasattr(self.db, 'log_spoofing_async'): 
                    self.db.log_spoofing_async(round(sp_sc, 2), round(sp_sc, 2), round(sp_sc, 2), sp_type, round(sp_lat, 2))
                print(f"\n⚠️ SPOOF: {sp_type} | Trgt: {disp_name} | Scr: {sp_sc:.2f} | Lat: {sp_lat:.0f}ms")
                
        return True 

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None: time.sleep(0.01); continue
            
            faces = self.detector.detect(frame)
            if self.state == ValidationState.UNLOCKED:
                if getattr(self.door, 'locked', True):
                    self._reset_state(); self.ui.update({"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": "", "light_cond": None})
                else:
                    self.ui.update({"wait": False, "bbox": max(faces, key=lambda f: f.bbox[2]*f.bbox[3]).bbox if faces else None})
                time.sleep(0.02); continue
            
            if not faces:
                self.missed_frames += 1 
                # PERBAIKAN: Toleransi wajah hilang sesaat dinaikkan dari 5 ke 15 frame
                if self.missed_frames >= 15: self._fail("", wait=True)
            else: 
                self.missed_frames, face = 0, max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                
                if self.state == ValidationState.IDLE or not self.locked_light_cond:
                    self.locked_light_cond = UIHelper.get_light_condition(frame, face.bbox)
                
                l_str = "Normal" if time.time() - self.app_start_time < 2.5 else self.locked_light_cond
                self.ui.update({"wait": False, "bbox": face.bbox, "light_cond": l_str}) 
                self._process_face(frame.copy(), UIHelper.enhance_adaptive(frame.copy(), face.bbox, l_str), face, l_str)
            time.sleep(0.01)

    def _process_face(self, raw, enhanced, face, l_str):
        if self.ui.get("status") in ("DIBUKA MANUAL", "STARTING"): return
        cx, cy = face.bbox[0] + face.bbox[2]//2, face.bbox[1] + face.bbox[3]//2
        
        # PERBAIKAN: Toleransi batas center wajah ("WAJAH BERGANTI") dilebarkan dari 0.4 ke 0.6
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(face.bbox[2:])*0.6: return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        
        self.prev_center = (cx, cy) 
        if face.bbox[3] > int(config.FRAME_HEIGHT * 0.70): return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur")

        if self.state == ValidationState.IDLE: self.state, self.auth_start = ValidationState.RECOGNIZING, time.time(); return

        t_val = time.time()
        b_name, sm_score, d_thr, f_acc, is_recog = self._check_identity(raw, enhanced, face, l_str) if self.state == ValidationState.RECOGNIZING else ("", 0,0,0,False)
        disp_name = b_name.split(" - ", 1)[-1] if is_recog else "TIDAK DIKENAL"

        t_sp = time.time()
        if self.state == ValidationState.RECOGNIZING:
            liveness_info = self.anti_spoof.is_real(raw, face.bbox)
            
            if not liveness_info.get("real", True):
                if self._handle_spoofing(raw, face, is_recog, disp_name, (time.time() - t_sp) * 1000, l_str, liveness_info): return
            else: 
                self.fake_frames = 0
        else: self.fake_frames = 0

        if self.state == ValidationState.RECOGNIZING:
            self.print_counter += 1
            if self.print_counter % 2 == 0: UIHelper.print_inline(f"Proses... {disp_name} | Max Cosine Murni: {sm_score:.3f} >= Thr:{d_thr:.2f} | Acc Murni: {f_acc:.1f}%")
            if is_recog:
                self.face_val_latency = (time.time() - t_val) * 1000 
                UIHelper.log(f"\nCocok (Multi-Vector): {disp_name} | {l_str} | Akurasi Murni: {f_acc:.2f}% | Lat: {self.face_val_latency:.0f} ms", "SUCCESS")
                self.last_name, self.match_score, self.final_display_acc, self.state, self.step_idx, self.wait_center = b_name, sm_score, f_acc, ValidationState.CHALLENGE, 0, True
                self.seq, self.challenge_start_time = [random.choice(["KANAN", "KIRI", "ATAS", "BAWAH", "MIRING_KANAN", "MIRING_KIRI"]), "BLINK"], time.time()
                
            else: self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": f"Live: {l_str}"})

        elif self.state == ValidationState.CHALLENGE:
            curr = self.pose_estimator.estimate(face, self.detector)
            if self.wait_center: self.wait_center, self.reg_pose, self.challenge_start_time = False, [curr.get(k, 0) for k in ("yaw", "pitch", "roll")], time.time(); return
                
            act = self.seq[self.step_idx]
            self.ui.update({"status": f"{self.last_name.split(' - ', 1)[-1]} ({l_str})", "color": config.COLOR_CYAN, "instr": f"Tantangan {self.step_idx+1}/{len(self.seq)}: {self.CHALLENGES[act]}"})
            
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
                    pts = self.last_name.split(" - ", 1); user_name = pts[1] if len(pts) > 1 else self.last_name
                    
                    self.ui.update({"status": f"{user_name}", "color": config.COLOR_GREEN, "instr": f"Akses Diterima ({l_str})"})
                    
                    print(f"\n{'='*60}\n🔓 AKSES DIBUKA | User: {user_name} | Acc Murni: {self.final_display_acc:.2f}%\n{'='*60}\n")
                    
                    if hasattr(self.db, 'push_access_log_async'):
                        total_waktu_proses = self.face_val_latency + sum([d["latensi_ms"] for d in self.access_details])
                        
                        akurasi_db = round(self.final_display_acc, 2)
                        latensi_wajah_db = round(self.face_val_latency, 2)
                        total_waktu_db = round(total_waktu_proses, 2)
                        
                        self.db.push_access_log_async(user_name, pts[0] if len(pts)>1 else None, "UNLOCKED", akurasi_db, l_str, self.access_details, total_waktu_db, latensi_wajah_db)

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

if __name__ == "__main__": SmartDoorApp().run()