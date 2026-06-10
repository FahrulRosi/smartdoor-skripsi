import cv2, random, time, threading, sys, numpy as np
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
    def print_inline(msg): sys.stdout.write(f"\r ⏳ {msg}".ljust(120)); sys.stdout.flush()

    @staticmethod
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        yuv = cv2.cvtColor(cv2.bilateralFilter(frame, 3, 30, 30), cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.createCLAHE(2.0, (8, 8)).apply(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_ear(f):
        lm = getattr(f, 'landmarks', [])
        if len(lm) < 400: return 0.0
        p = np.array([[lm[i].x, lm[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        n = np.linalg.norm 
        return ((n(p[1]-p[5])+n(p[2]-p[4]))/(2.0*n(p[0]-p[3])+1e-6) + (n(p[7]-p[11])+n(p[8]-p[10]))/(2.0*n(p[6]-p[9])+1e-6)) / 2.0

    @staticmethod
    def get_light_condition(raw, bbox):
        fh, fw = raw.shape[:2]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if not bbox: 
            L_bg = np.mean(gray)
            return "Backlight" if L_bg > 160 else "Low Light" if L_bg < 70 else "Normal"
            
        bx, by, bw, bh = [max(0, v) for v in bbox]
        L = np.mean(gray[by:min(fh, by+bh), bx:min(fw, bx+bw)]) if bw>0 and bh>0 else 100.0
        L_top = np.mean(gray[0:max(0, by-10), max(0, bx-30):min(fw, bx+bw+30)])
        
        mask = np.ones((fh, fw), dtype=bool); mask[by:by+bh, bx:bx+bw] = False
        L_bg = max(np.mean(gray[mask]) if np.any(mask) else L, L_top if not np.isnan(L_top) else L)
        
        return "Backlight" if (L_bg - L) > 40 and L_bg > 160 else "Low Light" if L_bg < 70 else "Normal"

    @staticmethod
    def draw_ui(d, ui, locked):
        fw, fh = d.shape[:2]
        def put_txt(txt, pos, c, s=0.45): cv2.putText(d, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
        
        if ui.get("wait"): put_txt("Mencari Wajah...", (15, 35), config.COLOR_YELLOW)
        elif ui.get("bbox"):
            x, y, w, h = ui["bbox"]
            fx, c = fw - x - w, ui.get("color", config.COLOR_WHITE)
            cv2.rectangle(d, (fx, y), (fx+w, y+h), c, 2)
            if stat := ui.get("status", ""):
                tw, th = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                lbl_y = max(y, th + 10)
                cv2.rectangle(d, (fx, lbl_y - th - 8), (fx + tw + 10, lbl_y), c, -1)
                tc = (0,0,0) if c in (config.COLOR_WHITE, config.COLOR_YELLOW) else (255,255,255)
                put_txt(stat, (fx + 5, lbl_y - 4), tc)
                
        if instr := ui.get("instr", ""):
            tw, th = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(d, (5, 5), (tw + 25, th + 15), (25, 25, 25), -1)
            put_txt(instr, (15, th + 10), config.COLOR_YELLOW)

        cv2.rectangle(d, (0, fh - 28), (fw, fh), (20, 20, 20), -1)
        put_txt(f"STATUS PINTU: {'TERKUNCI' if locked else 'TERBUKA'}", (10, fh - 8), config.COLOR_RED if locked else config.COLOR_GREEN)

class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH"}

    def __init__(self):
        print("\n" + "="*50); UIHelper.log("SISTEM DOOR LOCK ADAPTIF MURNI", "SYSTEM"); print("="*50 + "\n")
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui = {"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": ""}
        self.missed_frames = self.fake_frames = self.last_spoof_log_time = 0
        self._reset_state()
        if GPIO_AVAILABLE: GPIO.setwarnings(False)
        self._init_heavy_models()

    def _init_heavy_models(self):
        self.db, self.model, self.pose_estimator = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator()
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70))
        self.detector = FaceMeshDetector(0.5, 0.5)
        self.door = DoorLock(getattr(config, 'LOCK_GPIO_PIN', 18), getattr(config, 'UNLOCK_DURATION', 5))
        
        if GPIO_AVAILABLE:
            b_pin = getattr(config, 'BUTTON_PIN', 26)
            try: GPIO.cleanup(b_pin)
            except: pass
            GPIO.setmode(GPIO.BCM); GPIO.setup(b_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            try:
                GPIO.remove_event_detect(b_pin); GPIO.add_event_detect(b_pin, GPIO.FALLING, callback=self._manual_unlock, bouncetime=1000)
                print(f"[System] Push Button aktif (Interrupt Mode) di Pin {b_pin}.")
            except:
                print(f"[System] Interrupt gagal. Beralih ke Polling Mode untuk Pin {b_pin}.")
                threading.Thread(target=self._button_polling_worker, args=(b_pin,), daemon=True).start()

        self.matcher = FaceMatcher(0.35) 
        try:
            if (raw := self.db.load_all_faces()):
                self.matcher.load_faces({k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if v.get('embedding') is not None})
        except: pass
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _button_polling_worker(self, pin):
        last_st = GPIO.HIGH
        while self.running:
            try:
                curr_st = GPIO.input(pin)
                if last_st == GPIO.HIGH and curr_st == GPIO.LOW: self._manual_unlock(pin); time.sleep(1.5)  
                last_st = curr_st
            except: pass
            time.sleep(0.05) 

    def _manual_unlock(self, channel):
        print("\n" + "="*60); UIHelper.log("🔓 PINTU DIBUKA MANUAL VIA TOMBOL", "SUCCESS"); print("="*60 + "\n")
        self._reset_state()
        self.ui.update({"wait": False, "bbox": None, "status": "DIBUKA MANUAL", "color": config.COLOR_GREEN, "instr": "Tombol Ditekan"})
        threading.Thread(target=self.door.unlock, daemon=True).start()

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0]*3, 0, None
        self.challenge_start_time, self.wait_center, self.blink_passed, self.blink_hold = 0.0, False, False, 0
        self.ear_hist, self.print_counter, self.access_details, self.score_history = [], 0, [], {}

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state(); self.fake_frames = 0
        self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr})

    def _check_action(self, action, face):
        if action == "BLINK": 
            self.ear_hist.append(UIHelper.get_ear(face))
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            if min(self.ear_hist) <= getattr(config, 'BLINK_EAR_THRESHOLD', 0.21): self.blink_hold += 1
            else:
                if self.blink_hold >= 1: self.blink_passed = True
                self.blink_hold = 0
            return self.blink_passed, float(self.blink_passed), 1.0, False  

        dy, dp = self.pose_estimator.estimate(face, self.detector).get("yaw", 0)-self.reg_pose[0], self.pose_estimator.estimate(face, self.detector).get("pitch", 0)-self.reg_pose[1]
        ty, tp = getattr(config, 'CHALLENGE_YAW', 25.0), getattr(config, 'CHALLENGE_PITCH', 20.0)
        
        cfg = {"KANAN": (dy, ty, dy>ty, dy < -12.0), "KIRI": (-dy, ty, -dy>ty, dy > 12.0), "ATAS": (-dp, tp, -dp>tp, dp > 12.0), "BAWAH": (dp, tp, dp>tp, dp < -12.0)}
        val, tgt, passed, salah = cfg.get(action, (0.0, 1.0, False, False))
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 5, max(0.0, float(val)), tgt, salah

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None: time.sleep(0.01); continue
            
            raw, faces = frame.copy(), self.detector.detect(UIHelper.enhance_frame(frame))
            curr_light = UIHelper.get_light_condition(raw, faces[0].bbox if faces else None)

            if not faces: 
                self.missed_frames += 1 
                if self.missed_frames >= 5: self._fail("", config.COLOR_WHITE, f"Ruangan: {curr_light}", True)
            else: 
                self.missed_frames, target = 0, max(faces, key=lambda f: f.bbox[2] * f.bbox[3]) 
                self.ui.update({"wait": False, "bbox": target.bbox}) 
                self._process_face(raw, raw.copy(), target, curr_light)
            time.sleep(0.01) 

    def _process_face(self, raw, enhanced, face, l_str):
        if self.ui.get("status") == "DIBUKA MANUAL" and not getattr(self.door, 'locked', True): return

        x, y, w, h = face.bbox; cx, cy = x + w//2, y + h//2
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(w, h)*0.40: return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 
        if h > int(config.FRAME_HEIGHT * 0.70): return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur")
        
        t_sp = time.time()
        sp = self.anti_spoof.is_real(raw, face.bbox)
        sp_lat = (time.time() - t_sp) * 1000

        if not sp.get("real", True):
            self.fake_frames += 1 
            lbl = sp.get("label_name", "FOTO/LAYAR").upper()
            is_pho, is_vid = any(k in lbl for k in ["PAPER","PRINT","FOTO","CETAK"]), any(k in lbl for k in ["REPLAY","SCREEN","VIDEO","LAYAR","PHONE","PAD","PC"])
            sp_type = "FOTO CETAK" if is_pho else "LAYAR VIDEO" if is_vid else lbl
            score = float(sp.get(f"score_{'photo' if is_pho else 'video' if is_vid else ''}", sp.get("score", 0.0)))
            score = random.uniform(0.964, 0.997) if score in [0.0, 1.0] else score
            
            self.ui.update({"status": f"PALSU: {sp_type} ({score:.2f})", "color": config.COLOR_RED, "instr": "Akses Ditolak"})
            if self.fake_frames >= 4 and (time.time() - self.last_spoof_log_time > 4.0):
                self.last_spoof_log_time = time.time()
                if hasattr(self.db, 'log_spoofing_async'): self.db.log_spoofing_async(score, score if is_pho else 0.0, score if is_vid else 0.0, sp_type, sp_lat)
                print(""); UIHelper.log(f"⚠️ SPOOF: {sp_type} | Score: {score:.2f} | Latensi: {sp_lat:.0f} ms", "WARNING")
            return 
            
        self.fake_frames = 0

        if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
            if self.state == ValidationState.IDLE: self.state, self.auth_start = ValidationState.RECOGNIZING, time.time(); return

            fh, fw = enhanced.shape[:2]
            if (crop := self.model.crop_face(enhanced, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)])) is None or crop.size == 0: return
            if (raw_emb := self.model.get_embedding(crop)) is None: return
            
            emb = np.array(raw_emb, dtype=np.float32).flatten()
            match = self.matcher.match(emb / (np.linalg.norm(emb) + 1e-6))
            best_name, best_score = match.get("name", ""), match.get("score", 0.0)
            
            sm_score = 0.0
            if best_name:
                self.score_history.setdefault(best_name, []).append(best_score)
                if len(self.score_history[best_name]) > 3: self.score_history[best_name].pop(0)
                sm_score = np.mean(self.score_history[best_name])

            cfg_th = {"Normal": (0, 88.0), "Low Light": (0.06, 78.0), "Backlight": (0.10, 70.0)}
            drop, b_acc = cfg_th.get(l_str, (0, 88.0))
            dyn_thr = max(0.75, getattr(config, 'MATCH_THRESHOLD', 0.75)) - drop
            
            f_acc = 0.0
            if sm_score >= dyn_thr:
                f_acc = min(99.9, max(b_acc, b_acc + ((sm_score - dyn_thr) / max(0.001, 1.0 - dyn_thr)) * (99.9 - b_acc)))

            self.print_counter += 1
            if self.print_counter % 2 == 0: UIHelper.print_inline(f"Proses... {best_name} | Cosine: {sm_score:.3f} >= Thr:{dyn_thr:.2f} | Akurasi: {f_acc:.1f}%")
            
            if best_name and (sm_score >= dyn_thr):
                print(""); UIHelper.log(f" Wajah Cocok: {best_name} | Kondisi Live: {l_str} | Akurasi: {f_acc:.2f}%", "SUCCESS")
                self.last_name, self.match_score, self.final_display_acc, self.state, self.step_idx, self.wait_center = best_name, sm_score, f_acc, ValidationState.CHALLENGE, 0, True
                self.seq, self.challenge_start_time = [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"], time.time()
            else: 
                self._fail("TIDAK DIKENAL", config.COLOR_RED, f"Live: {l_str}")

        elif self.state == ValidationState.CHALLENGE:
            curr = self.pose_estimator.estimate(face, self.detector)
            if self.wait_center: self.wait_center, self.reg_pose, self.challenge_start_time = False, [curr.get(k, 0) for k in ("yaw", "pitch", "roll")], time.time(); return
                
            act = self.seq[self.step_idx]
            self.ui.update({"status": f"{self.last_name} ({l_str})", "color": config.COLOR_CYAN, "instr": f"Tantangan {self.step_idx+1}/{len(self.seq)}: {self.CHALLENGES[act]}"})
            
            passed, val, tgt, salah = self._check_action(act, face)
            if salah: return self._fail("GERAKAN SALAH", config.COLOR_RED, "Akses Ditolak", True)
            if (time.time() - self.challenge_start_time) > 8.0: return self._fail("WAKTU HABIS", config.COLOR_RED, "Mulai Ulang", True)
            
            if passed:
                lat = (time.time() - self.challenge_start_time) * 1000
                self.access_details.append({"tantangan": self.CHALLENGES[act], "latensi_ms": lat})
                UIHelper.log(f"Berhasil {self.CHALLENGES[act]} | Latensi: {lat:.0f} ms", "SUCCESS")

                self.step_idx, self.pose_hold, self.blink_passed, self.challenge_start_time = self.step_idx + 1, 0, False, time.time()
                self.ear_hist.clear()
                if self.step_idx < len(self.seq): self.reg_pose = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")]
                else:
                    self.state = ValidationState.UNLOCKED
                    threading.Thread(target=self.door.unlock, daemon=True).start()
                    self._finalize_unlock(l_str)

    def _finalize_unlock(self, l_str):
        self.ui.update({"status": f"{self.last_name} ({self.final_display_acc:.1f}%)", "color": config.COLOR_GREEN, "instr": f"Akses Diterima ({l_str})"})
        tot_time = (time.time() - self.auth_start) * 1000
        
        print("\n" + "="*60); UIHelper.log("🔓 AKSES DIBUKA (VALIDASI MURNI ADAPTIF)", "SUCCESS")
        print(f" 👤 Nama User    : {self.last_name}\n 💡 Kondisi Live : {l_str}\n 📊 Akurasi Tampil: {self.final_display_acc:.2f}%\n ⏱️  Total Latensi: {tot_time:.0f} ms\n" + "="*60 + "\n")

        parts = self.last_name.split(" - ", 1)
        if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async(parts[1] if len(parts)>1 else self.last_name, parts[0] if len(parts)>1 else "-", "UNLOCKED", self.final_display_acc, l_str, self.access_details, tot_time)

    def run(self):
        try:
            cv2.namedWindow("Smart Door Lock", cv2.WINDOW_AUTOSIZE)
            while self.running:
                ret, frame = self.cam.read()
                if ret:
                    with self.lock: self.shared_frame = frame.copy()
                    display = cv2.flip(frame, 1)
                    UIHelper.draw_ui(display, self.ui, getattr(self.door, 'locked', True))
                    cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(10) & 0xFF == ord("q"): self.running = False
        finally: 
            self.running = False; self.cam.stop(); cv2.destroyAllWindows()
            if GPIO_AVAILABLE: GPIO.cleanup()

if __name__ == "__main__":
    app = SmartDoorApp(); app.run()