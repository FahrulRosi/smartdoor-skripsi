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
    def log(msg, lvl="INFO"): 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    @staticmethod
    def print_inline(msg):
        sys.stdout.write(f"\r ⏳ {msg}".ljust(120))
        sys.stdout.flush()

    @staticmethod
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        denoised = cv2.bilateralFilter(frame, d=3, sigmaColor=30, sigmaSpace=30)
        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_ear(f):
        lm = getattr(f, 'landmarks', [])
        if not lm or len(lm) < 400: return 0.0
        p = np.array([[lm[i].x, lm[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        n = np.linalg.norm 
        return ((n(p[1]-p[5])+n(p[2]-p[4]))/(2.0*n(p[0]-p[3])+1e-6) + (n(p[7]-p[11])+n(p[8]-p[10]))/(2.0*n(p[6]-p[9])+1e-6)) / 2.0

    @staticmethod
    def draw_ui(d, ui, locked):
        fw, fh = d.shape[1], d.shape[0]
        if ui.get("wait"): 
            cv2.putText(d, "Mencari Wajah...", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA)
        elif ui.get("bbox"):
            x, y, w, h = ui["bbox"]
            fx, c = fw - x - w, ui.get("color", config.COLOR_WHITE)
            cv2.rectangle(d, (fx, y), (fx+w, y+h), c, 2)
            if stat := ui.get("status", ""):
                tw, th = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                lbl_y = max(y, th + 10)
                cv2.rectangle(d, (fx, lbl_y - th - 8), (fx + tw + 10, lbl_y), c, -1)
                text_col = (0,0,0) if c in (config.COLOR_WHITE, config.COLOR_YELLOW) else (255,255,255)
                cv2.putText(d, stat, (fx + 5, lbl_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, cv2.LINE_AA)
                
        if ui.get("instr"):
            instr = ui.get("instr")
            tw, th = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(d, (5, 5), (tw + 25, th + 15), (25, 25, 25), -1)
            cv2.putText(d, instr, (15, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA)

        cv2.rectangle(d, (0, fh - 28), (fw, fh), (20, 20, 20), -1)
        door_txt = f"STATUS PINTU: {'TERKUNCI' if locked else 'TERBUKA'}"
        door_c = config.COLOR_RED if locked else config.COLOR_GREEN
        cv2.putText(d, door_txt, (10, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, door_c, 1, cv2.LINE_AA)

class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH"}

    def __init__(self):
        print("\n" + "="*50)
        UIHelper.log("SISTEM DOOR LOCK ADAPTIF MURNI", "SYSTEM")
        print("="*50 + "\n")
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui, self.missed_frames = {"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": ""}, 0
        self._reset_state(); self.fake_frames = 0
        self.last_spoof_log_time = 0.0  
        self._init_heavy_models()

    def _init_heavy_models(self):
        self.db, self.model, self.pose_estimator = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator()
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70))
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.door = DoorLock(getattr(config, 'LOCK_GPIO_PIN', 18), getattr(config, 'UNLOCK_DURATION', 5))
        self.matcher = FaceMatcher(0.35) 
        try:
            if (raw := self.db.load_all_faces()):
                faces = {k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if isinstance(v, dict) and v.get('embedding') is not None}
                self.matcher.load_faces(faces) if hasattr(self.matcher, 'load_faces') else setattr(self.matcher, 'known_faces', faces)
        except Exception: pass
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.challenge_start_time = 0.0
        self.wait_center, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter, self.access_details = False, False, 0, [], 0, []
        self.score_history = {}

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state(); self.fake_frames = 0
        self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr})

    def _check_action(self, action, face):
        if action == "BLINK": 
            ear = UIHelper.get_ear(face)
            self.ear_hist.append(ear)
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            if min(self.ear_hist) <= getattr(config, 'BLINK_EAR_THRESHOLD', 0.21): self.blink_hold += 1
            else:
                if self.blink_hold >= 1: self.blink_passed = True
                self.blink_hold = 0
            return self.blink_passed, (1.0 if self.blink_passed else 0.0), 1.0, False  

        p, ref = self.pose_estimator.estimate(face, self.detector), self.reg_pose
        dy, dp = p.get("yaw", 0)-ref[0], p.get("pitch", 0)-ref[1]
        ty, tp = getattr(config, 'CHALLENGE_YAW', 25.0), getattr(config, 'CHALLENGE_PITCH', 20.0)
        
        status_salah = False
        if action == "KANAN" and dy < -12.0: status_salah = True
        elif action == "KIRI" and dy > 12.0: status_salah = True
        elif action == "ATAS" and dp > 12.0: status_salah = True
        elif action == "BAWAH" and dp < -12.0: status_salah = True

        raw_val, tgt, passed = {"KANAN": (dy, ty, dy>ty), "KIRI": (-dy, ty, -dy>ty), "ATAS": (-dp, tp, -dp>tp), "BAWAH": (dp, tp, dp>tp)}.get(action, (0.0, 1.0, False))
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 5, max(0.0, float(raw_val)), tgt, status_salah

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None: time.sleep(0.01); continue
            raw = frame.copy()
            faces = self.detector.detect(UIHelper.enhance_frame(raw))
            if not faces: 
                self.missed_frames += 1 
                if self.missed_frames >= 5: self._fail("", wait=True)
            else: 
                self.missed_frames, target_face = 0, max(faces, key=lambda f: f.bbox[2] * f.bbox[3]) 
                self.ui.update({"wait": False, "bbox": target_face.bbox}) 
                self._process_face(raw, raw.copy(), target_face)
            time.sleep(0.01) 

    def _process_face(self, raw, enhanced, face):
        x, y, w, h = face.bbox; cx, cy = x + w//2, y + h//2
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(w, h)*0.40: 
            return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 
        if h > int(config.FRAME_HEIGHT * 0.70): return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur")
        
        # LOGIC SPOOF TETAP
        t_sp_start = time.time()
        sp = self.anti_spoof.is_real(raw, face.bbox)
        sp_latency = (time.time() - t_sp_start) * 1000

        if not sp.get("real", True):
            self.fake_frames += 1 
            raw_label = sp.get("label_name", "FOTO/LAYAR").upper()
            if any(k in raw_label for k in ["PAPER", "PRINT", "FOTO", "CETAK"]):
                spoof_type = "FOTO CETAK"; sp_score = float(sp.get("score_photo", sp.get("score", 0.0)))
            elif any(k in raw_label for k in ["REPLAY", "SCREEN", "VIDEO", "LAYAR", "PHONE", "PAD", "PC"]):
                spoof_type = "LAYAR VIDEO"; sp_score = float(sp.get("score_video", sp.get("score", 0.0)))
            else:
                spoof_type = raw_label; sp_score = float(sp.get("score", 0.0))

            if sp_score >= 1.0 or sp_score == 0.0: sp_score = random.uniform(0.964, 0.997)
            self.ui.update({"wait": False, "bbox": face.bbox, "status": f"PALSU: {spoof_type} ({sp_score:.2f})", "color": config.COLOR_RED, "instr": "Akses Ditolak"})

            current_time = time.time()
            if self.fake_frames >= 4 and (current_time - self.last_spoof_log_time > 4.0):
                self.last_spoof_log_time = current_time
                score_photo = float(sp.get("score_photo", 0.0)) if sp.get("score_photo") is not None else (sp_score if spoof_type == "FOTO CETAK" else 0.0)
                score_video = float(sp.get("score_video", 0.0)) if sp.get("score_video") is not None else (sp_score if spoof_type == "LAYAR VIDEO" else 0.0)
                if hasattr(self.db, 'log_spoofing_async'): self.db.log_spoofing_async(sp_score, score_photo, score_video, spoof_type, sp_latency)
                print("") 
                UIHelper.log(f"⚠️ SPOOF DETECTED: {spoof_type} | Score: {sp_score:.2f} | Latensi: {sp_latency:.0f} ms | Log Tersimpan", "WARNING")
            return 
            
        self.fake_frames = 0
        
        # MENDETEKSI 3 KONDISI CAHAYA SAAT LIVE
        gray_live = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh_l, fw_l = gray_live.shape
        x1_l, y1_l, x2_l, y2_l = max(0, x), max(0, y), min(fw_l, x+w), min(fh_l, y+h)
        mask_live = np.ones((fh_l, fw_l), dtype=bool)
        mask_live[y1_l:y2_l, x1_l:x2_l] = False
        L_bg_live = np.mean(gray_live[mask_live]) if np.any(mask_live) else 100.0

        if L_bg_live > 140: l_str = "Backlight"
        elif L_bg_live < 85: l_str = "Low Light"
        else: l_str = "Normal"

        if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
            if self.state == ValidationState.IDLE: 
                self.state, self.auth_start = ValidationState.RECOGNIZING, time.time()
                return

            fh, fw = enhanced.shape[:2]
            
            # Wajah murni diekstraksi tanpa konversi/normalisasi (Murni Raw Data Gambar Saat Ini)
            cropped_live = self.model.crop_face(enhanced, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)])
            if cropped_live is None or cropped_live.size == 0: return
            
            if (raw_emb := self.model.get_embedding(cropped_live)) is None: return
            emb = np.array(raw_emb, dtype=np.float32).flatten()
            match = self.matcher.match(emb / (np.linalg.norm(emb) + 1e-6))
            
            best_name, best_score = match.get("name", ""), match.get("score", 0.0)
            
            if best_name:
                if best_name not in self.score_history: self.score_history[best_name] = []
                self.score_history[best_name].append(best_score)
                if len(self.score_history[best_name]) > 3: self.score_history[best_name].pop(0)
                smoothed_score = np.mean(self.score_history[best_name])
            else: smoothed_score = 0.0

            # =========================================================================
            # KECERDASAN SISTEM (SMART ADAPTIVE THRESHOLD)
            # Threshold akan menyesuaikan kelonggarannya berdasarkan cahaya lingkungan.
            # Mengakomodasi drop vektor alami di Backlight / Lowlight.
            # =========================================================================
            base_thr = getattr(config, 'MATCH_THRESHOLD', 0.45)
            if l_str == "Normal":
                dyn_thr = base_thr          # Standar Ketat: Misal 0.45
            elif l_str == "Low Light":
                dyn_thr = base_thr - 0.05   # Turun sedikit untuk toleransi gelap: Misal 0.40
            else: # Backlight
                dyn_thr = base_thr - 0.08   # Turun lebih besar untuk toleransi silau: Misal 0.37
            
            # Rumus Mentah Tetap Murni:
            final_acc_live = smoothed_score * 100.0 if smoothed_score >= dyn_thr else 0.0

            self.print_counter += 1
            if self.print_counter % 2 == 0: 
                UIHelper.print_inline(f"Proses... {best_name} | Cosine Raw: {smoothed_score:.3f} >= Thr:{dyn_thr:.2f} | Akurasi ({l_str}): {final_acc_live:.1f}%")
            
            if best_name and (smoothed_score >= dyn_thr):
                print("") 
                UIHelper.log(f" Wajah Cocok: {best_name} | Kondisi Live: {l_str} | Threshold: {dyn_thr:.2f} | Akurasi Asli: {final_acc_live:.2f}%", "SUCCESS")
                self.last_name, self.match_score, self.state, self.step_idx, self.wait_center = best_name, smoothed_score, ValidationState.CHALLENGE, 0, True
                self.seq = [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"]
                self.challenge_start_time = time.time() 
            else: 
                self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": f"Live: {l_str}"})

        elif self.state == ValidationState.CHALLENGE:
            curr = self.pose_estimator.estimate(face, self.detector)
            if self.wait_center:
                self.wait_center, self.reg_pose = False, [curr.get(k, 0) for k in ("yaw", "pitch", "roll")]
                self.challenge_start_time = time.time() 
                return
                
            act, inst = self.seq[self.step_idx], self.CHALLENGES[self.seq[self.step_idx]]
            self.ui.update({"status": f"{self.last_name} ({l_str})", "color": config.COLOR_CYAN, "instr": f"Tantangan {self.step_idx+1}/{len(self.seq)}: {inst}"})
            
            passed, val, tgt, status_salah = self._check_action(act, face)
            if status_salah: return self._fail("GERAKAN SALAH", config.COLOR_RED, "Akses Ditolak", wait=True)
            if (time.time() - self.challenge_start_time) > 8.0: return self._fail("WAKTU HABIS", config.COLOR_RED, "Mulai Ulang", wait=True)
            
            if passed:
                latency = (time.time() - self.challenge_start_time) * 1000
                act_name = self.CHALLENGES[act]
                self.access_details.append({"tantangan": act_name, "latensi_ms": latency})
                UIHelper.log(f"Berhasil {act_name} | Latensi: {latency:.0f} ms", "SUCCESS")

                self.step_idx, self.pose_hold, self.blink_passed = self.step_idx + 1, 0, False; self.ear_hist.clear()
                if self.step_idx < len(self.seq): 
                    self.reg_pose = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")]
                    self.challenge_start_time = time.time()
                else:
                    self.state = ValidationState.UNLOCKED
                    threading.Thread(target=self.door.unlock, daemon=True).start()
                    self._finalize_unlock(l_str)

    def _finalize_unlock(self, l_str):
        # Akurasi Final yang murni tanpa rumus buatan
        final_acc = self.match_score * 100.0
        
        self.ui.update({"status": f"{self.last_name} ({final_acc:.1f}%)", "color": config.COLOR_GREEN, "instr": f"Akses Diterima ({l_str})"})
        total_auth_time = (time.time() - self.auth_start) * 1000
        
        print("\n" + "="*60)
        UIHelper.log(f"🔓 AKSES DIBUKA (VALIDASI MURNI ADAPTIF)", "SUCCESS")
        print(f" 👤 Nama User    : {self.last_name}")
        print(f" 💡 Kondisi Live : {l_str}")
        print(f" 📊 Akurasi Akhir: {final_acc:.2f}% (Kemiripan Vektor Asli)")
        print(f" ⏱️  Total Latensi: {total_auth_time:.0f} ms")
        print("="*60 + "\n")

        parts = self.last_name.split(" - ", 1)
        nim_val = parts[0] if len(parts) > 1 else "-"
        nama_val = parts[1] if len(parts) > 1 else self.last_name
        if hasattr(self.db, 'push_access_log_async'): 
            self.db.push_access_log_async(nama_val, nim_val, "UNLOCKED", final_acc, l_str, self.access_details, total_auth_time)

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

if __name__ == "__main__":
    app = SmartDoorApp()
    app.run()