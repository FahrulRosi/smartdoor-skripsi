import sys
import cv2
import numpy as np
import random
from datetime import datetime
from enum import Enum

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from door.door_lock             import DoorLock
from liveness.head_pose         import HeadPoseEstimator
from liveness.blink             import BlinkDetector
from liveness.anti_spoofing     import SilentAntiSpoofing  

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

class ValidationState(Enum):
    IDLE        = 0
    RECOGNIZING = 1
    CHALLENGE   = 2
    UNMATCHED   = 3
    UNLOCKED    = 4

class UIManager:
    @staticmethod
    def print_log(msg, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {msg}")

    @staticmethod
    def put_text(frame, text, y, color=config.COLOR_WHITE, x=10, scale=0.7, thickness=2):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_status(frame, bbox, status, color):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        t_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.rectangle(frame, (x, y - 35), (x + t_size[0] + 15, y - 5), color, -1)
        UIManager.put_text(frame, status, y - 12, (255, 255, 255), x + 8, scale=0.65, thickness=2)

    @staticmethod
    def draw_challenge_info(frame, step_idx, total_steps, instruction):
        box_w, box_h = 450, 70
        x = (config.FRAME_WIDTH - box_w) // 2
        y = 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), config.COLOR_CYAN, 2)
        UIManager.put_text(frame, f"Tahap Liveness ({step_idx + 1}/{total_steps})", y + 25, config.COLOR_YELLOW, x + 15, scale=0.6)
        UIManager.put_text(frame, instruction, y + 55, config.COLOR_WHITE, x + 15, scale=0.75, thickness=2)

    @staticmethod
    def draw_door_status(frame, is_locked):
        status_text = "TERKUNCI" if is_locked else "TERBUKA"
        color = config.COLOR_RED if is_locked else config.COLOR_GREEN
        UIManager.put_text(frame, f"PINTU: {status_text}", config.FRAME_HEIGHT - 30, color, scale=0.75)

class SmartDoorApp:
    POSE_DIRECTIONS = ["KANAN", "KIRI", "ATAS", "BAWAH", "MIRING_KANAN", "MIRING_KIRI"]
    INSTRUCTION_TEXT = {
        "BLINK": "Tantangan 2: Kedipkan Mata Anda",
        "KANAN": "Tantangan 1: Toleh Kepala ke KANAN",
        "KIRI":  "Tantangan 1: Toleh Kepala ke KIRI",
        "ATAS":  "Tantangan 1: Dongak Kepala ke ATAS",
        "BAWAH": "Tantangan 1: Tunduk Kepala ke BAWAH",
        "MIRING_KANAN": "Tantangan 1: Miringkan Kepala ke KANAN",
        "MIRING_KIRI":  "Tantangan 1: Miringkan Kepala ke KIRI"
    }

    def __init__(self):
        UIManager.print_log("Sistem Smart Door Lock diaktifkan", "SYSTEM")
        UIManager.print_log("Menghubungkan ke Firebase Database...", "SYSTEM")
        
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, apply_enhancement=getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', False)).start()
        self.detector = FaceMeshDetector(min_detection_confidence=config.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE)
        self.model = MobileFaceNet()
        self.matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
        self.pose_estimator = HeadPoseEstimator()
        self.door = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=5)
        self.anti_spoof = SilentAntiSpoofing()
        
        # --- MEMUAT DATABASE WAJAH & LIVENESS CONFIG DARI FIREBASE ---
        from database.face_db import FaceDatabase
        db_url = getattr(config, "FIREBASE_URL", "")
        credentials_path = getattr(config, "FIREBASE_CREDENTIALS", "serviceAccount.json")
        self.face_db = FaceDatabase(db_url, credentials_path)
        self.user_profiles = {} # Untuk menyimpan liveness config (headpose & blink) masing-masing user

        try:
            # Menggunakan referensi langsung untuk menarik SEMUA data dari firebase
            full_db_data = self.face_db.ref_users.get()
            if full_db_data:
                processed_faces = {}
                for user_name, user_data in full_db_data.items():
                    if isinstance(user_data, dict):
                        # Ekstrak Embedding
                        if 'embedding' in user_data:
                            processed_faces[user_name] = np.array(user_data['embedding'], dtype=np.float32)
                        
                        # Ekstrak & Simpan Profil Liveness User (Headpose & Blink dari Database)
                        self.user_profiles[user_name] = user_data.get('liveness_config', {})
                
                # Masukkan data ke FaceMatcher
                if hasattr(self.matcher, 'load_faces'):
                    self.matcher.load_faces(processed_faces)
                else:
                    self.matcher.known_faces = processed_faces
                    
                UIManager.print_log(f"Berhasil memuat {len(processed_faces)} profil wajah beserta profil Liveness-nya.", "SUCCESS")
            else:
                UIManager.print_log("Database wajah kosong.", "WARNING")
        except Exception as e:
            UIManager.print_log(f"Gagal memuat data dari Firebase: {e}", "ERROR")

        self._reset_state()

    def _reset_state(self):
        if getattr(self, 'state', None) != ValidationState.UNLOCKED:
            self.state = ValidationState.IDLE
            self.last_name = ""
        
        self.door.lock()
        self.challenge_sequence = []
        self.current_step_idx = 0
        self.blink_checker = None
        self.pose_hold_frames = 0
        self.reg_headpose = [0.0, 0.0, 0.0]

    def _check_action_passed(self, action, face):
        if action == "BLINK":
            if self.blink_checker:
                res = self.blink_checker.update(face, self.detector)
                return res.get("complete", False)
            return False

        # --- VALIDASI POSE KEPALA BERDASARKAN DATA FIREBASE ---
        pose = self.pose_estimator.estimate(face, self.detector)
        yaw, pitch, roll = pose.get("yaw", 0), pose.get("pitch", 0), pose.get("roll", 0)

        # Mengambil pose netral user dari database
        db_yaw, db_pitch, db_roll = self.reg_headpose[0], self.reg_headpose[1], self.reg_headpose[2]

        # Menghitung PERGERAKAN RELATIF (Posisi saat ini dikurangi posisi netral di database)
        rel_yaw = yaw - db_yaw
        rel_pitch = pitch - db_pitch
        rel_roll = roll - db_roll

        pose_thresholds = {
            "KANAN": rel_yaw > config.CHALLENGE_YAW,
            "KIRI": rel_yaw < -config.CHALLENGE_YAW,
            "ATAS": rel_pitch < -config.CHALLENGE_PITCH,
            "BAWAH": rel_pitch > config.CHALLENGE_PITCH,
            "MIRING_KANAN": rel_roll > config.CHALLENGE_ROLL,
            "MIRING_KIRI": rel_roll < -config.CHALLENGE_ROLL,
        }
        
        # Penahan Pose: User harus menahan pose yang benar selama minimal 5 frame
        if pose_thresholds.get(action, False):
            self.pose_hold_frames += 1
            if self.pose_hold_frames >= 5: 
                self.pose_hold_frames = 0
                return True
        else:
            self.pose_hold_frames = 0
            
        return False

    def _handle_recognizing(self, frame, face):
        face_crop = self.model.crop_face(frame, face.bbox)
        embedding = self.model.get_embedding(face_crop)
        match = self.matcher.match(embedding)
        skor = match.get("score", 0.0)

        if match.get("matched", False):
            self.last_name = match["name"]
            
            # --- MENYIAPKAN TANTANGAN DARI DATABASE ---
            user_config = self.user_profiles.get(self.last_name, {})
            
            # 1. Tarik Headpose Netral dari Database
            self.reg_headpose = user_config.get("headpose_vector", [0.0, 0.0, 0.0])
            
            # 2. Tarik EAR Mata Tertutup dari Database
            personal_blink_ear = user_config.get("blink_closed", config.BLINK_EAR_THRESHOLD)
            if personal_blink_ear == 0.0:
                personal_blink_ear = config.BLINK_EAR_THRESHOLD
                
            # Kita tambah margin 0.01 agar user tidak perlu memicingkan mata terlalu ekstrem
            target_ear = float(personal_blink_ear) + 0.01 

            self.state = ValidationState.CHALLENGE
            # Selalu mengatur Challenge 1 = Pose Acak, Challenge 2 = Blink
            self.challenge_sequence = [random.choice(self.POSE_DIRECTIONS), "BLINK"]
            self.current_step_idx = 0
            
            # Inisialisasi Blink Detector menggunakan threshold personal dari Firebase
            self.blink_checker = BlinkDetector(ear_threshold=target_ear, target_blinks=1)
            
            UIManager.print_log(f"Wajah {self.last_name} dikenali. | DB Headpose: {self.reg_headpose}", "SUCCESS")
        else:
            self.last_name = "Tidak Dikenali"
            self.state = ValidationState.UNMATCHED
            UIManager.print_log(f"Ditolak! Skor kemiripan: {skor:.3f} (Butuh: {config.MATCH_THRESHOLD})", "WARNING")

    def _handle_challenge(self, display, face):
        current_action = self.challenge_sequence[self.current_step_idx]
        instruction = self.INSTRUCTION_TEXT.get(current_action, "Ikuti instruksi...")

        UIManager.draw_status(display, face.bbox, f"User: {self.last_name}", config.COLOR_CYAN)
        UIManager.draw_challenge_info(display, self.current_step_idx, len(self.challenge_sequence), instruction)

        if self._check_action_passed(current_action, face):
            UIManager.print_log(f"{current_action} challenge PASSED", "SUCCESS")
            self.current_step_idx += 1
            
            if self.current_step_idx >= len(self.challenge_sequence):
                self.state = ValidationState.UNLOCKED
                self.door.unlock()
                UIManager.print_log(f"CHALLENGE LENGKAP → Pintu terbuka untuk {self.last_name}", "SUCCESS")

    def run(self):
        self.cam.start()
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    continue

                display = frame.copy()
                faces = self.detector.detect(frame)

                enhancement_status = "ON" if getattr(self.cam, 'apply_enhancement', False) else "OFF"
                color_enh = config.COLOR_GREEN if enhancement_status == "ON" else config.COLOR_RED
                UIManager.put_text(display, f"CLAHE (Low-Light): {enhancement_status}", 30, color_enh, scale=0.6)

                if not faces:
                    self._reset_state()
                    UIManager.put_text(display, "Menunggu Wajah...", 60, config.COLOR_YELLOW, scale=0.9)
                else:
                    face = faces[0]
                    spoof = self.anti_spoof.is_real(frame, face.bbox)

                    if not spoof.get("real", True):
                        self._reset_state()
                        UIManager.draw_status(display, face.bbox, "WAJAH PALSU / FOTO HP!", config.COLOR_RED)
                        UIManager.put_text(display, f"Skor Liveness: {spoof.get('score', 0):.3f}", face.bbox[1] - 40, config.COLOR_RED)
                    else:
                        if self.state == ValidationState.IDLE:
                            self.state = ValidationState.RECOGNIZING

                        if self.state == ValidationState.RECOGNIZING:
                            self._handle_recognizing(frame, face)

                        elif self.state == ValidationState.CHALLENGE:
                            self._handle_challenge(display, face)

                        elif self.state == ValidationState.UNLOCKED:
                            UIManager.draw_status(display, face.bbox, f"SELAMAT DATANG, {self.last_name}", config.COLOR_GREEN)

                        elif self.state == ValidationState.UNMATCHED:
                            UIManager.draw_status(display, face.bbox, "WAJAH TIDAK DIKENALI", config.COLOR_RED)

                UIManager.draw_door_status(display, self.door.locked)

                cv2.imshow("Smart Door Lock", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("e"):
                    if hasattr(self.cam, 'apply_enhancement'):
                        self.cam.apply_enhancement = not self.cam.apply_enhancement
                        UIManager.print_log(f"CLAHE Enhancement diset ke: {self.cam.apply_enhancement}", "SYSTEM")

        except Exception as e:
            UIManager.print_log(f"Error utama: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        finally:
            self.cam.stop()
            self.door.cleanup()
            cv2.destroyAllWindows()
            UIManager.print_log("Sistem Smart Door ditutup.", "SYSTEM")

def run_unlock():
    app = SmartDoorApp()
    app.run()

if __name__ == "__main__":
    run_unlock()