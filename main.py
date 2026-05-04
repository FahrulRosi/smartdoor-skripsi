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
    """Kelas khusus untuk menangani semua antarmuka visual (HUD) pada frame."""
    
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
    """Kelas utama yang mengatur logika dan state mesin dari Smart Door Lock."""
    
    POSE_DIRECTIONS = ["KANAN", "KIRI", "ATAS", "BAWAH", "MIRING_KANAN", "MIRING_KIRI"]
    INSTRUCTION_TEXT = {
        "BLINK": "Tantangan: Kedipkan Mata Anda",
        "KANAN": "Tantangan: Toleh Kepala ke KANAN",
        "KIRI":  "Tantangan: Toleh Kepala ke KIRI",
        "ATAS":  "Tantangan: Dongak Kepala ke ATAS",
        "BAWAH": "Tantangan: Tunduk Kepala ke BAWAH",
        "MIRING_KANAN": "Tantangan: Miringkan Kepala ke KANAN",
        "MIRING_KIRI":  "Tantangan: Miringkan Kepala ke KIRI"
    }

    def __init__(self):
        UIManager.print_log("Sistem Smart Door Lock diaktifkan", "SYSTEM")
        UIManager.print_log("Menghubungkan ke Firebase Database...", "SYSTEM")
        
        # --- BARU: Tambahkan apply_enhancement=True pada pemanggilan CameraStream ---
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, apply_enhancement=True)
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.model = MobileFaceNet()
        self.matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
        self.pose_estimator = HeadPoseEstimator()
        self.door = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=5)
        self.anti_spoof = SilentAntiSpoofing()
        
        self._reset_state()

    def _reset_state(self):
        """Mengembalikan sistem ke kondisi awal (IDLE)."""
        if getattr(self, 'state', None) != ValidationState.UNLOCKED:
            self.state = ValidationState.IDLE
            self.last_name = ""
        
        self.door.lock()
        self.challenge_sequence = []
        self.current_step_idx = 0
        self.blink_checker = None

    def _check_action_passed(self, action, face):
        """Memvalidasi apakah gerakan challenge saat ini berhasil dilakukan."""
        if action == "BLINK":
            if self.blink_checker:
                res = self.blink_checker.update(face, self.detector)
                return res.get("complete", False)
            return False

        # Validasi Pose
        pose = self.pose_estimator.estimate(face, self.detector)
        yaw, pitch, roll = pose.get("yaw", 0), pose.get("pitch", 0), pose.get("roll", 0)

        pose_thresholds = {
            "KANAN": yaw > config.CHALLENGE_YAW,
            "KIRI": yaw < -config.CHALLENGE_YAW,
            "ATAS": pitch < -config.CHALLENGE_PITCH,
            "BAWAH": pitch > config.CHALLENGE_PITCH,
            "MIRING_KANAN": roll > config.CHALLENGE_ROLL,
            "MIRING_KIRI": roll < -config.CHALLENGE_ROLL,
        }
        return pose_thresholds.get(action, False)

    def _handle_recognizing(self, frame, face):
        """Proses pengenalan wajah dari database."""
        face_crop = self.model.crop_face(frame, face.bbox)
        embedding = self.model.get_embedding(face_crop)
        match = self.matcher.match(embedding)
        skor = match.get("score", 0.0)

        if match.get("matched", False):
            self.last_name = match["name"]
            self.state = ValidationState.CHALLENGE
            self.challenge_sequence = [random.choice(self.POSE_DIRECTIONS), "BLINK"]
            self.current_step_idx = 0
            self.blink_checker = BlinkDetector(target_blinks=1)
            
            UIManager.print_log(f"Wajah dikenali: {self.last_name} | Skor: {skor:.3f} | Challenge: {self.challenge_sequence}", "SUCCESS")
        else:
            self.last_name = "Tidak Dikenali"
            self.state = ValidationState.UNMATCHED
            UIManager.print_log(f"Ditolak! Skor kemiripan: {skor:.3f} (Butuh: {config.MATCH_THRESHOLD})", "WARNING")

    def _handle_challenge(self, display, face):
        """Mengelola proses verifikasi liveness (gerakan)."""
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
                UIManager.print_log(f"CHALLENGE BERHASIL → Pintu terbuka untuk {self.last_name}", "SUCCESS")

    def run(self):
        """Loop utama aplikasi."""
        self.cam.start()
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    continue

                display = frame.copy()
                faces = self.detector.detect(frame)

                # --- BARU: Menampilkan status CLAHE di layar ---
                enhancement_status = "ON" if getattr(self.cam, 'apply_enhancement', False) else "OFF"
                color_enh = config.COLOR_GREEN if enhancement_status == "ON" else config.COLOR_RED
                UIManager.put_text(display, f"CLAHE (Low-Light): {enhancement_status}", 30, color_enh, scale=0.6)

                if not faces:
                    self._reset_state()
                    UIManager.put_text(display, "Menunggu Wajah...", 60, config.COLOR_YELLOW, scale=0.9)
                else:
                    face = faces[0]
                    spoof = self.anti_spoof.is_real(frame, face.bbox)

                    # 1. Verifikasi Anti-Spoofing Dasar
                    if not spoof.get("real", True):
                        self._reset_state()
                        UIManager.draw_status(display, face.bbox, "WAJAH PALSU / FOTO HP!", config.COLOR_RED)
                        UIManager.put_text(display, f"Skor Liveness: {spoof.get('score', 0):.3f}", face.bbox[1] - 40, config.COLOR_RED)
                    
                    # 2. Proses State Machine Wajah
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

                # Update status pintu di UI
                UIManager.draw_door_status(display, self.door.locked)

                cv2.imshow("Smart Door Lock", display)
                
                # --- BARU: Tambahkan deteksi tombol 'e' untuk toggle Enhancement ---
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