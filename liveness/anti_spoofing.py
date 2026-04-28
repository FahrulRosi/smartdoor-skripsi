import cv2
import numpy as np
import os
import random
from liveness.head_pose import HeadPoseEstimator
from liveness.blink import BlinkDetector
import config

class SilentAntiSpoofing:
    def __init__(self, model_path="liveness/antispoofing.onnx", threshold=0.85):
        self.threshold = threshold
        self._session = None
        self._input_name = ""
        self.scale = 2.7 
        
        if os.path.isfile(model_path):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                print(f"[AntiSpoofing] Model ONNX berhasil dimuat: {model_path}")
            except Exception as e:
                print(f"[AntiSpoofing] Gagal memuat model: {e}")
        else:
            print(f"[AntiSpoofing] Model tidak ditemukan di '{model_path}'")

    def _get_new_box(self, src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2 + x
        center_y = box_h / 2 + y

        left_top_x = max(0, center_x - new_width / 2)
        left_top_y = max(0, center_y - new_height / 2)
        right_bottom_x = min(src_w - 1, center_x + new_width / 2)
        right_bottom_y = min(src_h - 1, center_y + new_height / 2)

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def is_real(self, frame, bbox):
        if not self._session:
            return {"real": True, "score": 1.0}

        src_h, src_w = frame.shape[:2]
        x1, y1, x2, y2 = self._get_new_box(src_w, src_h, bbox, self.scale)
        face_crop = frame[y1:y2+1, x1:x2+1]

        if face_crop.size == 0:
            return {"real": False, "score": 0.0}

        face_resized = cv2.resize(face_crop, (80, 80))
        face_float = face_resized.astype(np.float32)
        face_chw = np.transpose(face_float, (2, 0, 1))
        img_data = np.expand_dims(face_chw, axis=0)

        output = self._session.run(None, {self._input_name: img_data})[0]
        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
        softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        preds = softmax_output[0]

        score_real = float(preds[1])
        is_valid = score_real >= self.threshold
        return {"real": is_valid, "score": round(score_real, 4)}


class ActiveChallengeManager:
    """Manajer tantangan acak yang mengambil logika liveness dari registrasi."""
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self.blink_detector = None
        self.current_challenge = None
        self.passed = False
        self._step_frame_count = 0
        self._required_frames = 3 # Harus menahan gerakan selama 3 frame

    def generate_challenge(self):
        """Pilih satu tantangan acak untuk divalidasi."""
        challenges = ["YAW", "PITCH", "ROLL", "BLINK"]
        self.current_challenge = random.choice(challenges)
        self.passed = False
        self._step_frame_count = 0
        
        if self.current_challenge == "BLINK":
            self.blink_detector = BlinkDetector(target_blinks=1) 
            
        return self.current_challenge

    def verify_challenge(self, face, detector):
        """Memverifikasi tantangan gerakan/kedipan berdasarkan posisi wajah (Head Pose Estimator)."""
        if self.passed:
            return True, "Liveness Berhasil!"

        if self.current_challenge == "BLINK":
            res = self.blink_detector.update(face, detector)
            if res["complete"]:
                self.passed = True
                return True, "Kedipan Terdeteksi!"
            return False, "CHALLENGE: Silakan Berkedip"

        pose = self.pose_estimator.estimate(face, detector)
        if not pose["valid"]:
            return False, "Arahkan wajah lurus ke kamera"

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

        if self.current_challenge == "YAW":
            if abs(yaw) > config.CHALLENGE_YAW:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self.passed = True
                return True, "Gerakan Kanan/Kiri Berhasil!"
            return False, "CHALLENGE: Tolehkan Kepala ke Kanan/Kiri"

        elif self.current_challenge == "PITCH":
            if abs(pitch) > config.CHALLENGE_PITCH:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self.passed = True
                return True, "Angkat/Tunduk Berhasil!"
            return False, "CHALLENGE: Angkat atau Tundukkan Kepala"

        elif self.current_challenge == "ROLL":
            if abs(roll) > config.CHALLENGE_ROLL:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self.passed = True
                return True, "Miringkan Kepala Berhasil!"
            return False, "CHALLENGE: Miringkan Kepala Anda"

        return False, "Menunggu tantangan..."