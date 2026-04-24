import cv2
import numpy as np
import os

class SilentAntiSpoofing:
    def __init__(self, model_path="liveness/minifasnet.onnx", threshold=0.85):
        self.threshold = threshold
        self._session = None
        self._input_name = ""
        
        if os.path.isfile(model_path):
            try:
                import onnxruntime as ort
                # Menggunakan CPU execution provider untuk Raspberry Pi
                self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                print(f"[AntiSpoofing] Model ONNX berhasil dimuat: {model_path}")
            except Exception as e:
                print(f"[AntiSpoofing] Gagal memuat model: {e}")
        else:
            print(f"[AntiSpoofing] Model tidak ditemukan di '{model_path}'")

    def is_real(self, frame, bbox):
        if not self._session:
            return {"real": True, "score": 1.0}

        x, y, w, h = bbox
        ih, iw = frame.shape[:2]
        
        # Perlebar area potong agar model mendapatkan konteks tekstur
        scale = 1.5
        cx, cy = x + w//2, y + h//2
        new_w, new_h = int(w * scale), int(h * scale)
        
        x1, y1 = max(0, cx - new_w // 2), max(0, cy - new_h // 2)
        x2, y2 = min(iw, cx + new_w // 2), min(ih, cy + new_h // 2)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return {"real": False, "score": 0.0}

        # Preprocessing untuk MiniFASNet (80x80)
        img = cv2.resize(face_crop, (80, 80))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self._session.run(None, {self._input_name: img})
        preds = outputs[0][0]
        score_real = float(preds[1]) # Indeks 1 adalah probabilitas "Real"
        
        return {"real": score_real >= self.threshold, "score": round(score_real, 4)}