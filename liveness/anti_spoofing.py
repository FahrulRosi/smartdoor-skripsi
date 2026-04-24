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
                # Menggunakan CPU provider agar aman di Raspberry Pi
                self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                print(f"[AntiSpoofing] Loaded ONNX model: {model_path}")
            except Exception as exc:
                print(f"[AntiSpoofing] Gagal memuat ONNX ({exc}).")
        else:
            print(f"[AntiSpoofing] Model tidak ditemukan di '{model_path}'.")

    def is_real(self, frame: np.ndarray, bbox: tuple) -> dict:
        """
        Mengembalikan status True jika wajah asli, False jika foto/layar.
        """
        if not self._session:
            # Fallback jika model tidak ada, anggap True (bahaya, hanya untuk testing)
            return {"real": True, "score": 1.0, "label": "No_Model"}

        # 1. Perlebar bounding box (MiniFASNet butuh sedikit latar belakang)
        x, y, w, h = bbox
        ih, iw = frame.shape[:2]
        
        # Scale bbox sekitar 1.5x - 2.0x
        scale = 1.5
        cx, cy = x + w//2, y + h//2
        new_w, new_h = int(w * scale), int(h * scale)
        
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(iw, cx + new_w // 2)
        y2 = min(ih, cy + new_h // 2)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return {"real": False, "score": 0.0, "label": "Invalid_Crop"}

        # 2. Preprocessing (Resize ke 80x80 sesuai standar MiniFASNetV2)
        img = cv2.resize(face_crop, (80, 80))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC ke CHW
        img = np.expand_dims(img, axis=0)   # Tambah batch dimension

        # 3. Inference
        outputs = self._session.run(None, {self._input_name: img})
        preds = outputs[0][0] # Array probabilitas [Spoof, Real]

        # Indeks 1 biasanya untuk kelas 'Real' (Asli), Indeks 0 untuk 'Spoof' (Palsu)
        score_real = float(preds[1])
        
        is_real = score_real >= self.threshold
        label = "Asli" if is_real else "Palsu (Spoof)"

        return {
            "real": is_real,
            "score": round(score_real, 4),
            "label": label
        }