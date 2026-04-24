import cv2
import numpy as np
import os

class SilentAntiSpoofing:
    def __init__(self, model_path="2.7_80x80_MiniFASNetV2.onnx", threshold=0.85):
        self.threshold = threshold
        self._session = None
        self._input_name = ""
        
        # Sesuai nama file "2.7_80x80...", scale yang wajib digunakan adalah 2.7
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
        """Menghitung kotak pemotongan wajah sesuai algoritma Minivision asli"""
        x, y, box_w, box_h = bbox
        
        # Batasi scale agar tidak keluar dari frame kamera
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2 + x
        center_y = box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def is_real(self, frame, bbox):
        if not self._session:
            return {"real": True, "score": 1.0}

        src_h, src_w = frame.shape[:2]

        # 1. Potong gambar sesuai skala 2.7
        x1, y1, x2, y2 = self._get_new_box(src_w, src_h, bbox, self.scale)
        face_crop = frame[y1:y2+1, x1:x2+1]

        if face_crop.size == 0:
            return {"real": False, "score": 0.0}

        # 2. Preprocessing (WAJIB FORMAT BGR - Jangan di-convert ke RGB)
        face_resized = cv2.resize(face_crop, (80, 80))
        face_float = face_resized.astype(np.float32)
        face_chw = np.transpose(face_float, (2, 0, 1))
        img_data = np.expand_dims(face_chw, axis=0)

        # 3. Prediksi (Mendapatkan Raw Output)
        output = self._session.run(None, {self._input_name: img_data})[0]

        # 4. Softmax (Ubah output mentah jadi probabilitas 0.0 sampai 1.0)
        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
        softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        preds = softmax_output[0]

        # Indeks dari model: 0=Kertas/Foto, 1=Wajah Asli, 2=Layar HP
        score_real = float(preds[1])

        # Hapus tanda pagar '#' di bawah ini jika ingin melihat skor aslinya di terminal
        # print(f"[Cek Liveness] Kertas: {preds[0]:.2f} | Asli: {preds[1]:.2f} | Layar: {preds[2]:.2f}")

        is_valid = score_real >= self.threshold
        return {"real": is_valid, "score": round(score_real, 4)}