import cv2
import numpy as np
import os
import config


_MODEL_PATH = getattr(config, "MOBILEFACENET_PATH",
                      os.path.join(os.path.dirname(__file__), "mobilefacenet.onnx"))


def _preprocess(face_img: np.ndarray, size: int = 112) -> np.ndarray:
    img = cv2.resize(face_img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0           # MobileFaceNet normalisation
    img = img.transpose(2, 0, 1)          # HWC → CHW
    img = np.expand_dims(img, 0)          # add batch dim
    return img


class MobileFaceNet:
    def __init__(self):
        self._session = None
        self._input_name: str = ""
        self._stub = False

        if os.path.isfile(_MODEL_PATH):
            try:
                import onnxruntime as ort
                self._session   = ort.InferenceSession(
                    _MODEL_PATH,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                print(f"[MobileFaceNet] Loaded ONNX model: {_MODEL_PATH}")
            except Exception as exc:
                print(f"[MobileFaceNet] ONNX load failed ({exc}). Using stub.")
                self._stub = True
        else:
            print(f"[MobileFaceNet] Model not found at '{_MODEL_PATH}'. Using stub.")
            self._stub = True

    # ------------------------------------------------------------------ #
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Returns a 1-D float32 embedding vector (512-dim with real model,
        128-dim with stub).
        """
        if self._stub:
            return self._stub_embedding(face_img)

        blob = _preprocess(face_img)
        outputs = self._session.run(None, {self._input_name: blob})
        embedding = outputs[0][0]                      # shape (512,)
        embedding /= np.linalg.norm(embedding) + 1e-8  # L2-normalise
        return embedding.astype(np.float32)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _stub_embedding(img: np.ndarray) -> np.ndarray:
        """
        Deterministic 128-dim pseudo-embedding derived from image statistics.
        Good enough for integration tests; NOT suitable for real recognition.
        """
        resized = cv2.resize(img, (64, 64)).astype(np.float32) / 255.0
        rng = np.random.default_rng(seed=int(resized.mean() * 1e6) % (2**32))
        vec = rng.standard_normal(128).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec

    # ------------------------------------------------------------------ #
    def crop_face(self, frame: np.ndarray, bbox: tuple,
                  margin: float = 0.2) -> np.ndarray:
        """Crop + margin from frame using the FaceResult bbox (x, y, w, h)."""
        x, y, w, h = bbox
        ih, iw = frame.shape[:2]
        mx = int(w * margin)
        my = int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(iw, x + w + mx)
        y2 = min(ih, y + h + my)
        return frame[y1:y2, x1:x2]