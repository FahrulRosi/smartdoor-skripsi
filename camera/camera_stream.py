import cv2
import threading


class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.src}")

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()