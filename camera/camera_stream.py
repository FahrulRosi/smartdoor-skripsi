import cv2
import threading
import platform

class CameraStream:
    def __init__(self, src=1, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        # Deteksi OS: Gunakan V4L2 HANYA jika dijalankan di Linux (Raspberry Pi)
        if platform.system() == "Linux":
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        else:
            # Gunakan default backend untuk Windows/Mac
            self.cap = cv2.VideoCapture(self.src)

        # Set resolusi kamera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.src}")

        self.running = True
        # Jalankan pembacaan kamera di thread terpisah agar tidak membuat lag program utama
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Loop internal untuk terus membaca frame dari kamera."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Gunakan lock agar frame tidak rusak/bertumpuk saat diakses oleh thread utama
                with self.lock:
                    self.frame = frame

    def read(self):
        """Mengambil frame terbaru untuk diproses oleh program utama."""
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        """Menghentikan kamera dan membersihkan resource."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()