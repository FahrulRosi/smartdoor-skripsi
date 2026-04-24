import threading
import cv2
import numpy as np

# Coba import picamera2 (libcamera native untuk Raspi terbaru)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("[WARNING] picamera2 not available. Will use cv2.VideoCapture (needs libcamerify)")


class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.picam = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.use_picamera2 = False

    def start(self):
        """Inisialisasi camera dengan fallback: picamera2 -> cv2.VideoCapture"""
        
        # ──── Coba picamera2 (libcamera native) ────
        if PICAMERA2_AVAILABLE:
            try:
                print("[CAMERA] Menggunakan picamera2 (libcamera native)")
                self.picam = Picamera2()
                config = self.picam.create_preview_configuration(
                    main={"format": "BGR888", "size": (self.width, self.height)}
                )
                self.picam.configure(config)
                self.picam.start()
                self.use_picamera2 = True
            except Exception as e:
                print(f"[WARNING] picamera2 failed: {e}. Fallback to cv2.VideoCapture")
                self.picam = None
                self.use_picamera2 = False
        
        # ──── Fallback ke cv2.VideoCapture ────
        if not self.use_picamera2:
            print("[CAMERA] Menggunakan cv2.VideoCapture (legacy)")
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
        """Update frame dari camera (picamera2 atau cv2)"""
        if self.use_picamera2:
            self._update_picamera2()
        else:
            self._update_cv2()

    def _update_picamera2(self):
        """Thread untuk picamera2"""
        while self.running:
            try:
                request = self.picam.capture_request()
                frame = request.make_array("main")
                request.release()
                
                with self.lock:
                    self.frame = frame
            except Exception as e:
                print(f"[ERROR] picamera2 update failed: {e}")
                break

    def _update_cv2(self):
        """Thread untuk cv2.VideoCapture"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        """Baca frame (return format sama untuk kedua backend)"""
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        """Hentikan camera stream"""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        if self.use_picamera2 and self.picam:
            try:
                self.picam.stop()
                self.picam.close()
            except:
                pass
        
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()