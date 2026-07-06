import cv2
import threading
import platform

class CameraStream:
    def __init__(self, src=0, width=480, height=320):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        # Helper untuk mencoba membuka camera dengan API backend yang sesuai
        def try_open(src):
            if platform.system() == "Linux":
                # Coba dengan CAP_V4L2 terlebih dahulu (Rekomendasi untuk Raspberry Pi)
                try:
                    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                    if cap and cap.isOpened():
                        return cap
                except Exception as e:
                    print(f"[CameraStream] Gagal dengan CAP_V4L2 di Linux untuk index {src}: {e}")
                # Fallback ke default API jika CAP_V4L2 gagal
                try:
                    cap = cv2.VideoCapture(src)
                    if cap and cap.isOpened():
                        return cap
                except Exception as e:
                    print(f"[CameraStream] Gagal dengan default API di Linux untuk index {src}: {e}")
            else:
                # Gunakan default backend untuk Windows/Mac
                try:
                    cap = cv2.VideoCapture(src)
                    if cap and cap.isOpened():
                        return cap
                except Exception as e:
                    print(f"[CameraStream] Gagal membuka kamera index {src}: {e}")
            return None

        # Coba buka kamera yang dikonfigurasi
        self.cap = try_open(self.src)

        # Fallback ke index lain jika camera index utama gagal dibuka
        if not self.cap or not self.cap.isOpened():
            print(f"[CameraStream] Gagal membuka kamera index {self.src}. Mencari index kamera lain yang aktif...")
            candidates = [0, 1, 2, 4, 6, 8, 10]
            if self.src in candidates:
                candidates.remove(self.src)
            # Urutkan pencarian: list candidate utama terlebih dahulu, kemudian index 0-10 lainnya
            all_candidates = candidates + [i for i in range(11) if i not in candidates and i != self.src]
            
            for candidate in all_candidates:
                print(f"[CameraStream] Mencoba membuka kamera index {candidate}...")
                cap = try_open(candidate)
                if cap and cap.isOpened():
                    print(f"[CameraStream] Berhasil menemukan dan membuka kamera pada index {candidate}!")
                    self.src = candidate
                    self.cap = cap
                    break

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.src} or any fallback indices.")

        # Set resolusi kamera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

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
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
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