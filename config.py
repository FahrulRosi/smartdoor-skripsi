# ── Camera & Hardware ────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
IR_CUT_PIN      = 12

# ── Database Firebase (BARU) ──────────────────────────────────────────────────
# GANTI LINK DI BAWAH INI DENGAN URL REALTIME DATABASE ANDA
FIREBASE_URL         = "https://smart-door-lock-feb6b-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_CREDENTIALS = "serviceAccountKey.json"

# ── FaceMesh ─────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

# ── Register Liveness Thresholds ─────────────────────────────────────────────
# Toleransi maksimal untuk deteksi wajah sedang "Tatap Lurus"
MAX_YAW   = 12.0  
MAX_PITCH = 12.0
MAX_ROLL  = 12.0

# Toleransi maksimal wajah lurus khusus saat tahap ekstraksi MobileFaceNet
EXTRACTION_MAX_YAW   = 12.0  
EXTRACTION_MAX_PITCH = 12.0
EXTRACTION_MAX_ROLL  = 12.0

# Syarat derajat minimum agar gerakan "Menoleh / Mengangguk / Miring" dianggap valid 
CHALLENGE_YAW   = 25.0  # Toleh Kanan/Kiri minimal 25 derajat
CHALLENGE_PITCH = 20.0  # Angguk Atas/Bawah minimal 20 derajat
CHALLENGE_ROLL  = 25.0  # Miring Kanan/Kiri minimal 25 derajat

# Mapping agar register.py bisa langsung membaca threshold tantangan
YAW_THRESHOLD   = CHALLENGE_YAW
PITCH_THRESHOLD = CHALLENGE_PITCH
ROLL_THRESHOLD  = CHALLENGE_ROLL

# Threshold kedipan mata (Eye Aspect Ratio). Diturunkan ke 0.16 agar tidak mudah salah deteksi
BLINK_EAR_THRESHOLD  = 0.21 
REGISTER_BLINK_COUNT = 2

# ── Anti-Spoofing & Recognition ──────────────────────────────────────────────
ANTI_SPOOFING_MODEL     = "liveness/antispoofing.onnx"
ANTI_SPOOFING_THRESHOLD = 0.85 

# Threshold kemiripan wajah (Cosine Similarity). 
# Jika masih sering "Wajah Tidak Dikenali", turunkan angka ini (misal ke 0.50 atau 0.45)
MATCH_THRESHOLD         = 0.75
MOBILEFACENET_PATH      = "recognition/mobilefacenet.onnx"

# ── Door Lock & UI ───────────────────────────────────────────────────────────
LOCK_GPIO_PIN   = 18
UNLOCK_DURATION = 5   

# Definisi Warna BGR untuk OpenCV
COLOR_GREEN  = (0,   220,   0)
COLOR_RED    = (0,     0, 220)
COLOR_YELLOW = (0,   210, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 220,   0)