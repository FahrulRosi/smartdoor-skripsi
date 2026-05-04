# ── Camera & Hardware ────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
IR_CUT_PIN      = 12

# ── Peningkatan Gambar (Low-Light & Backlight) ───────────────────────────────
ENABLE_CLAHE_ENHANCEMENT = True
CLAHE_CLIP_LIMIT         = 2.5
CLAHE_TILE_GRID_SIZE     = (8, 8)

# ── Database Firebase ────────────────────────────────────────────────────────
FIREBASE_URL         = "https://smart-door-lock-feb6b-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_CREDENTIALS = "serviceAccount.json"

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

# Threshold kedipan mata (Eye Aspect Ratio)
BLINK_EAR_THRESHOLD  = 0.21 
REGISTER_BLINK_COUNT = 2

# ── Anti-Spoofing & Recognition ──────────────────────────────────────────────
ANTI_SPOOFING_MODEL     = "liveness/antispoofing.onnx"
ANTI_SPOOFING_THRESHOLD = 0.85 

# Threshold kemiripan wajah (Cosine Similarity).
# Diturunkan dari 0.75 menjadi 0.60 untuk mencegah pendaftaran wajah ganda (duplikasi)
# dan meningkatkan akurasi pengenalan saat kondisi wajah sedikit berbeda.
MATCH_THRESHOLD         = 0.60
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