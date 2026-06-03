# ── Camera & Hardware ────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480

# ── Peningkatan Gambar (Low-Light & Backlight) ───────────────────────────────
ENABLE_CLAHE_ENHANCEMENT = True
CLAHE_CLIP_LIMIT         = 2.5
CLAHE_TILE_GRID_SIZE     = (8, 8)

# ── Database Supabase ────────────────────────────────────────────────────────
SUPABASE_URL = "https://gwwxebdmavlmxxcdrlge.supabase.co"


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
CHALLENGE_YAW   = 25.0  
CHALLENGE_PITCH = 20.0  
CHALLENGE_ROLL  = 25.0  

# Mapping agar register.py bisa langsung membaca threshold tantangan
YAW_THRESHOLD   = CHALLENGE_YAW
PITCH_THRESHOLD = CHALLENGE_PITCH
ROLL_THRESHOLD  = CHALLENGE_ROLL

# Threshold kedipan mata cadangan (Eye Aspect Ratio Fallback)
BLINK_EAR_THRESHOLD  = 0.21 
REGISTER_BLINK_COUNT = 2

# ── Quality Control Kedipan Baru (Garansi Data Valid saat Register) ──────────
MIN_BLINK_OPEN_EAR   = 0.22  # Nilai minimum EAR agar mata dianggap BENAR-BENAR TERBUKA
MAX_BLINK_CLOSED_EAR = 0.20  # Nilai maksimum EAR agar mata dianggap BENAR-BENAR TERTUTUP
MIN_BLINK_DELTA      = 0.04  # Selisih wajib melek-merem demi memutus rantai data fraud

# ── Anti-Spoofing & Recognition ──────────────────────────────────────────────
ANTI_SPOOFING_MODEL     = "liveness/antispoofing.onnx"
# --- PERBAIKAN: Diturunkan dari 0.95 ke 0.85 agar tidak terlalu sering false rejection di jarak dekat/cahaya rendah ---
ANTI_SPOOFING_THRESHOLD = 0.85 

# --- PERBAIKAN: Threshold kemiripan wajah DINAIKKAN AGAR KETAT ---
MATCH_THRESHOLD         = 0.82
MOBILEFACENET_PATH      = "recognition/mobilefacenet.onnx"
# -----------------------------------------------------------------

# ── Door Lock & UI ───────────────────────────────────────────────────────────
LOCK_GPIO_PIN   = 18
BUTTON_PIN      = 23
UNLOCK_DURATION = 5   

# Definisi Warna BGR untuk OpenCV
COLOR_GREEN  = (0,   220,   0)
COLOR_RED    = (0,     0, 220)
COLOR_YELLOW = (0,   210, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 220,   0)