CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
IR_CUT_PIN      = 12

# ── FaceMesh ──────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

# ── Register Liveness Thresholds ──────────────────────────────────────────────
# Batas toleransi untuk wajah dianggap "Lurus / Tengah" 
# (Ditingkatkan sedikit agar lebih mudah mendeteksi posisi lurus)
MAX_YAW   = 12.0  
MAX_PITCH = 12.0
MAX_ROLL  = 12.0

# Batas untuk pengambilan foto database (Tahap Akhir)
# Disamakan dengan MAX agar tidak terjadi "bottleneck" saat transisi
EXTRACTION_MAX_YAW   = 12.0  
EXTRACTION_MAX_PITCH = 12.0
EXTRACTION_MAX_ROLL  = 12.0

# Batas derajat gerakan untuk dinyatakan lulus tantangan
# Diturunkan dari 15.0 ke 12.0 agar tantangan tidak terasa terlalu sulit/berat
CHALLENGE_YAW   = 12.0  
CHALLENGE_PITCH = 12.0  
CHALLENGE_ROLL  = 12.0  

REGISTER_BLINK_COUNT = 1 # Dikurangi menjadi 1 agar proses lebih cepat dan efisien

# ── Anti-Spoofing & Recognition ──────────────────────────────────────────────
ANTI_SPOOFING_MODEL = "liveness/antispoofing.onnx"
ANTI_SPOOFING_THRESHOLD = 0.85 
MATCH_THRESHOLD = 0.60        # Ditingkatkan ke 0.60 agar akurasi verifikasi wajah lebih aman
MOBILEFACENET_PATH = "recognition/mobilefacenet.onnx"

# ── Door Lock & UI ──────────────────────────────────────────────────────────
LOCK_GPIO_PIN   = 18
UNLOCK_DURATION = 5   

COLOR_GREEN  = (0,   220,   0)
COLOR_RED    = (0,     0, 220)
COLOR_YELLOW = (0,   210, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 220,   0)