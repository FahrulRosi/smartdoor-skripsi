# config.py

CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
IR_CUT_PIN      = 12

# ── FaceMesh ──────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

# ── Register Liveness Thresholds ──────────────────────────────────────────────
# Batas toleransi untuk wajah dianggap "Lurus / Tengah" 
MAX_YAW   = 12.0  
MAX_PITCH = 12.0
MAX_ROLL  = 12.0

# Batas untuk pengambilan foto database (Tahap Akhir)
EXTRACTION_MAX_YAW   = 12.0  
EXTRACTION_MAX_PITCH = 12.0
EXTRACTION_MAX_ROLL  = 12.0

# Batas derajat gerakan untuk dinyatakan lulus tantangan
CHALLENGE_YAW   = 12.0  
CHALLENGE_PITCH = 12.0  
CHALLENGE_ROLL  = 12.0  

# Mapping agar register.py bisa membaca threshold tersebut
YAW_THRESHOLD   = CHALLENGE_YAW
PITCH_THRESHOLD = CHALLENGE_PITCH
ROLL_THRESHOLD  = CHALLENGE_ROLL

# Liveness Blinking
BLINK_EAR_THRESHOLD  = 0.22 # Pastikan nilai ini ada untuk deteksi kedipan
REGISTER_BLINK_COUNT = 1 

# ── Anti-Spoofing & Recognition ──────────────────────────────────────────────
ANTI_SPOOFING_MODEL     = "liveness/antispoofing.onnx"
ANTI_SPOOFING_THRESHOLD = 0.85 
MATCH_THRESHOLD         = 0.60 
MOBILEFACENET_PATH      = "recognition/mobilefacenet.onnx"

# ── Door Lock & UI ──────────────────────────────────────────────────────────
LOCK_GPIO_PIN   = 18
UNLOCK_DURATION = 5   

COLOR_GREEN  = (0,   220,   0)
COLOR_RED    = (0,     0, 220)
COLOR_YELLOW = (0,   210, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 220,   0)