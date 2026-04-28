CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
IR_CUT_PIN      = 12

# ── FaceMesh ──────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

# ── Register Liveness Thresholds (Alur Bertahap Tanpa Delay) ─────────────────
# Batas toleransi untuk wajah dianggap "Lurus / Tengah" 
MAX_YAW   = 10.0  
MAX_PITCH = 10.0
MAX_ROLL  = 10.0

# Batas SANGAT KETAT untuk pengambilan foto database (Tahap Akhir)
EXTRACTION_MAX_YAW   = 8.0  
EXTRACTION_MAX_PITCH = 8.0
EXTRACTION_MAX_ROLL  = 8.0

# Batas minimum derajat gerakan untuk dinyatakan lulus tantangan
CHALLENGE_YAW   = 15.0  
CHALLENGE_PITCH = 15.0  
CHALLENGE_ROLL  = 15.0  

REGISTER_BLINK_COUNT = 2   

# ── Anti-Spoofing & Recognition ──────────────────────────────────────────────
ANTI_SPOOFING_MODEL = "liveness/antispoofing.onnx"
ANTI_SPOOFING_THRESHOLD = 0.85 
MATCH_THRESHOLD = 0.55         
MOBILEFACENET_PATH = "recognition/mobilefacenet.onnx"

# ── Door Lock & UI ──────────────────────────────────────────────────────────
LOCK_GPIO_PIN   = 18
UNLOCK_DURATION = 5   

COLOR_GREEN  = (0,   220,   0)
COLOR_RED    = (0,     0, 220)
COLOR_YELLOW = (0,   210, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 220,   0)