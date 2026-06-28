# ── Camera & Hardware ────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
FRAME_WIDTH     = 480
FRAME_HEIGHT    = 320
FPS             = 30

# ── Peningkatan Gambar (Low-Light & Backlight) ───────────────────────────────
ENABLE_CLAHE_ENHANCEMENT = True
CLAHE_CLIP_LIMIT         = 1.2
CLAHE_TILE_GRID_SIZE     = (8, 8)

# ── Database Supabase ────────────────────────────────────────────────────────
SUPABASE_URL = "https://gwwxebdmavlmxxcdrlge.supabase.co"
SUPABASE_KEY = "sb_publishable_tUsmI6J1BubOlRf9YhtNPQ_F9dWIyOv"

# ── FaceMesh ─────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE  = 0.55

# ── Register Liveness Thresholds ─────────────────────────────────────────────
MAX_YAW   = 12.0  
MAX_PITCH = 12.0
MAX_ROLL  = 12.0

EXTRACTION_MAX_YAW   = 12.0  
EXTRACTION_MAX_PITCH = 12.0
EXTRACTION_MAX_ROLL  = 12.0

CHALLENGE_YAW   = 25.0  
CHALLENGE_PITCH = 20.0  
CHALLENGE_ROLL  = 25.0  

YAW_THRESHOLD   = CHALLENGE_YAW
PITCH_THRESHOLD = CHALLENGE_PITCH
ROLL_THRESHOLD  = CHALLENGE_ROLL

BLINK_EAR_THRESHOLD  = 0.22
REGISTER_BLINK_COUNT = 2

MIN_BLINK_OPEN_EAR   = 0.23  
MAX_BLINK_CLOSED_EAR = 0.21  
MIN_BLINK_DELTA      = 0.025  

ANTI_SPOOFING_MODEL     = "liveness/antispoofing.onnx"
ANTI_SPOOFING_THRESHOLD = 0.80
MATCH_THRESHOLD         = 0.70
ANTI_DUPLICATE_THRESHOLD = 0.65
MOBILEFACENET_PATH      = "recognition/mobilefacenet.onnx"
MAX_SPOOF_FRAMES        = 8

# ── Door Lock & UI ───────────────────────────────────────────────────────────
LOCK_GPIO_PIN   = 18
BUTTON_PIN      = 26  
UNLOCK_DURATION = 5  
RELAY_ACTIVE_HIGH = False  # Set False jika menggunakan relay Active Low (default modul Arduino/Raspberry Pi)


# Definisi Warna BGR untuk OpenCV
COLOR_GREEN  = (0,   220,   0)
COLOR_RED    = (0,     0, 220)
COLOR_YELLOW = (0,   210, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 220,   0)