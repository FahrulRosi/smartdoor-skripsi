# ──────────────────────────────────────────────────────────────────────────────
#  SMART_DOOR_LOCK  –  config.py
# ──────────────────────────────────────────────────────────────────────────────

# Camera
CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480

# FaceMesh
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.7

# ── Register liveness thresholds ──────────────────────────────────────────────
MAX_YAW   = 15.0    # degrees — head must face forward (±)
MAX_PITCH = 15.0
MAX_ROLL  = 15.0
REGISTER_BLINK_COUNT = 2   # must blink twice during registration

# ── Unlock challenge ──────────────────────────────────────────────────────────
NUM_CHALLENGES      = 2     # how many random challenges per unlock attempt
CHALLENGE_TIMEOUT   = 8.0   # seconds to complete each challenge
LOOK_YAW_THRESHOLD  = 20.0  # minimum |yaw| to count as "looking left/right"

# ── Face recognition ──────────────────────────────────────────────────────────
MATCH_THRESHOLD = 0.55          # cosine similarity threshold
MOBILEFACENET_PATH = "recognition/mobilefacenet.onnx"

# ── Door lock ─────────────────────────────────────────────────────────────────
LOCK_GPIO_PIN    = 18
UNLOCK_DURATION  = 5   # seconds door stays unlocked

# ── Display ───────────────────────────────────────────────────────────────────
FONT       = 1          # cv2.FONT_HERSHEY_SIMPLEX shorthand
FONT_SCALE = 0.65
LINE_TYPE  = 2

COLOR_GREEN  = (0,   220,   0)
COLOR_RED    = (0,     0, 220)
COLOR_YELLOW = (0,   210, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 220,   0)