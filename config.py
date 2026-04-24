# ──────────────────────────────────────────────────────────────────────────────
#  SMART_DOOR_LOCK  –  config.py
# ──────────────────────────────────────────────────────────────────────────────

# ── Camera & Hardware ─────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
IR_CUT_PIN      = 12   # Pin GPIO untuk mengontrol Filter IR-CUT (Mode Siang/Malam)

# ── FaceMesh ──────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.5  # ← Turun dari 0.7
MIN_TRACKING_CONFIDENCE  = 0.5  # ← Turun dari 0.7

# ── Register Liveness Thresholds (Alur Bertahap) ──────────────────────────────
# Batas toleransi untuk wajah dianggap "Lurus / Tengah" (untuk FaceMesh awal)
MAX_YAW   = 10.0  
MAX_PITCH = 10.0
MAX_ROLL  = 10.0

# Batas minimum derajat gerakan untuk dinyatakan lulus tantangan (Challenge)
CHALLENGE_YAW   = 20.0  # Harus menoleh (Kiri/Kanan) lebih dari 20 derajat
CHALLENGE_PITCH = 15.0  # Harus mendongak/menunduk lebih dari 15 derajat
CHALLENGE_ROLL  = 15.0  # Harus miringkan kepala lebih dari 15 derajat

REGISTER_BLINK_COUNT = 2   # Jumlah wajib kedip saat tahap Blink

# ── Anti-Spoofing (Passive Liveness) ──────────────────────────────────────────
ANTI_SPOOFING_MODEL = "liveness/2.7_80x80_MiniFASNetV2.onnx"
ANTI_SPOOFING_THRESHOLD = 0.85 # Skor minimum untuk dianggap manusia hidup (bukan foto)

# ── Face Recognition (MobileFaceNet) ──────────────────────────────────────────
MATCH_THRESHOLD = 0.55         # Batas kecocokan wajah (Cosine Similarity)
MOBILEFACENET_PATH = "recognition/mobilefacenet.onnx"

# ── Door Lock ─────────────────────────────────────────────────────────────────
LOCK_GPIO_PIN   = 18
UNLOCK_DURATION = 5   # Detik pintu tetap terbuka saat akses diterima

# ── Display & UI ──────────────────────────────────────────────────────────────
FONT       = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
LINE_TYPE  = 2

COLOR_GREEN  = (0,   220,   0)
COLOR_RED    = (0,     0, 220)
COLOR_YELLOW = (0,   210, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_CYAN   = (255, 220,   0)