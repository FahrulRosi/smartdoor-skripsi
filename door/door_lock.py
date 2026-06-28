import time
import threading

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[DoorLock] RPi.GPIO not found — running in SIMULATION mode.")


LOCK_PIN = 18          # BCM pin connected to relay / solenoid
UNLOCK_DURATION = 5    # seconds the door stays unlocked


class DoorLock:
    def __init__(self, pin: int = LOCK_PIN, unlock_duration: int = UNLOCK_DURATION, active_high: bool = False):
        self.pin = pin
        self.unlock_duration = unlock_duration
        self.active_high = active_high
        self.locked = True
        self._timer: threading.Timer | None = None

        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            # Menyesuaikan status awal terkunci berdasarkan tipe relay (Active High/Low)
            initial_state = GPIO.LOW if self.active_high else GPIO.HIGH
            GPIO.output(self.pin, initial_state)   
            state_str = "LOW" if initial_state == GPIO.LOW else "HIGH"
            print(f"[DoorLock] GPIO ready on pin {self.pin}. Status awal: TERKUNCI ({state_str}).")

    # ------------------------------------------------------------------ #
    def unlock(self):
        if self._timer:
            self._timer.cancel()

        self._set_lock(False)
        print(f"[DoorLock] 🔓 UNLOCKED — will re-lock in {self.unlock_duration}s")
        self._timer = threading.Timer(self.unlock_duration, self.lock)
        self._timer.daemon = True
        self._timer.start()

    def lock(self):
        self._set_lock(True)
        print("[DoorLock] 🔒 LOCKED")

    def _set_lock(self, locked: bool):
        self.locked = locked
        if GPIO_AVAILABLE:
            # Jika locked=True (terkunci): Active High butuh LOW, Active Low butuh HIGH
            # Jika locked=False (terbuka): Active High butuh HIGH, Active Low butuh LOW
            if self.active_high:
                output_val = GPIO.LOW if locked else GPIO.HIGH
            else:
                output_val = GPIO.HIGH if locked else GPIO.LOW
            GPIO.output(self.pin, output_val)

    def status(self) -> str:
        return "LOCKED" if self.locked else "UNLOCKED"

    def cleanup(self):
        if self._timer:
            self._timer.cancel()
        if GPIO_AVAILABLE:
            # Mengembalikan pin ke mode input agar aman saat program dimatikan
            GPIO.cleanup()
        print("[DoorLock] Cleanup done.")

    # ------------------------------------------------------------------ #
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()