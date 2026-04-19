import random
import time
from enum import Enum


class ChallengeType(str, Enum):
    BLINK      = "blink"
    LOOK_LEFT  = "look_left"
    LOOK_RIGHT = "look_right"


CHALLENGE_LABELS = {
    ChallengeType.BLINK:      "Please BLINK",
    ChallengeType.LOOK_LEFT:  "Look LEFT",
    ChallengeType.LOOK_RIGHT: "Look RIGHT",
}


class Challenge:
    def __init__(self, timeout: float = 8.0, num_challenges: int = 2):
        self.timeout = timeout
        self.num_challenges = num_challenges
        self._sequence: list[ChallengeType] = []
        self._index = 0
        self._start_time: float = 0.0
        self.complete = False
        self.failed = False

    # ------------------------------------------------------------------ #
    def generate(self) -> list[ChallengeType]:
        all_types = list(ChallengeType)
        self._sequence = random.sample(all_types, k=min(self.num_challenges, len(all_types)))
        self._index = 0
        self._start_time = time.time()
        self.complete = False
        self.failed = False
        return self._sequence

    @property
    def current(self) -> ChallengeType | None:
        if self._index < len(self._sequence):
            return self._sequence[self._index]
        return None

    @property
    def current_label(self) -> str:
        c = self.current
        return CHALLENGE_LABELS[c] if c else "All done!"

    def advance(self) -> bool:
        """Call when the current challenge is satisfied. Returns True if more remain."""
        self._index += 1
        if self._index >= len(self._sequence):
            self.complete = True
            return False
        return True

    def check_timeout(self) -> bool:
        if time.time() - self._start_time > self.timeout:
            self.failed = True
            return True
        return False

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.timeout - (time.time() - self._start_time))

    @property
    def progress(self) -> str:
        return f"[{self._index}/{len(self._sequence)}] {self.current_label}"