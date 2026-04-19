"""
LivenessManager
===============
Register flow  → checks Yaw + Pitch + Roll range  AND  blink count
Unlock  flow   → executes a random Challenge sequence (blink / look left / right)
"""

from dataclasses import dataclass, field
from facemesh.facemesh_detector import FaceResult, FaceMeshDetector
from liveness.blink import BlinkDetector
from liveness.head_pose import HeadPoseEstimator
from liveness.challenge import Challenge, ChallengeType
import config


@dataclass
class LivenessState:
    yaw_ok:   bool = False
    pitch_ok: bool = False
    roll_ok:  bool = False
    blink_ok: bool = False

    @property
    def all_passed(self) -> bool:
        return self.yaw_ok and self.pitch_ok and self.roll_ok and self.blink_ok

    def summary(self) -> str:
        checks = {
            "Yaw":   self.yaw_ok,
            "Pitch": self.pitch_ok,
            "Roll":  self.roll_ok,
            "Blink": self.blink_ok,
        }
        parts = [f"{'✅' if v else '❌'} {k}" for k, v in checks.items()]
        return "  ".join(parts)


class LivenessManager:
    """
    Handles liveness for BOTH register and unlock modes.
    Instantiate once per session.
    """

    # ── Register ──────────────────────────────────────────────────────── #
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self._register_state: LivenessState | None = None
        self._register_blink: BlinkDetector | None = None
        self._challenge: Challenge | None = None
        self._unlock_blink: BlinkDetector | None = None

    # ── REGISTER ─────────────────────────────────────────────────────── #
    def start_register(self):
        self._register_state = LivenessState()
        self._register_blink = BlinkDetector(target_blinks=config.REGISTER_BLINK_COUNT)

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> LivenessState:
        state = self._register_state
        pose  = self.pose_estimator.estimate(face, detector)
        blink = self._register_blink.update(face, detector)

        if pose["valid"]:
            state.yaw_ok   = abs(pose["yaw"])   <= config.MAX_YAW
            state.pitch_ok = abs(pose["pitch"])  <= config.MAX_PITCH
            state.roll_ok  = abs(pose["roll"])   <= config.MAX_ROLL

        state.blink_ok = blink["complete"]
        return state

    @property
    def register_complete(self) -> bool:
        return self._register_state is not None and self._register_state.all_passed

    # ── UNLOCK ───────────────────────────────────────────────────────── #
    def start_unlock(self) -> list[ChallengeType]:
        self._challenge    = Challenge(
            timeout=config.CHALLENGE_TIMEOUT,
            num_challenges=config.NUM_CHALLENGES,
        )
        self._unlock_blink = BlinkDetector(target_blinks=1)
        return self._challenge.generate()

    def update_unlock(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        ch = self._challenge
        if ch.check_timeout():
            return {"status": "timeout", "challenge": None}

        current = ch.current
        satisfied = False

        if current == ChallengeType.BLINK:
            result = self._unlock_blink.update(face, detector)
            if result["complete"]:
                self._unlock_blink.reset()
                satisfied = True

        elif current in (ChallengeType.LOOK_LEFT, ChallengeType.LOOK_RIGHT):
            pose = self.pose_estimator.estimate(face, detector)
            if pose["valid"]:
                if current == ChallengeType.LOOK_LEFT  and pose["yaw"] < -config.LOOK_YAW_THRESHOLD:
                    satisfied = True
                elif current == ChallengeType.LOOK_RIGHT and pose["yaw"] >  config.LOOK_YAW_THRESHOLD:
                    satisfied = True

        if satisfied:
            more = ch.advance()
            if not more:
                return {"status": "complete", "challenge": current}

        return {
            "status":    "pending",
            "challenge": current,
            "label":     ch.current_label,
            "progress":  ch.progress,
            "remaining": round(ch.remaining_seconds, 1),
        }

    @property
    def unlock_complete(self) -> bool:
        return self._challenge is not None and self._challenge.complete

    @property
    def unlock_failed(self) -> bool:
        return self._challenge is not None and self._challenge.failed