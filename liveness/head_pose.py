import cv2
import numpy as np
from facemesh.facemesh_detector import FaceResult, FaceMeshDetector


class HeadPoseEstimator:
    """
    Returns yaw, pitch, roll in degrees.

    Convention (positive direction):
        yaw   → looking right
        pitch → looking up
        roll  → head tilting right
    """

    def estimate(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        image_pts, model_pts, cam_mat, dist = detector.get_head_pose_points(face)

        success, rvec, tvec = cv2.solvePnP(
            model_pts, image_pts, cam_mat, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "valid": False}

        rmat, _ = cv2.Rodrigues(rvec)
        proj_mat = np.hstack([rmat, tvec])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)

        pitch = float(euler[0, 0])
        yaw   = float(euler[1, 0])
        roll  = float(euler[2, 0])

        return {
            "yaw":   round(yaw,   2),
            "pitch": round(pitch, 2),
            "roll":  round(roll,  2),
            "valid": True,
        }