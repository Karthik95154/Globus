from __future__ import annotations

from pathlib import Path

import cv2


MODEL_DIR = Path(__file__).resolve().parent / "models"
FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
MODEL_POINTER = MODEL_DIR / "face_detector_path.txt"


def train_model() -> Path:
    """Prepare the OpenCV face detector used by the verification service.

    The selected Face Authentication approach uses OpenCV's pretrained Haar
    cascade for face detection and deterministic HOG-style descriptors for
    comparison, so there is no dataset-specific fitting step.
    """
    if not FACE_CASCADE_PATH.exists():
        raise FileNotFoundError(f"OpenCV face detector not found: {FACE_CASCADE_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_POINTER.write_text(str(FACE_CASCADE_PATH), encoding="utf-8")
    return MODEL_POINTER


if __name__ == "__main__":
    pointer = train_model()
    print(f"Model metadata saved to {pointer}")
