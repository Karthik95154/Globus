from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MODEL_POINTER = Path(__file__).resolve().parent / "models" / "face_detector_path.txt"
DEFAULT_FACE_CASCADE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
SAME_PERSON_THRESHOLD = 0.82


@dataclass(frozen=True)
class FaceMatchResult:
    verification_result: str
    similarity_score: float
    face1_box: list[int]
    face2_box: list[int]


def load_model() -> cv2.CascadeClassifier:
    model_path = DEFAULT_FACE_CASCADE
    if MODEL_POINTER.exists():
        saved_path = Path(MODEL_POINTER.read_text(encoding="utf-8").strip())
        if saved_path.exists():
            model_path = saved_path

    detector = cv2.CascadeClassifier(str(model_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load face detector from {model_path}")
    return detector


def read_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image


def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode uploaded image")
    return image


def detect_largest_face(image: np.ndarray, detector: cv2.CascadeClassifier) -> tuple[np.ndarray, list[int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = gray[y : y + h, x : x + w]
    return face, [int(x), int(y), int(w), int(h)]


def extract_embedding(face_gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_gray, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = cv2.equalizeHist(resized)

    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9,
    )
    descriptor = hog.compute(normalized).reshape(-1).astype(np.float32)
    descriptor_norm = np.linalg.norm(descriptor)
    if descriptor_norm == 0:
        return descriptor
    return descriptor / descriptor_norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def predict_from_arrays(
    image1: np.ndarray,
    image2: np.ndarray,
    detector: cv2.CascadeClassifier | None = None,
    threshold: float = SAME_PERSON_THRESHOLD,
) -> FaceMatchResult:
    detector = detector or load_model()
    face1, box1 = detect_largest_face(image1, detector)
    face2, box2 = detect_largest_face(image2, detector)

    similarity = cosine_similarity(extract_embedding(face1), extract_embedding(face2))
    label = "same person" if similarity >= threshold else "different person"
    return FaceMatchResult(
        verification_result=label,
        similarity_score=round(similarity, 4),
        face1_box=box1,
        face2_box=box2,
    )


def predict(image1_path: str | Path, image2_path: str | Path) -> dict[str, object]:
    result = predict_from_arrays(read_image(image1_path), read_image(image2_path))
    return {
        "verification_result": result.verification_result,
        "similarity_score": result.similarity_score,
        "bounding_boxes": {
            "image1": result.face1_box,
            "image2": result.face2_box,
        },
    }
