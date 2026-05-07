from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from task2.testing import decode_image, load_model, predict_from_arrays


app = FastAPI(title="Face Authentication Service", version="1.0.0")
detector = load_model()


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "face-authentication"}


@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
) -> dict[str, object]:
    try:
        first_image = decode_image(await image1.read())
        second_image = decode_image(await image2.read())
        result = predict_from_arrays(first_image, second_image, detector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "verification_result": result.verification_result,
        "similarity_score": result.similarity_score,
        "bounding_boxes": {
            "image1": result.face1_box,
            "image2": result.face2_box,
        },
    }
