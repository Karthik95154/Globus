# Globussoft Data Science Task

This repository contains both compulsory tasks from `DataScienceTask-new.pdf`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Task 1: Amazon Laptop Scraper

The scraper collects laptop search results from `amazon.in` and saves a timestamped CSV.

```bash
python task1/amazon_scraper.py --keyword laptop --pages 1 --output-dir outputs
```

CSV columns:

- `image`
- `title`
- `rating`
- `price`
- `result_type`
- `product_url`

Amazon can show bot checks or alter markup, so the script uses browser-like headers and conservative delays.

## Task 2: Face Authentication

Chosen option: **A) Face Authentication (Face Verification)**.

The service uses OpenCV's pretrained Haar cascade to detect faces and a deterministic HOG-style OpenCV descriptor to compare the largest detected face in each image.

Prepare model metadata:

```bash
python task2/training.py
```

Run the FastAPI app:

```bash
python -m uvicorn task2.app:app --reload
```

Verify two uploaded images:

```bash
curl -X POST "http://127.0.0.1:8000/verify" ^
  -F "image1=@samples/face_sample_1.png" ^
  -F "image2=@samples/face_sample_2.png"
```

The API returns:

- `verification_result`: `same person` or `different person`
- `similarity_score`
- `bounding_boxes` for each detected face in `[x, y, width, height]` format

The standalone prediction function is available in `task2/testing.py`:

```python
from task2.testing import predict

result = predict("samples/face_sample_1.png", "samples/face_sample_2.png")
print(result)
```

Sample images are derived from the public-domain scikit-image astronaut image, originally from NASA.
