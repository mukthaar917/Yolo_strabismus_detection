import base64
import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from strabismus_yolo import MediaPipeEyeLocator, process_image_with_yolo

# ========================================================
# MODEL & LOCATOR LIFECYCLE (LOADED ONCE AT STARTUP)
# ========================================================
WEIGHTS_DIR = r"weights"
MP_TASK_MODEL = r"weights\face_landmarker.task"

# Default production thresholds (96.0% triage-aware accuracy, 0% FP on negatives)
DEFAULT_T_HORIZ = 3
DEFAULT_T_VERT = 5
DEFAULT_T_DOMINANCE = 1
DEFAULT_T_SHARED_SHIFT = 8
DEFAULT_T_IRIS_ASYM = 0.105
DEFAULT_ENABLE_ROLL_COMP = False
DEFAULT_ENABLE_CLAHE_ROI = False

state = {
    "model": None,
    "mp_locator": None,
    "is_custom_model": False,
}


def _resolve_model_path() -> tuple[str, bool]:
    if os.path.isdir(WEIGHTS_DIR):
        custom_weights = [
            f
            for f in os.listdir(WEIGHTS_DIR)
            if f.endswith(".pt") and os.path.basename(f).lower() != "yolov8n.pt"
        ]
        if custom_weights:
            return os.path.join(WEIGHTS_DIR, custom_weights[0]), True
        default_local = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
        if os.path.exists(default_local):
            return default_local, False
    return "yolov8n.pt", False


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path, is_custom = _resolve_model_path()
    state["model"] = YOLO(model_path)
    state["is_custom_model"] = is_custom
    state["mp_locator"] = MediaPipeEyeLocator(
        task_model_path=MP_TASK_MODEL,
        enable_roll_compensation=DEFAULT_ENABLE_ROLL_COMP,
    )
    yield
    if state["mp_locator"] is not None:
        state["mp_locator"].close()


app = FastAPI(
    title="Strabismus Screening API",
    description="2D image screening API for preliminary strabismus triage (Flutter frontend integration).",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS so Flutter Web / Mobile clients can connect during development & staging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "eye_locator": (
            state["mp_locator"].backend_name
            if state["mp_locator"] is not None
            else "unavailable"
        ),
        "custom_yolo_weights": state["is_custom_model"],
    }


@app.post("/api/v1/strabismus/analyze")
async def analyze_strabismus(
    file: UploadFile = File(..., description="Facial image (JPG, PNG, WEBP)"),
    include_annotated_image: bool = Query(
        False,
        description="Set true to include base64-encoded annotated JPEG preview for Flutter UI",
    ),
    high_sensitivity: bool = Query(
        False,
        description="Set true to enable RollComp + CLAHE high-sensitivity triage mode",
    ),
    t_iris_asym: Optional[float] = Query(
        None, ge=0.01, le=0.50, description="Optional override for iris asymmetry threshold"
    ),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload a valid image.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image bytes.")

    effective_iris_asym = (
        t_iris_asym if t_iris_asym is not None else DEFAULT_T_IRIS_ASYM
    )
    use_clahe = True if high_sensitivity else DEFAULT_ENABLE_CLAHE_ROI

    # Toggle roll compensation dynamically if requested
    if state["mp_locator"] is not None:
        state["mp_locator"].enable_roll_compensation = high_sensitivity

    try:
        res = process_image_with_yolo(
            img_bgr=img_bgr,
            filename=file.filename or "upload.jpg",
            model=state["model"],
            mp_locator=state["mp_locator"],
            is_custom_model=state["is_custom_model"],
            T_horiz=DEFAULT_T_HORIZ,
            T_vert=DEFAULT_T_VERT,
            T_dominance=DEFAULT_T_DOMINANCE,
            T_shared_shift=DEFAULT_T_SHARED_SHIFT,
            T_iris_asym=effective_iris_asym,
            prefer_horizontal_when_clear=True,
            suppress_false_positives=True,
            enable_clahe_roi=use_clahe,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    payload = dict(res["api_payload"])
    payload["disclaimer"] = (
        "Preliminary screening triage output only; not a standalone clinical diagnosis."
    )

    if include_annotated_image:
        ok, buf = cv2.imencode(".jpg", res["annotated_image"])
        if ok:
            payload["annotated_image_base64"] = base64.b64encode(buf.tobytes()).decode(
                "utf-8"
            )

    return payload