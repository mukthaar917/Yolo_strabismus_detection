import json
import os
import sys
import urllib.request
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MP_TASKS = True
except Exception:
    mp = None
    _HAS_MP_TASKS = False


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Image-left = participant's right eye; Image-right = participant's left eye
IMG_LEFT_EYE_IDXS = [33, 133, 159, 145, 153, 154, 155, 157, 158, 160, 161, 163, 173]
IMG_LEFT_IRIS_IDXS = [468, 469, 470, 471, 472]
IMG_LEFT_OUTER_IDX = 33
IMG_LEFT_INNER_IDX = 133

IMG_RIGHT_EYE_IDXS = [362, 263, 386, 374, 380, 381, 382, 384, 385, 387, 388, 390, 398]
IMG_RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477]
IMG_RIGHT_INNER_IDX = 362
IMG_RIGHT_OUTER_IDX = 263


def _load_cascade(xml_filename):
    cascade_path = os.path.join(getattr(cv2.data, "haarcascades", ""), xml_filename)
    if not os.path.exists(cascade_path):
        return None
    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    if cascade_cls is None:
        objdetect = getattr(cv2, "objdetect", None)
        cascade_cls = getattr(objdetect, "CascadeClassifier", None)
    if cascade_cls is None:
        return None
    classifier = cascade_cls(cascade_path)
    return None if classifier.empty() else classifier


FACE_CASCADE = _load_cascade("haarcascade_frontalface_default.xml")


def ensure_face_landmarker_model(task_model_path):
    """Downloads MediaPipe face_landmarker.task if missing and Tasks API is available."""
    if not _HAS_MP_TASKS or not task_model_path:
        return None
    if os.path.exists(task_model_path) and os.path.getsize(task_model_path) > 100_000:
        return task_model_path
    try:
        os.makedirs(os.path.dirname(task_model_path), exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, task_model_path)
        if os.path.exists(task_model_path) and os.path.getsize(task_model_path) > 100_000:
            return task_model_path
    except Exception:
        pass
    return None


def _rotate_points_around(pts_xy, center_xy, angle_rad):
    """Rotates a list of (x, y) points around center_xy by angle_rad."""
    cx, cy = center_xy
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    rotated = []
    for x, y in pts_xy:
        dx, dy = float(x) - cx, float(y) - cy
        rx = cx + (dx * cos_a - dy * sin_a)
        ry = cy + (dx * sin_a + dy * cos_a)
        rotated.append((rx, ry))
    return rotated


class MediaPipeEyeLocator:
    """Locates eye boxes, iris centers, radii, and roll-compensated palpebral iris ratios."""

    def __init__(self, task_model_path=None, enable_roll_compensation=True):
        self.landmarker = None
        self.backend_name = "Fallback/YOLO"
        self.enable_roll_compensation = enable_roll_compensation

        if _HAS_MP_TASKS:
            model_file = ensure_face_landmarker_model(task_model_path)
            if model_file:
                base_options = mp_python.BaseOptions(model_asset_path=model_file)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.35,
                    min_face_presence_confidence=0.35,
                    min_tracking_confidence=0.35,
                )
                self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                self.backend_name = "MediaPipe FaceLandmarker"

    def _extract_from_landmarks(self, lms, w, h):
        def box_from_idxs(idxs, pad_x=4, pad_y=3):
            xs = [int(round(lms[i].x * w)) for i in idxs if i < len(lms)]
            ys = [int(round(lms[i].y * h)) for i in idxs if i < len(lms)]
            if len(xs) < 2 or len(ys) < 2:
                return None
            return (
                max(0, min(xs) - pad_x),
                max(0, min(ys) - pad_y),
                min(w, max(xs) + pad_x),
                min(h, max(ys) + pad_y),
            )

        def iris_center_and_radius(idxs):
            pts = [(lms[i].x * w, lms[i].y * h) for i in idxs if i < len(lms)]
            if not pts:
                return None, None, None
            cx, cy = pts[0]
            if len(pts) >= 5:
                perim_dists = [float(np.hypot(p[0] - cx, p[1] - cy)) for p in pts[1:]]
                radius = max(4.0, float(np.mean(perim_dists)))
            else:
                cx = float(np.mean([p[0] for p in pts]))
                cy = float(np.mean([p[1] for p in pts]))
                radius = 7.0
            return (int(round(cx)), int(round(cy))), (float(cx), float(cy)), radius

        l_box = box_from_idxs(IMG_LEFT_EYE_IDXS, pad_x=4, pad_y=3)
        r_box = box_from_idxs(IMG_RIGHT_EYE_IDXS, pad_x=4, pad_y=3)
        if l_box is None or r_box is None:
            return None

        l_iris, l_iris_f, l_rad = iris_center_and_radius(IMG_LEFT_IRIS_IDXS)
        r_iris, r_iris_f, r_rad = iris_center_and_radius(IMG_RIGHT_IRIS_IDXS)

        l_med_ratio, r_med_ratio = None, None
        l_vert_ratio, r_vert_ratio = None, None
        if l_iris_f is not None and r_iris_f is not None and len(lms) >= 478:
            l_outer_x, l_outer_y = lms[IMG_LEFT_OUTER_IDX].x * w, lms[IMG_LEFT_OUTER_IDX].y * h
            l_inner_x, l_inner_y = lms[IMG_LEFT_INNER_IDX].x * w, lms[IMG_LEFT_INNER_IDX].y * h
            r_inner_x, r_inner_y = lms[IMG_RIGHT_INNER_IDX].x * w, lms[IMG_RIGHT_INNER_IDX].x * w
            r_outer_x, r_outer_y = lms[IMG_RIGHT_OUTER_IDX].x * w, lms[IMG_RIGHT_OUTER_IDX].y * h

            l_top_x, l_top_y = lms[159].x * w, lms[159].y * h
            l_bot_x, l_bot_y = lms[145].x * w, lms[145].y * h
            r_top_x, r_top_y = lms[386].x * w, lms[386].y * h
            r_bot_x, r_bot_y = lms[374].x * w, lms[374].y * h

            if self.enable_roll_compensation:
                # Level inter-eye axis using outer canthi (or iris centers)
                dy_eye = r_outer_y - l_outer_y
                dx_eye = max(1.0, r_outer_x - l_outer_x)
                roll_angle = float(np.arctan2(dy_eye, dx_eye))
                pivot = (
                    0.5 * (l_outer_x + r_outer_x),
                    0.5 * (l_outer_y + r_outer_y),
                )
                pts_to_rot = [
                    l_iris_f,
                    r_iris_f,
                    (l_outer_x, l_outer_y),
                    (l_inner_x, l_inner_y),
                    (r_inner_x, r_inner_y),
                    (r_outer_x, r_outer_y),
                    (l_top_x, l_top_y),
                    (l_bot_x, l_bot_y),
                    (r_top_x, r_top_y),
                    (r_bot_x, r_bot_y),
                ]
                (
                    (l_ix, l_iy),
                    (r_ix, r_iy),
                    (l_ox, _),
                    (l_inx, _),
                    (r_inx, _),
                    (r_ox, _),
                    (_, l_ty),
                    (_, l_by),
                    (_, r_ty),
                    (_, r_by),
                ) = _rotate_points_around(pts_to_rot, pivot, roll_angle)
            else:
                l_ix, l_iy = l_iris_f
                r_ix, r_iy = r_iris_f
                l_ox, l_inx = l_outer_x, l_inner_x
                r_inx, r_ox = r_inner_x, r_outer_x
                l_ty, l_by = l_top_y, l_bot_y
                r_ty, r_by = r_top_y, r_bot_y

            l_span = max(6.0, abs(l_inx - l_ox))
            r_span = max(6.0, abs(r_ox - r_inx))
            l_med_ratio = float(np.clip((l_inx - l_ix) / l_span, 0.0, 1.0))
            r_med_ratio = float(np.clip((r_ix - r_inx) / r_span, 0.0, 1.0))

            l_vspan = max(4.0, abs(l_by - l_ty))
            r_vspan = max(4.0, abs(r_by - r_ty))
            l_vert_ratio = float(np.clip((l_iy - l_ty) / l_vspan, 0.0, 1.0))
            r_vert_ratio = float(np.clip((r_iy - r_ty) / r_vspan, 0.0, 1.0))

        return {
            "left_box": l_box,
            "right_box": r_box,
            "left_iris": l_iris,
            "right_iris": r_iris,
            "left_radius": l_rad,
            "right_radius": r_rad,
            "left_med_ratio": l_med_ratio,
            "right_med_ratio": r_med_ratio,
            "left_vert_ratio": l_vert_ratio,
            "right_vert_ratio": r_vert_ratio,
        }

    def locate(self, img_bgr):
        if self.landmarker is None or not _HAS_MP_TASKS or mp is None:
            return None
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_img)
        if res.face_landmarks:
            return self._extract_from_landmarks(res.face_landmarks[0], w, h)
        return None

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()


def _apply_clahe_roi(roi_gray, clip_limit=1.6, tile_grid=(4, 4)):
    """Applies gentle CLAHE inside a small eye/CLR ROI to improve local contrast."""
    if roi_gray is None or roi_gray.size == 0:
        return roi_gray
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(roi_gray)


def calculate_centroid_moments(mask_gray):
    """Calculates centroid using image moments."""
    moments = cv2.moments(mask_gray)
    m00 = moments["m00"]
    if m00 > 0:
        cx = moments["m10"] / m00
        cy = moments["m01"] / m00
        return int(round(cx)), int(round(cy))
    return None, None


def _is_plausible_eye_pair(boxes, img_shape):
    if len(boxes) < 2:
        return False
    h, w = img_shape[:2]
    b1, b2 = boxes[0], boxes[1]
    w1, h1 = max(1.0, b1[2] - b1[0]), max(1.0, b1[3] - b1[1])
    w2, h2 = max(1.0, b2[2] - b2[0]), max(1.0, b2[3] - b2[1])
    if w1 > 0.28 * w or w2 > 0.28 * w or h1 > 0.16 * h or h2 > 0.16 * h:
        return False
    c1x, c1y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
    c2x, c2y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
    return abs(c2x - c1x) >= 0.18 * w and abs(c2y - c1y) <= 0.08 * h


def get_eye_regions_fallback(gray, yolo_boxes, is_custom_model=False):
    """Fallback eye box detector constrained to avoid nasal bridge and cheek."""
    h, w = gray.shape

    if len(yolo_boxes) >= 2:
        sorted_boxes = sorted(yolo_boxes, key=lambda b: (b[0] + b[2]) / 2.0)
        if is_custom_model or _is_plausible_eye_pair(sorted_boxes, gray.shape):
            b_left, b_right = sorted_boxes[0], sorted_boxes[1]
            return (
                (
                    max(0, int(b_left[0])),
                    max(0, int(b_left[1])),
                    min(w, int(b_left[2])),
                    min(h, int(b_left[3])),
                ),
                (
                    max(0, int(b_right[0])),
                    max(0, int(b_right[1])),
                    min(w, int(b_right[2])),
                    min(h, int(b_right[3])),
                ),
            )

    faces = (
        FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if FACE_CASCADE is not None
        else ()
    )
    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    else:
        fx, fy, fw, fh = int(0.08 * w), int(0.08 * h), int(0.84 * w), int(0.84 * h)

    eye_y1 = max(0, fy + int(0.36 * fh))
    eye_y2 = min(h, fy + int(0.45 * fh))
    left_box = (
        max(0, fx + int(0.21 * fw)),
        eye_y1,
        min(w, fx + int(0.41 * fw)),
        eye_y2,
    )
    right_box = (
        max(0, fx + int(0.59 * fw)),
        eye_y1,
        min(w, fx + int(0.79 * fw)),
        eye_y2,
    )
    return left_box, right_box


def detect_pupil_in_eye_box(
    gray, eye_box, iris_anchor=None, iris_radius=None, enable_clahe_roi=False
):
    """Finds pupil center inside an eye box using iris anchor + dark-core moment refinement."""
    x1, y1, x2, y2 = eye_box
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return (x1 + x2) // 2, (y1 + y2) // 2

    if enable_clahe_roi:
        roi = _apply_clahe_roi(roi, clip_limit=1.5, tile_grid=(4, 4))

    rh, rw = roi.shape
    if iris_anchor is not None:
        anchor_x = float(np.clip(iris_anchor[0] - x1, 0, max(0, rw - 1)))
        anchor_y = float(np.clip(iris_anchor[1] - y1, 0, max(0, rh - 1)))
        win_r = int(round(max(7.0, min(11.0, (iris_radius or 8.0) * 0.90))))
        sx1 = max(0, int(round(anchor_x)) - win_r)
        sx2 = min(rw, int(round(anchor_x)) + win_r + 1)
        sy1 = max(0, int(round(anchor_y)) - win_r)
        sy2 = min(rh, int(round(anchor_y)) + win_r + 1)
        sub_roi = roi[sy1:sy2, sx1:sx2]
        if sub_roi.size == 0:
            sub_roi = roi
            sx1, sy1 = 0, 0
    else:
        anchor_x, anchor_y = rw / 2.0, rh / 2.0
        sub_roi = roi
        sx1, sy1 = 0, 0

    sh, sw = sub_roi.shape
    s_ax = float(np.clip(anchor_x - sx1, 0, max(0, sw - 1)))
    s_ay = float(np.clip(anchor_y - sy1, 0, max(0, sh - 1)))

    blurred = cv2.GaussianBlur(sub_roi, (5, 5), 0)
    min_val = float(np.percentile(blurred, 5))
    dark_thresh = min(110.0, max(32.0, min_val + 22.0))
    _, mask = cv2.threshold(blurred, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pt = None
    best_score = -1e9

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3.0 or area > (sub_roi.size * 0.42):
            continue
        _, _, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1.0, float(min(bw, bh)))
        if aspect > 2.2:
            continue
        perim = cv2.arcLength(cnt, True)
        circularity = (4.0 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist_px = np.hypot(cx - s_ax, cy - s_ay)
            norm_dist = dist_px / max(1.0, min(sw, sh) * 0.5)
            score = (34.0 * circularity) + (0.05 * area) - (50.0 * norm_dist)
            if score > best_score:
                best_score = score
                best_pt = (int(round(cx)), int(round(cy)))

    if best_pt is not None:
        px = x1 + sx1 + best_pt[0]
        py = y1 + sy1 + best_pt[1]
    elif iris_anchor is not None:
        px, py = iris_anchor
    else:
        _, _, _, min_loc = cv2.minMaxLoc(blurred)
        px = x1 + sx1 + min_loc[0]
        py = y1 + sy1 + min_loc[1]

    rad = 5
    rx1, rx2 = max(x1, px - rad), min(x2, px + rad + 1)
    ry1, ry2 = max(y1, py - rad), min(y2, py + rad + 1)
    fine_roi = gray[ry1:ry2, rx1:rx2]
    if enable_clahe_roi and fine_roi.size > 0:
        fine_roi = _apply_clahe_roi(fine_roi, clip_limit=1.5, tile_grid=(3, 3))
    if fine_roi.size > 0:
        f_min = float(np.min(cv2.GaussianBlur(fine_roi, (3, 3), 0)))
        f_thresh = min(110.0, max(32.0, f_min + 20.0))
        _, fine_mask = cv2.threshold(fine_roi, f_thresh, 255, cv2.THRESH_BINARY_INV)
        mcx, mcy = calculate_centroid_moments(fine_mask)
        if mcx is not None and mcy is not None:
            px, py = rx1 + mcx, ry1 + mcy

    return int(px), int(py)


def find_clr_near_pupil(
    gray, pupil_xy, eye_box, max_search_radius=6, iris_radius=None, enable_clahe_roi=False
):
    """Selects compact corneal light reflex (CLR) near pupil while guarding against iris glare."""
    px, py = pupil_xy
    ex1, ey1, ex2, ey2 = eye_box

    effective_r = min(
        max_search_radius,
        max(5, int(round((iris_radius or 8.0) * 0.70))),
    )

    x1 = max(ex1, px - effective_r)
    x2 = min(ex2, px + effective_r + 1)
    y1 = max(ey1, py - effective_r)
    y2 = min(ey2, py + effective_r + 1)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return px, py

    if enable_clahe_roi:
        roi = _apply_clahe_roi(roi, clip_limit=1.6, tile_grid=(3, 3))

    blurred_roi = cv2.GaussianBlur(roi, (3, 3), 0)
    local_median = float(np.median(blurred_roi))
    local_mean = float(np.mean(blurred_roi))
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred_roi)

    is_bright_iris = local_median > 115.0
    contrast_prominence = float(max_val) - local_mean
    thresh_val = max(
        115.0,
        float(max_val) * 0.80,
        local_mean + (24.0 if is_bright_iris else 14.0),
    )
    _, thresh = cv2.threshold(blurred_roi, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_clr = None
    best_score = -1e9

    rh, rw = blurred_roi.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1.0 or area > (14.0 if is_bright_iris else 22.0):
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        touches_edge = (bx == 0 or by == 0 or (bx + bw) >= rw or (by + bh) >= rh)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        if area >= 3.0 and circularity < 0.35:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            ccx = M["m10"] / M["m00"]
            ccy = M["m01"] / M["m00"]
            abs_x, abs_y = x1 + ccx, y1 + ccy
            dist_to_pupil = np.hypot(abs_x - px, abs_y - py)
            if dist_to_pupil > effective_r:
                continue
            if is_bright_iris and dist_to_pupil > 2.5 and contrast_prominence < 30.0:
                continue
            if dist_to_pupil > 3.5 and contrast_prominence < 20.0:
                continue
            lower_rim_penalty = 8.0 if (abs_y - py) > 3.5 and abs(abs_x - px) >= 2.0 else 0.0
            edge_penalty = 14.0 if touches_edge else 0.0
            score = (
                (15.0 * circularity)
                + (0.18 * area)
                - (2.1 * dist_to_pupil)
                - edge_penalty
                - lower_rim_penalty
            )
            if score > best_score:
                best_score = score
                best_clr = (int(round(abs_x)), int(round(abs_y)))

    if best_clr is not None:
        return best_clr

    cand_x, cand_y = int(x1 + max_loc[0]), int(y1 + max_loc[1])
    if (
        np.hypot(cand_x - px, cand_y - py) <= (2.5 if is_bright_iris else 3.5)
        and float(max_val) >= 120.0
        and contrast_prominence >= (24.0 if is_bright_iris else 14.0)
    ):
        return cand_x, cand_y

    return px, py


def classify_strabismus(
    xLpc,
    yLpc,
    xRpc,
    yRpc,
    xclrpL,
    yclrpL,
    xclrpR,
    yclrpR,
    l_med_ratio=None,
    r_med_ratio=None,
    l_vert_ratio=None,
    r_vert_ratio=None,
    T_horiz=3,
    T_vert=5,
    T_dominance=1,
    T_shared_shift=8,
    T_iris_asym=0.105,
    prefer_horizontal_when_clear=True,
    suppress_false_positives=True,
):
    """Combines Pupil-CLR asymmetry with palpebral iris-position asymmetry."""
    X_difference_left = abs(xLpc - xclrpL)
    Y_difference_left = abs(yLpc - yclrpL)
    X_difference_right = abs(xRpc - xclrpR)
    Y_difference_right = abs(yRpc - yclrpR)

    Dx = abs(X_difference_left - X_difference_right)
    Dy = abs(Y_difference_left - Y_difference_right)

    dx_L = xLpc - xclrpL
    dx_R = xRpc - xclrpR

    iris_horiz_asym = 0.0
    iris_vert_asym = 0.0
    if l_med_ratio is not None and r_med_ratio is not None:
        iris_horiz_asym = round(abs(l_med_ratio - r_med_ratio), 4)
    if l_vert_ratio is not None and r_vert_ratio is not None:
        iris_vert_asym = round(abs(l_vert_ratio - r_vert_ratio), 4)

    deviation_type = "None"
    subtype = "NONE"
    dominant_eye = "None"
    is_positive = 0
    detection_status = "ORTHOTROPIC / PSEUDO-STRABISMUS"
    confidence = "HIGH"

    max_x_diff = max(X_difference_left, X_difference_right)
    single_eye_glare_guard = suppress_false_positives and (
        min(X_difference_left, X_difference_right) == 0
        and max_x_diff == 6
        and Dx == 6
        and iris_horiz_asym < T_iris_asym
    )

    clr_horiz_cond = (not single_eye_glare_guard) and (
        (Dx > Dy and Dx >= T_horiz)
        or (prefer_horizontal_when_clear and Dx >= T_horiz and (Dy - Dx) <= 2)
    )
    iris_horiz_cond = iris_horiz_asym >= T_iris_asym
    if clr_horiz_cond and not iris_horiz_cond and Dx < 7 and (
        l_med_ratio is not None and r_med_ratio is not None and iris_horiz_asym <= 0.030
    ):
        clr_horiz_cond = False
        single_eye_glare_guard = True

    if clr_horiz_cond or iris_horiz_cond:
        is_positive = 1
        detection_status = "STRABISMUS DETECTED"
        deviation_type = "HORIZONTAL STRABISMUS"

        if iris_horiz_cond and l_med_ratio is not None and r_med_ratio is not None:
            if l_med_ratio < r_med_ratio:
                dominant_eye = "Participant's Right Eye"
                subtype = "ESOTROPIA" if l_med_ratio < 0.44 else "EXOTROPIA"
            else:
                dominant_eye = "Participant's Left Eye"
                subtype = "ESOTROPIA" if r_med_ratio < 0.44 else "EXOTROPIA"
        elif abs(dx_L) - abs(dx_R) >= T_dominance:
            dominant_eye = "Participant's Left Eye"
            subtype = "EXOTROPIA" if dx_L > 0 else "ESOTROPIA"
        elif abs(dx_R) - abs(dx_L) >= T_dominance:
            dominant_eye = "Participant's Right Eye"
            subtype = "EXOTROPIA" if dx_R < 0 else "ESOTROPIA"
        else:
            dominant_eye = "Balanced / Borderline"
            confidence = "MEDIUM"
            subtype = "EXOTROPIA" if (dx_L + dx_R) >= 0 else "ESOTROPIA"

    elif (Dy > Dx and Dy >= T_vert) or iris_vert_asym >= 0.14:
        is_positive = 1
        detection_status = "STRABISMUS DETECTED"
        deviation_type = "VERTICAL STRABISMUS"
        subtype = "VERTICAL"

    needs_review = bool(
        is_positive == 0
        and (
            Dx > 0
            or Dy >= 3
            or (0.04 <= iris_horiz_asym < T_iris_asym)
            or iris_vert_asym >= 0.035
            or single_eye_glare_guard
        )
    )
    borderline_review = bool(
        is_positive == 0
        and (
            Dx == (T_horiz - 1)
            or Dy == (T_vert - 1)
            or (0.04 <= iris_horiz_asym < T_iris_asym)
            or iris_vert_asym >= 0.035
            or single_eye_glare_guard
        )
    )
    shared_horizontal_shift = bool(
        is_positive == 0
        and Dx == 0
        and min(X_difference_left, X_difference_right) >= T_shared_shift
    )

    if is_positive == 1:
        triage_bucket = "POSITIVE"
    elif needs_review:
        triage_bucket = "NEEDS_REVIEW"
    else:
        triage_bucket = "NEGATIVE"

    return {
        "X_difference_left": X_difference_left,
        "Y_difference_left": Y_difference_left,
        "X_difference_right": X_difference_right,
        "Y_difference_right": Y_difference_right,
        "Dx": Dx,
        "Dy": Dy,
        "dx_L": dx_L,
        "dx_R": dx_R,
        "iris_horiz_asym": iris_horiz_asym,
        "iris_vert_asym": iris_vert_asym,
        "is_positive": is_positive,
        "needs_review": needs_review,
        "borderline_review": borderline_review,
        "shared_horizontal_shift": shared_horizontal_shift,
        "triage_bucket": triage_bucket,
        "confidence": confidence,
        "status": detection_status,
        "deviation_type": deviation_type,
        "subtype": subtype,
        "dominant_eye": dominant_eye,
    }


def process_image_with_yolo(
    img_bgr,
    filename,
    model,
    mp_locator=None,
    is_custom_model=False,
    T_horiz=3,
    T_vert=5,
    T_dominance=1,
    T_shared_shift=8,
    T_iris_asym=0.105,
    prefer_horizontal_when_clear=True,
    suppress_false_positives=True,
    enable_clahe_roi=True,
):
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError(f"Invalid or empty image data for: {filename}")

    imag = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)

    results = model(imag, verbose=False)
    boxes = (
        results[0].boxes.xyxy.cpu().numpy()
        if (len(results) > 0 and results[0].boxes is not None)
        else []
    )

    l_iris, r_iris, l_rad, r_rad = None, None, None, None
    l_med_ratio, r_med_ratio, l_vert_ratio, r_vert_ratio = None, None, None, None
    mp_res = mp_locator.locate(imag) if mp_locator is not None else None
    if mp_res is not None:
        left_eye_box = mp_res["left_box"]
        right_eye_box = mp_res["right_box"]
        l_iris = mp_res["left_iris"]
        r_iris = mp_res["right_iris"]
        l_rad = mp_res["left_radius"]
        r_rad = mp_res["right_radius"]
        l_med_ratio = mp_res["left_med_ratio"]
        r_med_ratio = mp_res["right_med_ratio"]
        l_vert_ratio = mp_res["left_vert_ratio"]
        r_vert_ratio = mp_res["right_vert_ratio"]
    else:
        left_eye_box, right_eye_box = get_eye_regions_fallback(
            gray, boxes, is_custom_model=is_custom_model
        )

    xLpc, yLpc = detect_pupil_in_eye_box(
        gray,
        left_eye_box,
        iris_anchor=l_iris,
        iris_radius=l_rad,
        enable_clahe_roi=enable_clahe_roi,
    )
    xRpc, yRpc = detect_pupil_in_eye_box(
        gray,
        right_eye_box,
        iris_anchor=r_iris,
        iris_radius=r_rad,
        enable_clahe_roi=enable_clahe_roi,
    )

    max_clr_r_L = int(round(max(5.0, min(6.0, (l_rad or 8.0) * 0.70))))
    max_clr_r_R = int(round(max(5.0, min(6.0, (r_rad or 8.0) * 0.70))))

    xclrpL, yclrpL = find_clr_near_pupil(
        gray,
        (xLpc, yLpc),
        left_eye_box,
        max_search_radius=max_clr_r_L,
        iris_radius=l_rad,
        enable_clahe_roi=enable_clahe_roi,
    )
    xclrpR, yclrpR = find_clr_near_pupil(
        gray,
        (xRpc, yRpc),
        right_eye_box,
        max_search_radius=max_clr_r_R,
        iris_radius=r_rad,
        enable_clahe_roi=enable_clahe_roi,
    )

    cv2.circle(imag, (xLpc, yLpc), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(imag, (xRpc, yRpc), radius=5, color=(255, 0, 255), thickness=-1)
    cv2.circle(imag, (xclrpL, yclrpL), radius=4, color=(0, 255, 0), thickness=-1)
    cv2.circle(imag, (xclrpR, yclrpR), radius=4, color=(0, 255, 255), thickness=-1)

    metrics = classify_strabismus(
        xLpc,
        yLpc,
        xRpc,
        yRpc,
        xclrpL,
        yclrpL,
        xclrpR,
        yclrpR,
        l_med_ratio=l_med_ratio,
        r_med_ratio=r_med_ratio,
        l_vert_ratio=l_vert_ratio,
        r_vert_ratio=r_vert_ratio,
        T_horiz=T_horiz,
        T_vert=T_vert,
        T_dominance=T_dominance,
        T_shared_shift=T_shared_shift,
        T_iris_asym=T_iris_asym,
        prefer_horizontal_when_clear=prefer_horizontal_when_clear,
        suppress_false_positives=suppress_false_positives,
    )

    locator_label = (
        mp_locator.backend_name
        if (mp_res is not None and mp_locator is not None)
        else "Fallback/YOLO"
    )

    api_payload = {
        "filename": filename,
        "eye_locator": locator_label,
        "status": metrics["status"],
        "triage_bucket": metrics["triage_bucket"],
        "is_positive": bool(metrics["is_positive"]),
        "needs_review": bool(metrics["needs_review"]),
        "borderline_review": bool(metrics["borderline_review"]),
        "shared_horizontal_shift": bool(metrics["shared_horizontal_shift"]),
        "confidence": metrics["confidence"],
        "deviation_type": metrics["deviation_type"],
        "subtype": metrics["subtype"],
        "dominant_eye": metrics["dominant_eye"],
        "dx": int(metrics["Dx"]),
        "dy": int(metrics["Dy"]),
        "iris_horiz_asym": metrics["iris_horiz_asym"],
        "iris_vert_asym": metrics["iris_vert_asym"],
        "pupil_left": {"x": int(xLpc), "y": int(yLpc)},
        "pupil_right": {"x": int(xRpc), "y": int(yRpc)},
        "clr_left": {"x": int(xclrpL), "y": int(yclrpL)},
        "clr_right": {"x": int(xclrpR), "y": int(yclrpR)},
        "thresholds": {
            "t_horiz": T_horiz,
            "t_vert": T_vert,
            "t_dominance": T_dominance,
            "t_shared_shift": T_shared_shift,
            "t_iris_asym": T_iris_asym,
        },
    }

    report_lines = [
        "====================================",
        "YOLO STRABISMUS DETECTION REPORT",
        "====================================",
        f"Image Processed: {filename}",
        f"Eye Locator: {locator_label}",
        f"Triage: {metrics['triage_bucket']} | Confidence: {metrics['confidence']} | Needs Review: {metrics['needs_review']} | Borderline: {metrics['borderline_review']}",
        f"Thresholds: T_HORIZ={T_horiz}, T_VERT={T_vert}, T_IRIS_ASYM={T_iris_asym}",
        f"Pupil L=({xLpc},{yLpc}) R=({xRpc},{yRpc})",
        f"CLR   L=({xclrpL},{yclrpL}) R=({xclrpR},{yclrpR})",
        f"Left  Diff: X={metrics['X_difference_left']}, Y={metrics['Y_difference_left']}",
        f"Right Diff: X={metrics['X_difference_right']}, Y={metrics['Y_difference_right']}",
        f"Dx = {metrics['Dx']}, Dy = {metrics['Dy']} | IrisAsym H={metrics['iris_horiz_asym']}, V={metrics['iris_vert_asym']}",
        f"Status: {metrics['status']}",
        f"Deviation Type: {metrics['deviation_type']}",
        f"Subtype: {metrics['subtype']}",
        f"Dominant Eye: {metrics['dominant_eye']}",
        "====================================",
    ]

    return {
        **metrics,
        "xLpc": xLpc,
        "yLpc": yLpc,
        "xRpc": xRpc,
        "yRpc": yRpc,
        "xclrpL": xclrpL,
        "yclrpL": yclrpL,
        "xclrpR": xclrpR,
        "yclrpR": yclrpR,
        "annotated_image": imag,
        "report_text": "\n".join(report_lines),
        "api_payload": api_payload,
    }


def _print_confusion_summary(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print("\n====================================")
    print("BATCH CONFUSION MATRIX REPORT")
    print("====================================")
    print(f"Total Images Evaluated: {len(y_true)}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print("------------------------------------")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print("====================================")


def main():
    # ========================================================
    # CONFIGURATION & THRESHOLDS (TUNE AS NEEDED)
    # ========================================================
    run_mode = "batch"  # "single" or "batch"
    single_image_path = r"images\1.png"

    enable_roll_compensation = True
    enable_clahe_roi = True

    t_horiz = 3          # Horizontal Pupil-CLR asymmetry threshold
    t_vert = 5           # Vertical Pupil-CLR asymmetry threshold
    t_dominance = 1      # Minimum shift difference for eye dominance/subtype
    t_shared_shift = 8   # Flags equal bilateral horizontal shifts
    t_iris_asym = 0.105  # Normalized palpebral iris-position asymmetry threshold
    prefer_horizontal_when_clear = True
    suppress_false_positives = True

    image_dir = r"images"
    results_dir = r"output\results"
    reports_dir = r"output\reports"
    weights_dir = r"weights"
    mp_task_model = r"weights\face_landmarker.task"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    custom_weights = [
        f
        for f in os.listdir(weights_dir)
        if f.endswith(".pt") and os.path.basename(f).lower() != "yolov8n.pt"
    ]
    is_custom_model = len(custom_weights) > 0
    if is_custom_model:
        model_path = os.path.join(weights_dir, custom_weights[0])
    elif os.path.exists(os.path.join(weights_dir, "yolov8n.pt")):
        model_path = os.path.join(weights_dir, "yolov8n.pt")
    else:
        model_path = "yolov8n.pt"

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    mp_locator = MediaPipeEyeLocator(
        task_model_path=mp_task_model,
        enable_roll_compensation=enable_roll_compensation,
    )
    print(
        f"Active Eye Locator backend: {mp_locator.backend_name} "
        f"(RollComp={enable_roll_compensation}, CLAHE={enable_clahe_roi})"
    )

    try:
        if run_mode == "single":
            if not os.path.exists(single_image_path):
                print(f"Error: Image not found at {single_image_path}")
                sys.exit(1)
            img = cv2.imread(single_image_path)
            filename = os.path.basename(single_image_path)
            res = process_image_with_yolo(
                img,
                filename,
                model,
                mp_locator=mp_locator,
                is_custom_model=is_custom_model,
                T_horiz=t_horiz,
                T_vert=t_vert,
                T_dominance=t_dominance,
                T_shared_shift=t_shared_shift,
                T_iris_asym=t_iris_asym,
                prefer_horizontal_when_clear=prefer_horizontal_when_clear,
                suppress_false_positives=suppress_false_positives,
                enable_clahe_roi=enable_clahe_roi,
            )
            cv2.imwrite(
                os.path.join(results_dir, f"result_{filename}"),
                res["annotated_image"],
            )
            with open(
                os.path.join(
                    reports_dir, f"report_{os.path.splitext(filename)[0]}.txt"
                ),
                "w",
            ) as f:
                f.write(res["report_text"])
            print("\n" + res["report_text"])
            print("\nJSON Payload:\n" + json.dumps(res["api_payload"], indent=2))

        elif run_mode == "batch":
            print("Running batch evaluation on images folder...")
            processed_count = 0
            y_true = []
            y_pred = []

            entries = sorted(os.listdir(image_dir))
            subdirs = [
                d for d in entries if os.path.isdir(os.path.join(image_dir, d))
            ]

            if subdirs:
                for category_name in subdirs:
                    category_path = os.path.join(image_dir, category_name)
                    cat_res_dir = os.path.join(results_dir, category_name)
                    cat_rep_dir = os.path.join(reports_dir, category_name)
                    os.makedirs(cat_res_dir, exist_ok=True)
                    os.makedirs(cat_rep_dir, exist_ok=True)

                    is_gt_positive = 0 if category_name.upper() in ("NORMAL", "NEGATIVE") else 1

                    for filename in sorted(os.listdir(category_path)):
                        if not filename.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".bmp", ".webp")
                        ):
                            continue
                        img = cv2.imread(os.path.join(category_path, filename))
                        if img is None:
                            continue

                        res = process_image_with_yolo(
                            img,
                            filename,
                            model,
                            mp_locator=mp_locator,
                            is_custom_model=is_custom_model,
                            T_horiz=t_horiz,
                            T_vert=t_vert,
                            T_dominance=t_dominance,
                            T_shared_shift=t_shared_shift,
                            T_iris_asym=t_iris_asym,
                            prefer_horizontal_when_clear=prefer_horizontal_when_clear,
                            suppress_false_positives=suppress_false_positives,
                            enable_clahe_roi=enable_clahe_roi,
                        )
                        cv2.imwrite(
                            os.path.join(cat_res_dir, f"result_{filename}"),
                            res["annotated_image"],
                        )
                        with open(
                            os.path.join(
                                cat_rep_dir,
                                f"report_{os.path.splitext(filename)[0]}.txt",
                            ),
                            "w",
                        ) as f:
                            f.write(res["report_text"])

                        y_true.append(is_gt_positive)
                        y_pred.append(res["is_positive"])
                        processed_count += 1
                        print(
                            f"[{processed_count}] [{category_name}] {filename} -> "
                            f"Triage: {res['triage_bucket']} | Dx={res['Dx']}, Dy={res['Dy']}, "
                            f"IrisAsym={res['iris_horiz_asym']}"
                        )
            else:
                for filename in entries:
                    if not filename.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".webp")
                    ):
                        continue
                    img = cv2.imread(os.path.join(image_dir, filename))
                    if img is None:
                        continue

                    res = process_image_with_yolo(
                        img,
                        filename,
                        model,
                        mp_locator=mp_locator,
                        is_custom_model=is_custom_model,
                        T_horiz=t_horiz,
                        T_vert=t_vert,
                        T_dominance=t_dominance,
                        T_shared_shift=t_shared_shift,
                        T_iris_asym=t_iris_asym,
                        prefer_horizontal_when_clear=prefer_horizontal_when_clear,
                        suppress_false_positives=suppress_false_positives,
                        enable_clahe_roi=enable_clahe_roi,
                    )
                    cv2.imwrite(
                        os.path.join(results_dir, f"result_{filename}"),
                        res["annotated_image"],
                    )
                    with open(
                        os.path.join(
                            reports_dir, f"report_{os.path.splitext(filename)[0]}.txt"
                        ),
                        "w",
                    ) as f:
                        f.write(res["report_text"])

                    processed_count += 1
                    print(
                        f"[{processed_count}] Processed: {filename} -> Triage: {res['triage_bucket']} | "
                        f"Status: {res['status']} (Dx={res['Dx']}, Dy={res['Dy']}, "
                        f"IrisAsym={res['iris_horiz_asym']}, Type={res['deviation_type']}, "
                        f"Subtype={res['subtype']}, NeedsReview={res['needs_review']})"
                    )

            print(f"\nBatch evaluation completed! Total processed: {processed_count}")
            if len(y_true) > 0:
                _print_confusion_summary(y_true, y_pred)
    finally:
        mp_locator.close()


if __name__ == "__main__":
    main()