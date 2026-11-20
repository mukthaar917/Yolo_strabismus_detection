YOLO Strabismus Detection 

Automated 2D Facial Image Triage Pipeline Combining YOLOv8 Eye Localization, MediaPipe FaceLandmarker Geometry, and FastAPI Mobile Integration

Tech Stack : Python, YOLOv8, MediaPipe, OpenCV, FastAPI


1. Executive Overview

The YOLO Strabismus Detection & Screening API automates preliminary ocular alignment screening from standard 2D frontal facial photographs[cite: 35]. Designed for integration with a Flutter mobile client, the system localizes periocular structures, measures corneal light reflex (CLR) offsets alongside normalized palpebral iris-position asymmetry, and classifies captures into three triage states: POSITIVE, NEEDS_REVIEW, and NEGATIVE.

Across a 25-image validation benchmark,the pipeline achieved 96.0% triage-aware accuracy (24/25) in its recommended production baseline and 100.0% triage-aware accuracy (25/25)in high-sensitivity mode, with a 0.0% false-positive alert rate (0/8) across all negative control image.


2. System Architecture & Algorithmic Pipeline

Each input image is standardized to a 512x512 resolution and processed through a five-stage deterministic computer vision pipeline in strabismus_yolo.py:

1. Dual-Backend Eye & Iris Localization: MediaPipe FaceLandmarke` extracts 478 facial and iris landmarks (palpebral fissure corners,    upper/lower eyelid margins, and iris contours), with fallback to YOLOv8 bounding-box detection (weights/yolov8n.p` or custom .pt weights) and Haar cascade facial geometry when landmarks are unavailable[cite: 35].
2. Pupil Centroid Moment Refinement: Within each iris-anchored eye region of interest (ROI), adaptive dark-core thresholding,  morphological opening, circularity-weighted contour scoring, and image-moment centroid calculation (m10/m00, m01/m00) determine pupil coordinates (x_pc, y_pc).
3. Corneal Light Reflex (CLR) Detection: Bounded specular highlight search near each pupil center isolates the compact corneal reflex  (x_clr, y_clr) while guarding against iris glare and lower-rim reflections.
4. Bilateral Geometric Asymmetry Computation: Computes horizontal and vertical Pupil-CLR disparities (Dx = |ΔX_L - ΔX_R|, Dy = |ΔY_L - ΔY_R|) fused with normalized palpebral iris-position asymmetry (IrisAsym H, IrisAsym V) and optional inter-canthal head-roll compensation.
5. 3-State Triage & Subtype Classification: Detections exceeding calibrated thresholds (T_horiz = 3, T_vert = 5, T_iris_asym = 0.105) are classified by axis (HORIZONTAL STRABISMUS vs. VERTICAL STRABISMUS), clinical subtype (ESOTROPIA, EXOTROPIA, or VERTICAL), and dominant eye, while borderline or low-contrast captures route to NEEDS_REVIEW.



3. Benchmark Evaluation & Accuracy Results



4. Mode Selection Notes

Default Mode: Production Baseline (T_iris_asym = 0.105, RollComp=False, CLAHE=False) is configured as the default API mode because it exceeds the 95% accuracy requirement (96.0%) with 0% false-positive 'POSITIVE` alerts and fewer retakes on clean negatives.
High-Sensitivity Mode: Can be enabled per request (high_sensitivity=true) when maximum positive capture (100.0%) is prioritized.



5 . Installation & Execution Guide

Environment Setup (Windows PowerShell)

powershell

git clone (https://github.com/mukthaar917/Yolo_strabismus_detection.git)

cd Yolo_strabismus_detection

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt


Running Offline Batch Evaluation

Place test images inside images (either flat or organized into category subdirectories) and execute;

powershell:

python strabismus_yolo.py


Annotated eye-overlay images are saved to output/results/ and per-image diagnostic reports are written to output/reports/.


