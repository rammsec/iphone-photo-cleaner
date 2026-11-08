"""
Photo & Video Quality and Clutter Analyzer for iPhone 16 Pro Max / 18 Pro Max.
Handles:
- HEIC / ProRAW / JPEG / PNG / MOV / MP4 parsing
- Live Photo pair detection (.HEIC + .MOV sharing same stem)
- Perceptual Hashing (pHash, dHash, aHash, colorhash) for cross-phone duplicate & burst clustering
- Quality Scoring (0-100):
  * Sharpness / Blur detection (Laplacian variance + Sobel gradient)
  * Face & Eye detection (OpenCV Haar Cascades for frontal face + open eyes)
  * Exposure / Lighting balance (Histogram spread & clipping penalties)
  * Resolution bonus
- Junk / Clutter Detection:
  * Screenshots (EXIF UserComment / PNG screen dimensions / UI flat regions)
  * WhatsApp Forwards (missing camera EXIF, WhatsApp filename patterns, standard WA compression sizes)
  * Banners & Flyers (extreme aspect ratios, high text/edge density over flat backgrounds)
  * Wish Messages ("Happy New Year", "Good Morning", festival greetings: high color saturation + dense text overlays + no camera EXIF)
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image, ExifTags

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


PHOTO_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp", ".dng", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v"}

# Standard iPhone screen resolutions (including 16 Pro Max, 18 Pro Max, and recent Pro models)
IPHONE_SCREENSHOT_RESOLUTIONS = {
    (1320, 2868), (2868, 1320),  # 16 Pro Max / 18 Pro Max
    (1344, 2992), (2992, 1344),  # 18 Pro Max (6.9"+)
    (1290, 2796), (2796, 1290),  # 14/15 Pro Max
    (1284, 2778), (2778, 1284),  # 12/13 Pro Max
    (1206, 2622), (2622, 1206),  # 16 Pro
    (1179, 2556), (2556, 1179),  # 14/15 Pro
    (1170, 2532), (2532, 1170),  # 12/13/14
}

# Initialize OpenCV cascades for face and eye quality scoring (with safe fallback across OpenCV 4.x / 5.x)
_FACE_CASCADE = None
_EYE_CASCADE = None
if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
    try:
        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    except Exception:
        pass


@dataclass
class MediaItem:
    id: str
    filepath: str
    filename: str
    rel_path: str
    media_type: str  # "photo" or "video"
    file_size: int
    width: int = 0
    height: int = 0
    capture_time: float = 0.0
    capture_time_iso: str = ""
    camera_model: str = ""
    has_camera_exif: bool = False
    is_live_photo: bool = False
    live_video_path: Optional[str] = None
    live_video_size: int = 0

    # Hashes for similarity
    phash: str = ""
    dhash: str = ""
    colorhash: str = ""
    sha256: str = ""

    # Quality metrics (0 - 100)
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    face_score: float = 0.0
    faces_count: int = 0
    eyes_Detected: int = 0
    is_blurry: bool = False

    # Clutter / Junk classification
    junk_category: Optional[str] = None  # "screenshot", "whatsapp_forward", "wish_greeting", "banner", "blurry", None
    junk_confidence: float = 0.0
    junk_reasons: List[str] = field(default_factory=list)

    # Video specifics
    duration_sec: float = 0.0
    fps: float = 0.0

    # Clustering & action state
    cluster_id: Optional[str] = None
    is_best_in_cluster: bool = False
    recommended_action: str = "keep"  # "keep" (stay on iPhone) or "archive" (move to PC, delete from iPhone)
    action_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compute_sha256(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def extract_exif_metadata(img: Image.Image, filepath: str) -> Tuple[float, str, bool, bool]:
    """
    Returns (timestamp, camera_model, has_camera_exif, is_exif_screenshot)
    """
    capture_ts = os.path.getmtime(filepath)
    camera_model = ""
    has_camera_exif = False
    is_exif_screenshot = False

    try:
        exif_data = img.getexif()
        if exif_data:
            for tag_id, val in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "Model" and val:
                    camera_model = str(val).strip()
                    if "iPhone" in camera_model or "Apple" in camera_model:
                        has_camera_exif = True
                elif tag == "Make" and val and "Apple" in str(val):
                    has_camera_exif = True
                elif tag in ("DateTimeOriginal", "DateTime") and val:
                    try:
                        dt = datetime.strptime(str(val).strip(), "%Y:%m:%d %H:%M:%S")
                        capture_ts = dt.timestamp()
                    except Exception:
                        pass
                elif tag == "UserComment" and val:
                    if "Screenshot" in str(val):
                        is_exif_screenshot = True

            # Check Exif IFD for DateTimeOriginal / LensModel / UserComment
            exif_ifd = exif_data.get_ifd(0x8769)
            if exif_ifd:
                for tag_id, val in exif_ifd.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal" and val:
                        try:
                            dt = datetime.strptime(str(val).strip(), "%Y:%m:%d %H:%M:%S")
                            capture_ts = dt.timestamp()
                        except Exception:
                            pass
                    elif tag == "LensModel" and val:
                        has_camera_exif = True
                    elif tag == "UserComment" and val and "Screenshot" in str(val):
                        is_exif_screenshot = True
    except Exception:
        pass

    # Also check PNG info for Apple screenshot metadata
    if hasattr(img, "info") and isinstance(img.info, dict):
        info_str = str(img.info)
        if "Screenshot" in info_str or "Apple" in info_str and filepath.lower().endswith(".png"):
            is_exif_screenshot = True

    return capture_ts, camera_model, has_camera_exif, is_exif_screenshot


def score_image_quality(cv_gray: np.ndarray, cv_bgr: np.ndarray, width: int, height: int) -> dict:
    """
    Computes Sharpness, Exposure, Face/Eye scores and overall Quality Score (0-100).
    """
    # Resize to standard analysis size (max 800px) so resolution doesn't skew Laplacian variance
    h, w = cv_gray.shape[:2]
    scale = min(800.0 / max(h, w, 1), 1.0)
    if scale < 1.0:
        small_gray = cv2.resize(cv_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_gray = cv_gray

    # 1. Sharpness via Laplacian Variance + Tenengrad (Sobel)
    lap_var = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())
    gx = cv2.Sobel(small_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(small_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = float(np.mean(np.sqrt(gx**2 + gy**2)))

    # Normalize sharpness to 0..100 (typical sharp photo has lap_var > 180)
    sharpness_score = min(100.0, (np.log1p(lap_var) / np.log1p(650.0)) * 75.0 + min(25.0, sobel_mag / 2.2))
    is_blurry = lap_var < 15.0

    # 2. Exposure & Dynamic Range (0..100)
    mean_brightness = float(np.mean(small_gray))
    std_contrast = float(np.std(small_gray))
    dark_clip = float(np.mean(small_gray < 12))
    bright_clip = float(np.mean(small_gray > 245))

    # Ideal brightness is around 115-135, ideal std contrast > 45
    brightness_penalty = abs(mean_brightness - 125.0) / 125.0 * 40.0
    clip_penalty = (dark_clip + bright_clip) * 60.0
    contrast_bonus = min(30.0, (std_contrast / 60.0) * 30.0)
    exposure_score = max(5.0, min(100.0, 75.0 - brightness_penalty - clip_penalty + contrast_bonus))

    # 3. Face & Open Eye Quality
    faces_count = 0
    eyes_count = 0
    face_score = 50.0  # neutral baseline when no faces

    try:
        faces = _FACE_CASCADE.detectMultiScale(small_gray, scaleFactor=1.12, minNeighbors=5, minSize=(32, 32))
        faces_count = len(faces)
        if faces_count > 0:
            face_sharpness_vals = []
            for (fx, fy, fw, fh) in faces:
                roi_gray = small_gray[fy : fy + fh, fx : fx + fw]
                f_lap = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
                face_sharpness_vals.append(f_lap)
                eyes = _EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10))
                eyes_count += len(eyes)

            avg_face_sharp = float(np.mean(face_sharpness_vals))
            eye_ratio = min(1.0, eyes_count / max(1, faces_count * 2))
            face_score = min(100.0, 50.0 + min(30.0, avg_face_sharp / 8.0) + eye_ratio * 20.0)
    except Exception:
        pass

    # 4. Resolution score (12MP / 24MP / 48MP iPhone Pro Max bonus)
    megapixels = (width * height) / 1_000_000.0
    res_score = min(100.0, 40.0 + min(60.0, (megapixels / 24.0) * 60.0))

    # Weighted Overall Quality Score
    overall = (
        sharpness_score * 0.45
        + exposure_score * 0.25
        + face_score * 0.15
        + res_score * 0.15
    )

    return {
        "quality_score": round(overall, 1),
        "sharpness_score": round(sharpness_score, 1),
        "exposure_score": round(exposure_score, 1),
        "face_score": round(face_score, 1),
        "faces_count": int(faces_count),
        "eyes_detected": int(eyes_count),
        "is_blurry": bool(is_blurry),
    }


def detect_junk_category(
    filename: str,
    ext: str,
    width: int,
    height: int,
    has_camera_exif: bool,
    is_exif_screenshot: bool,
    cv_gray: np.ndarray,
    cv_bgr: np.ndarray,
    is_blurry: bool,
) -> Tuple[Optional[str], float, List[str]]:
    """
    Detects:
    - "screenshot": iPhone screenshots
    - "whatsapp_forward": WhatsApp forwarded images / memes
    - "wish_greeting": "Happy New Year", "Good Morning", festival/birthday wish graphics
    - "banner": Web banners, flyers, promotional posters, tall stitched images
    - "blurry": Severely out-of-focus / pocket shots
    """
    reasons: List[str] = []
    fname_lower = filename.lower()

    # 1. Explicit Screenshot Check
    if is_exif_screenshot or "screenshot" in fname_lower or fname_lower.startswith("img_s"):
        reasons.append("EXIF / Filename tagged as iPhone Screenshot")
        return "screenshot", 0.98, reasons

    if ext == ".png" and (width, height) in IPHONE_SCREENSHOT_RESOLUTIONS:
        reasons.append(f"PNG matching exact iPhone Pro Max screen resolution ({width}x{height})")
        return "screenshot", 0.96, reasons

    if not has_camera_exif and (width, height) in IPHONE_SCREENSHOT_RESOLUTIONS:
        reasons.append(f"Exact iPhone screen dimensions ({width}x{height}) without camera lens EXIF")
        return "screenshot", 0.90, reasons

    # Compute visual features for Banner / Wish Greeting / WhatsApp detection
    h, w = cv_gray.shape[:2]
    aspect_ratio = max(w, h) / max(1.0, min(w, h))

    # Resize for fast text/color analysis
    small_bgr = cv2.resize(cv_bgr, (320, 320), interpolation=cv2.INTER_AREA)
    small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)

    # High-frequency text-like edge density (morphological gradient in horizontal strips)
    edges = cv2.Canny(small_gray, 80, 200)
    edge_density = float(np.mean(edges > 0))

    # Detect text-like horizontal clusters (typical of "Happy New Year", "Good Morning", banners)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_box_count = 0
    for c in contours:
        _, _, cw, ch = cv2.boundingRect(c)
        if 18 < cw < 260 and 6 < ch < 55 and (cw / max(1, ch)) > 1.8:
            text_box_count += 1

    # Color saturation & flat background metrics
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    high_sat_ratio = float(np.mean(sat > 150))
    very_bright_ratio = float(np.mean(val > 235))

    # Color quantization flatness (synthetic graphics/posters have large uniform blocks)
    quantized = (small_bgr // 32) * 32
    unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))

    # 2. Banner / Flyer / Promotional Poster Check
    if "banner" in fname_lower or "flyer" in fname_lower or "promo" in fname_lower:
        reasons.append("Filename matches banner/promotional graphic")
        return "banner", 0.94, reasons

    if aspect_ratio >= 2.35 and not has_camera_exif:
        reasons.append(f"Extreme banner/scroll aspect ratio ({aspect_ratio:.1f}:1) with no camera EXIF")
        return "banner", 0.91, reasons

    if not has_camera_exif and text_box_count >= 14 and unique_colors < 95 and very_bright_ratio > 0.25:
        reasons.append(f"Poster/Banner layout ({text_box_count} text blocks over flat graphic background)")
        return "banner", 0.86, reasons

    # 3. Wish Messages ("Happy New Year", "Good Morning", Festival Wishes)
    wish_keywords = ("wish", "new_year", "newyear", "greeting", "morning", "diwali", "christmas", "eid", "bday", "happy")
    if any(k in fname_lower for k in wish_keywords):
        reasons.append("Filename matches greeting/wish card pattern")
        return "wish_greeting", 0.95, reasons

    # Greeting graphics typically have: no camera EXIF, high color saturation (sparkles/flowers/gold/red),
    # overlaid typography (text_box_count >= 5), and medium-low resolution (720p-1600p)
    if (
        not has_camera_exif
        and high_sat_ratio > 0.34
        and text_box_count >= 5
        and edge_density > 0.065
        and max(width, height) <= 2048
    ):
        reasons.append(
            f"Vibrant graphic ({int(high_sat_ratio*100)}% high saturation) with overlaid greeting typography ({text_box_count} text regions) & no camera EXIF"
        )
        return "wish_greeting", 0.88, reasons

    # 4. WhatsApp Forward Detection
    # WhatsApp filenames: IMG-20261105-WA0012.jpg or UUID-style or stripped EXIF JPEG <= 1600px
    wa_pattern = re.compile(r"(img[-_]\d{8}[-_]wa\d+|whatsapp|wa\d{4}|^[0-9a-f]{8}-[0-9a-f]{4})", re.I)
    if wa_pattern.search(filename):
        reasons.append("Matches WhatsApp media naming pattern (IMG-YYYYMMDD-WAxxxx / UUID)")
        return "whatsapp_forward", 0.97, reasons

    # WhatsApp standard resized dimensions (commonly max side 1600, 1280, 1080, 1024, 800 with stripped EXIF)
    max_dim = max(width, height)
    if not has_camera_exif and ext in (".jpg", ".jpeg") and max_dim in (1600, 1280, 1080, 1024, 960, 800, 720):
        reasons.append(f"Stripped camera EXIF + standard WhatsApp compression resolution ({width}x{height})")
        return "whatsapp_forward", 0.87, reasons

    if not has_camera_exif and ext in (".jpg", ".jpeg") and max_dim <= 1600 and text_box_count >= 4:
        reasons.append(f"Forwarded social image ({width}x{height}, no camera EXIF, embedded text/caption)")
        return "whatsapp_forward", 0.82, reasons

    # 5. Blurry Camera Shot
    if is_blurry and has_camera_exif:
        reasons.append("Camera photo suffers from severe motion blur or out-of-focus lens")
        return "blurry", 0.84, reasons

    return None, 0.0, reasons


def analyze_video_file(filepath: str, rel_path: str, file_id: str) -> MediaItem:
    file_size = os.path.getsize(filepath)
    capture_ts = os.path.getmtime(filepath)
    width, height, fps, duration_sec = 0, 0, 0.0, 0.0
    phash_str, dhash_str = "", ""
    sharpness_score = 70.0

    try:
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 0 and frame_count > 0:
                duration_sec = round(frame_count / fps, 1)

            # Sample a frame at 20% into the video for thumbnail & perceptual hash
            if frame_count > 5:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * 0.2))
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                phash_str = str(imagehash.phash(pil_img))
                dhash_str = str(imagehash.dhash(pil_img))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness_score = round(min(100.0, (np.log1p(lap) / np.log1p(500.0)) * 100.0), 1)
        cap.release()
    except Exception:
        pass

    res_bonus = min(30.0, ((width * height) / (3840 * 2160)) * 30.0) if width and height else 15.0
    quality = round(sharpness_score * 0.7 + res_bonus, 1)

    return MediaItem(
        id=file_id,
        filepath=filepath,
        filename=os.path.basename(filepath),
        rel_path=rel_path,
        media_type="video",
        file_size=file_size,
        width=width,
        height=height,
        capture_time=capture_ts,
        capture_time_iso=datetime.fromtimestamp(capture_ts).strftime("%Y-%m-%d %H:%M:%S"),
        phash=phash_str,
        dhash=dhash_str,
        quality_score=quality,
        sharpness_score=sharpness_score,
        duration_sec=duration_sec,
        fps=round(fps, 1),
    )


def analyze_photo_file(
    filepath: str,
    rel_path: str,
    file_id: str,
    live_video_path: Optional[str] = None,
) -> Optional[MediaItem]:
    try:
        file_size = os.path.getsize(filepath)
        ext = os.path.splitext(filepath)[1].lower()

        pil_img = Image.open(filepath)
        capture_ts, camera_model, has_camera_exif, is_exif_screenshot = extract_exif_metadata(pil_img, filepath)

        rgb_img = pil_img.convert("RGB")
        width, height = rgb_img.size

        # Compute perceptual hashes
        phash_str = str(imagehash.phash(rgb_img))
        dhash_str = str(imagehash.dhash(rgb_img))
        colorhash_str = str(imagehash.colorhash(rgb_img))

        # Convert to OpenCV arrays for quality & junk analysis
        cv_rgb = np.array(rgb_img)
        cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

        q = score_image_quality(cv_gray, cv_bgr, width, height)
        junk_cat, junk_conf, junk_reasons = detect_junk_category(
            filename=os.path.basename(filepath),
            ext=ext,
            width=width,
            height=height,
            has_camera_exif=has_camera_exif,
            is_exif_screenshot=is_exif_screenshot,
            cv_gray=cv_gray,
            cv_bgr=cv_bgr,
            is_blurry=q["is_blurry"],
        )

        live_size = os.path.getsize(live_video_path) if live_video_path and os.path.exists(live_video_path) else 0

        item = MediaItem(
            id=file_id,
            filepath=filepath,
            filename=os.path.basename(filepath),
            rel_path=rel_path,
            media_type="photo",
            file_size=file_size + live_size,
            width=width,
            height=height,
            capture_time=capture_ts,
            capture_time_iso=datetime.fromtimestamp(capture_ts).strftime("%Y-%m-%d %H:%M:%S"),
            camera_model=camera_model or ("iPhone Camera" if has_camera_exif else "External / Social"),
            has_camera_exif=has_camera_exif,
            is_live_photo=bool(live_video_path),
            live_video_path=live_video_path,
            live_video_size=live_size,
            phash=phash_str,
            dhash=dhash_str,
            colorhash=colorhash_str,
            quality_score=q["quality_score"],
            sharpness_score=q["sharpness_score"],
            exposure_score=q["exposure_score"],
            face_score=q["face_score"],
            faces_count=q["faces_count"],
            eyes_Detected=q["eyes_detected"],
            is_blurry=q["is_blurry"],
            junk_category=junk_cat,
            junk_confidence=round(junk_conf, 2),
            junk_reasons=junk_reasons,
        )

        if junk_cat:
            item.recommended_action = "archive"
            item.action_reason = f"Detected {junk_cat.replace('_', ' ').title()}: {junk_reasons[0]}"

        return item
    except Exception as exc:
        print(f"[WARN] Could not analyze {filepath}: {exc}")
        return None


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b or len(hash_a) != len(hash_b):
        return 64
    try:
        return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
    except ValueError:
        return 64


def cluster_similar_media(
    items: List[MediaItem],
    phash_threshold: int = 12,
    burst_time_window_sec: float = 90.0,
    burst_phash_threshold: int = 18,
) -> List[ dict ]:
    """
    Groups photos & videos across both phones (16 Pro Max + 18 Pro Max) into similarity clusters.
    Uses:
    1. Direct visual similarity (pHash distance <= phash_threshold) regardless of time
    2. Burst / Multi-take similarity (taken within burst_time_window_sec AND pHash distance <= burst_phash_threshold)
    Within each cluster, ranks items by quality_score descending:
    - Top item gets `is_best_in_cluster = True` and `recommended_action = "keep"`
    - Remaining items get `recommended_action = "archive"` with a clear reason comparing quality.
    """
    photos = [it for it in items if it.media_type == "photo" and it.phash]
    videos = [it for it in items if it.media_type == "video" and it.phash]

    clusters: List[dict] = []

    def _build_clusters(media_list: List[MediaItem], prefix: str):
        n = len(media_list)
        if n < 2:
            return

        # Sort by capture_time for fast window comparisons
        sorted_media = sorted(media_list, key=lambda x: x.capture_time)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            a = sorted_media[i]
            for j in range(i + 1, n):
                b = sorted_media[j]
                p_dist = hamming_distance(a.phash, b.phash)
                d_dist = hamming_distance(a.dhash, b.dhash)
                best_dist = min(p_dist, d_dist)
                time_diff = abs(a.capture_time - b.capture_time)

                # Rule 1: Identical or near-identical visual match anywhere in library
                if best_dist <= phash_threshold:
                    union(i, j)
                # Rule 2: Taken within burst time window and visually related scene
                elif time_diff <= burst_time_window_sec and best_dist <= burst_phash_threshold:
                    union(i, j)

        groups: Dict[int, List[MediaItem]] = {}
        for idx, item in enumerate(sorted_media):
            root = find(idx)
            groups.setdefault(root, []).append(item)

        cluster_idx = 1
        for group_items in groups.values():
            if len(group_items) < 2:
                continue

            cid = f"{prefix}_cluster_{cluster_idx:03d}"
            cluster_idx += 1

            # Sort by quality_score descending (prefer non-junk, sharper, higher resolution)
            group_items.sort(
                key=lambda x: (
                    0 if x.junk_category else 1,
                    x.quality_score,
                    x.sharpness_score,
                    x.file_size,
                ),
                reverse=True,
            )

            winner = group_items[0]
            winner.cluster_id = cid
            winner.is_best_in_cluster = True
            if not winner.junk_category:
                winner.recommended_action = "keep"
                winner.action_reason = (
                    f"Best shot in group of {len(group_items)} "
                    f"(Quality {winner.quality_score}/100, Sharpness {winner.sharpness_score})"
                )

            reclaimable_bytes = 0
            for runner_up in group_items[1:]:
                runner_up.cluster_id = cid
                runner_up.is_best_in_cluster = False
                runner_up.recommended_action = "archive"
                reclaimable_bytes += runner_up.file_size

                q_diff = round(winner.quality_score - runner_up.quality_score, 1)
                if hamming_distance(winner.phash, runner_up.phash) <= 2:
                    runner_up.action_reason = f"Exact/Near duplicate of {winner.filename}"
                elif runner_up.is_blurry:
                    runner_up.action_reason = f"Blurry alternate take (Sharpness {runner_up.sharpness_score} vs {winner.sharpness_score})"
                elif q_diff > 0:
                    runner_up.action_reason = f"Lower quality take (-{q_diff} pts vs {winner.filename})"
                else:
                    runner_up.action_reason = f"Redundant take in burst with {winner.filename}"

            clusters.append(
                {
                    "cluster_id": cid,
                    "media_type": prefix,
                    "count": len(group_items),
                    "winner_id": winner.id,
                    "winner_filename": winner.filename,
                    "capture_time_iso": winner.capture_time_iso,
                    "reclaimable_bytes": reclaimable_bytes,
                    "items": [it.to_dict() for it in group_items],
                }
            )

    _build_clusters(photos, "photo")
    _build_clusters(videos, "video")

    # Sort clusters by reclaimable space descending
    clusters.sort(key=lambda c: c["reclaimable_bytes"], reverse=True)
    return clusters


def scan_Backup_directory(source_dir: str) -> dict:
    """
    Scans a combined backup directory containing photos and videos from both iPhones.
    Pairs Live Photos (.HEIC + .MOV), analyzes quality & junk categories, and builds similarity clusters.
    """
    root_path = Path(source_dir).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Source directory not found: {root_path}")

    all_files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip hidden or archive folders if nested inside
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("PC_Archive_Moved", "_Archive")]
        for fn in filenames:
            if not fn.startswith("."):
                all_files.append(Path(dirpath) / fn)

    # Index by (parent_dir, stem_upper) to pair Live Photos (.HEIC + .MOV)
    stem_map: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for p in all_files:
        ext = p.suffix.lower()
        if ext in PHOTO_EXTENSIONS or ext in VIDEO_EXTENSIONS:
            key = (str(p.parent), p.stem.upper())
            stem_map.setdefault(key, {})[ext] = p

    paired_mov_paths = set()
    photo_tasks: List[Tuple[Path, Optional[Path]]] = []
    video_tasks: List[Path] = []

    for (_, _), ext_dict in stem_map.items():
        photo_path = None
        for p_ext in PHOTO_EXTENSIONS:
            if p_ext in ext_dict:
                photo_path = ext_dict[p_ext]
                break

        mov_path = ext_dict.get(".mov")
        # If both a photo and a small .MOV (< 15MB) exist with the same stem, treat as Live Photo pair
        if photo_path and mov_path:
            try:
                if mov_path.stat().st_size < 18 * 1024 * 1024:
                    paired_mov_paths.add(str(mov_path))
                    photo_tasks.append((photo_path, mov_path))
                    continue
            except Exception:
                pass

        if photo_path:
            photo_tasks.append((photo_path, None))

        for v_ext in VIDEO_EXTENSIONS:
            if v_ext in ext_dict:
                vp = ext_dict[v_ext]
                if str(vp) not in paired_mov_paths:
                    video_tasks.append(vp)

    total_to_scan = len(photo_tasks) + len(video_tasks)
    print(f"\n[SCAN] Found {len(photo_tasks)} photos (incl. {len(paired_mov_paths)} Live Photo pairs) and {len(video_tasks)} videos in {root_path}", flush=True)

    media_items: List[MediaItem] = []
    idx_counter = 1

    for i, (photo_path, live_mov) in enumerate(sorted(photo_tasks, key=lambda x: str(x[0])), 1):
        if i == 1 or i % 25 == 0 or i == len(photo_tasks):
            print(f"  -> Analyzing photos: {i}/{len(photo_tasks)} ({photo_path.name})...", flush=True)
        rel_p = str(photo_path.relative_to(root_path))
        file_id = f"media_{idx_counter:05d}"
        idx_counter += 1
        item = analyze_photo_file(
            filepath=str(photo_path),
            rel_path=rel_p,
            file_id=file_id,
            live_video_path=str(live_mov) if live_mov else None,
        )
        if item:
            media_items.append(item)

    for j, video_path in enumerate(sorted(video_tasks), 1):
        if j == 1 or j % 10 == 0 or j == len(video_tasks):
            print(f"  -> Analyzing videos: {j}/{len(video_tasks)} ({video_path.name})...", flush=True)
        rel_p = str(video_path.relative_to(root_path))
        file_id = f"media_{idx_counter:05d}"
        idx_counter += 1
        v_item = analyze_video_file(
            filepath=str(video_path),
            rel_path=rel_p,
            file_id=file_id,
        )
        media_items.append(v_item)

    print(f"[SCAN] Clustering {len(media_items)} items for visual similarity & bursts...", flush=True)
    clusters = cluster_similar_media(media_items)
    print(f"[SCAN] Complete! Found {len(clusters)} similarity clusters.\n", flush=True)

    # Calculate summary statistics
    total_size = sum(it.file_size for it in media_items)
    archive_candidates = [it for it in media_items if it.recommended_action == "archive"]
    reclaimable_size = sum(it.file_size for it in archive_candidates)

    junk_breakdown = {
        "screenshot": [],
        "whatsapp_forward": [],
        "wish_greeting": [],
        "banner": [],
        "blurry": [],
    }
    for it in media_items:
        if it.junk_category in junk_breakdown:
            junk_breakdown[it.junk_category].append(it.to_dict())

    videos_list = sorted(
        [it.to_dict() for it in media_items if it.media_type == "video"],
        key=lambda x: x["file_size"],
        reverse=True,
    )

    return {
        "source_dir": str(root_path),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_items": len(media_items),
            "total_photos": sum(1 for it in media_items if it.media_type == "photo"),
            "total_videos": sum(1 for it in media_items if it.media_type == "video"),
            "total_size_bytes": total_size,
            "cluster_count": len(clusters),
            "archive_candidates_count": len(archive_candidates),
            "keep_count": len(media_items) - len(archive_candidates),
            "reclaimable_bytes": reclaimable_size,
            "junk_counts": {k: len(v) for k, v in junk_breakdown.items()},
        },
        "clusters": clusters,
        "junk": junk_breakdown,
        "videos": videos_list,
        "items": {it.id: it.to_dict() for it in media_items},
    }
