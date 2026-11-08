"""
High-Speed Multi-Core Photo & Video Analyzer for iPhone 16 Pro Max / 18 Pro Max (Optimized for Intel i7 / 16GB RAM).

Key iOS-Specific Intelligence:
1. Real iPhone Camera vs. WhatsApp/Social Forwards on iOS:
   - On iOS, WhatsApp forwards, wish cards ("Happy New Year", "Good Morning"), and banners saved to Photos
     are renamed by iOS to `IMG_xxxx.JPG` / `.PNG` — NOT Android's `IMG-YYYYMMDD-WAxxxx.jpg`!
   - However, genuine iPhone 16/18 Pro Max camera photos ALWAYS contain Apple Camera/Lens EXIF (`Make=Apple`, `Model=iPhone...`),
     whereas WhatsApp/social forwards have stripped EXIF (`has_camera_exif=False`).
   - Our detector uses EXIF presence + text/typography layout + color saturation + aspect ratio to accurately classify:
     * `screenshot` (Exact iPhone screen resolutions / PNG UI / EXIF Screenshot tag)
     * `banner` (Wide/tall promotional aspect ratios >= 2.0:1 or dense flyer text layout)
     * `wish_greeting` ("Happy New Year" / "Good Morning" / festival cards: vibrant colors + overlaid text regions + no camera EXIF)
     * `whatsapp_forward` (Any remaining stripped-EXIF social/forwarded JPEG/PNG/WebP saved in DCIM)
     * `blurry` (Out-of-focus or motion-blurred camera shots)
2. Translation-Tolerant 95% Similarity Engine:
   - Handheld iPhone multi-take shots shift slightly by a few pixels between takes.
   - Combines translation-invariant HSV Color/Luminance Histogram correlation + Perceptual Hashes (pHash/dHash) +
     EXIF Timestamp proximity + GPS Location (within 50m) so real handheld duplicate takes score >= 95%!
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

CACHE_VERSION = "v3_ios_tuned"

IPHONE_SCREENSHOT_RESOLUTIONS = {
    (1320, 2868), (2868, 1320),  # 16 Pro Max / 18 Pro Max
    (1344, 2992), (2992, 1344),  # 18 Pro Max (6.9"+)
    (1290, 2796), (2796, 1290),  # 14/15 Pro Max
    (1284, 2778), (2778, 1284),  # 12/13 Pro Max
    (1206, 2622), (2622, 1206),  # 16 Pro
    (1179, 2556), (2556, 1179),  # 14/15 Pro
    (1170, 2532), (2532, 1170),  # 12/13/14
    (1125, 2436), (2436, 1125),
    (1242, 2688), (2688, 1242),
    (828, 1792), (1792, 828),
    (750, 1334), (1334, 750),
}

_FACE_CASCADE = None
if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
    try:
        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
    mtime: float = 0.0
    width: int = 0
    height: int = 0
    capture_time: float = 0.0
    capture_time_iso: str = ""
    camera_model: str = ""
    has_camera_exif: bool = False
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    is_live_photo: bool = False
    live_video_path: Optional[str] = None
    live_video_size: int = 0

    # Hashes & scene signature for similarity %
    phash: str = ""
    dhash: str = ""
    colorhash: str = ""
    scene_vec: List[float] = field(default_factory=list)
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
    junk_category: Optional[str] = None
    junk_confidence: float = 0.0
    junk_reasons: List[str] = field(default_factory=list)

    # Video specifics
    duration_sec: float = 0.0
    fps: float = 0.0

    # Clustering & action state
    cluster_id: Optional[str] = None
    is_best_in_cluster: bool = False
    similarity_pct: float = 100.0
    time_diff_sec: float = 0.0
    distance_meters: Optional[float] = None
    recommended_action: str = "keep"
    action_reason: str = ""

    def to_dict(self, include_scene_vec: bool = False) -> dict:
        d = asdict(self)
        if not include_scene_vec:
            d.pop("scene_vec", None)
        return d


def _dms_to_decimal(dms, ref: str) -> Optional[float]:
    try:
        def _to_float(v):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
                return float(v[0]) / float(v[1])
            return float(v)

        deg = _to_float(dms[0])
        minutes = _to_float(dms[1])
        seconds = _to_float(dms[2])
        dec = deg + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ("S", "W"):
            dec = -dec
        return round(dec, 6)
    except Exception:
        return None


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def extract_exif_metadata(img: Image.Image, filepath: str) -> Tuple[float, str, bool, bool, Optional[float], Optional[float]]:
    capture_ts = os.path.getmtime(filepath)
    camera_model = ""
    has_camera_exif = False
    is_exif_screenshot = False
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None

    try:
        exif_data = img.getexif()
        if exif_data:
            for tag_id, val in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "Model" and val:
                    camera_model = str(val).strip()
                    if any(k in camera_model for k in ("iPhone", "Apple", "Canon", "Sony", "Nikon", "Pixel", "Galaxy")):
                        has_camera_exif = True
                elif tag == "Make" and val:
                    if any(k in str(val) for k in ("Apple", "Canon", "Sony", "Nikon", "Google", "Samsung")):
                        has_camera_exif = True
                elif tag in ("DateTimeOriginal", "DateTime") and val:
                    try:
                        clean_dt = str(val).strip()[:19]
                        dt = datetime.strptime(clean_dt, "%Y:%m:%d %H:%M:%S")
                        capture_ts = dt.timestamp()
                    except Exception:
                        pass
                elif tag == "UserComment" and val and "Screenshot" in str(val):
                    is_exif_screenshot = True

            exif_ifd = exif_data.get_ifd(0x8769)
            if exif_ifd:
                for tag_id, val in exif_ifd.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal" and val:
                        try:
                            clean_dt = str(val).strip()[:19]
                            dt = datetime.strptime(clean_dt, "%Y:%m:%d %H:%M:%S")
                            capture_ts = dt.timestamp()
                        except Exception:
                            pass
                    elif tag in ("LensModel", "LensMake", "FocalLength", "FNumber", "ExposureTime") and val:
                        has_camera_exif = True
                    elif tag == "UserComment" and val and "Screenshot" in str(val):
                        is_exif_screenshot = True

            gps_ifd = exif_data.get_ifd(0x8825)
            if gps_ifd:
                lat_dms = gps_ifd.get(2)
                lat_ref = gps_ifd.get(1, "N")
                lon_dms = gps_ifd.get(4)
                lon_ref = gps_ifd.get(3, "E")
                if lat_dms and lon_dms:
                    gps_lat = _dms_to_decimal(lat_dms, str(lat_ref))
                    gps_lon = _dms_to_decimal(lon_dms, str(lon_ref))
    except Exception:
        pass

    if hasattr(img, "info") and isinstance(img.info, dict):
        info_str = str(img.info)
        if "Screenshot" in info_str or ("Apple" in info_str and filepath.lower().endswith(".png")):
            is_exif_screenshot = True

    return capture_ts, camera_model, has_camera_exif, is_exif_screenshot, gps_lat, gps_lon


def compute_spatial_scene_vector(cv_bgr: np.ndarray) -> List[float]:
    """
    Computes a translation-tolerant global + 2x2 quadrant HSV color & luminance histogram vector.
    Allows slight handheld camera movement between burst shots while still distinguishing different scenes.
    """
    small = cv2.resize(cv_bgr, (120, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # 1. Global HSV histogram (12 Hue bins + 6 Sat bins + 6 Val bins = 24 floats)
    h_glob = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    s_glob = cv2.calcHist([hsv], [1], None, [6], [0, 256]).flatten()
    v_glob = cv2.calcHist([hsv], [2], None, [6], [0, 256]).flatten()
    glob_vec = np.concatenate([
        h_glob / (np.sum(h_glob) + 1e-6),
        s_glob / (np.sum(s_glob) + 1e-6),
        v_glob / (np.sum(v_glob) + 1e-6),
    ])

    # 2. Soft 2x2 Quadrant HSV stats (60x60 each -> 8 floats per quadrant = 32 floats)
    quad_feats: List[float] = []
    for qy in range(2):
        for qx in range(2):
            patch = hsv[qy * 60 : (qy + 1) * 60, qx * 60 : (qx + 1) * 60]
            h_q = cv2.calcHist([patch], [0], None, [6], [0, 180]).flatten()
            h_q = h_q / (np.sum(h_q) + 1e-6)
            s_m = float(np.mean(patch[:, :, 1])) / 255.0
            v_m = float(np.mean(patch[:, :, 2])) / 255.0
            quad_feats.extend(h_q.tolist())
            quad_feats.extend([s_m, v_m])

    vec = np.concatenate([glob_vec * 1.2, np.array(quad_feats, dtype=np.float32)])
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec = vec / norm
    return [round(float(x), 5) for x in vec]


def score_image_quality_fast(small_gray: np.ndarray, orig_width: int, orig_height: int) -> dict:
    lap_var = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())
    gx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = float(np.mean(cv2.magnitude(gx, gy)))

    sharpness_score = min(100.0, (np.log1p(lap_var) / np.log1p(650.0)) * 75.0 + min(25.0, sobel_mag / 2.2))
    is_blurry = lap_var < 18.0

    mean_brightness = float(np.mean(small_gray))
    std_contrast = float(np.std(small_gray))
    dark_clip = float(np.mean(small_gray < 12))
    bright_clip = float(np.mean(small_gray > 245))

    brightness_penalty = abs(mean_brightness - 125.0) / 125.0 * 40.0
    clip_penalty = (dark_clip + bright_clip) * 60.0
    contrast_bonus = min(30.0, (std_contrast / 60.0) * 30.0)
    exposure_score = max(5.0, min(100.0, 75.0 - brightness_penalty - clip_penalty + contrast_bonus))

    faces_count = 0
    face_score = 50.0

    if _FACE_CASCADE is not None:
        try:
            h, w = small_gray.shape[:2]
            f_scale = min(280.0 / max(h, w, 1), 1.0)
            tiny_gray = cv2.resize(small_gray, (int(w * f_scale), int(h * f_scale)), interpolation=cv2.INTER_AREA) if f_scale < 1.0 else small_gray
            faces = _FACE_CASCADE.detectMultiScale(tiny_gray, scaleFactor=1.2, minNeighbors=5, minSize=(24, 24))
            faces_count = len(faces)
            if faces_count > 0:
                face_score = min(100.0, 65.0 + min(35.0, sharpness_score * 0.35))
        except Exception:
            pass

    megapixels = (orig_width * orig_height) / 1_000_000.0
    res_score = min(100.0, 40.0 + min(60.0, (megapixels / 24.0) * 60.0))

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
        "eyes_detected": int(faces_count * 2 if faces_count > 0 else 0),
        "is_blurry": bool(is_blurry),
    }


def detect_junk_category(
    filename: str,
    ext: str,
    width: int,
    height: int,
    has_camera_exif: bool,
    is_exif_screenshot: bool,
    small_gray: np.ndarray,
    small_bgr: np.ndarray,
    is_blurry: bool,
) -> Tuple[Optional[str], float, List[str]]:
    """
    iOS-Tuned Clutter & Social Junk Detector:
    - Catches iPhone Screenshots (.PNG or exact screen dimensions)
    - Catches Banners / Flyers / Tall stitched images
    - Catches Wish Greetings ("Happy New Year", "Good Morning", festival cards)
    - Catches WhatsApp Forwards saved on iPhone (even when renamed to `IMG_xxxx.JPG` by iOS!)
    - Catches Blurry camera photos
    """
    reasons: List[str] = []
    fname_lower = filename.lower()

    # 1. Explicit Screenshot Check
    if is_exif_screenshot or "screenshot" in fname_lower or fname_lower.startswith("img_s"):
        reasons.append("EXIF / Filename tagged as iPhone Screenshot")
        return "screenshot", 0.98, reasons

    if ext == ".png" and (width, height) in IPHONE_SCREENSHOT_RESOLUTIONS:
        reasons.append(f"PNG matching exact iPhone screen resolution ({width}x{height})")
        return "screenshot", 0.96, reasons

    if not has_camera_exif and (width, height) in IPHONE_SCREENSHOT_RESOLUTIONS:
        reasons.append(f"Exact iPhone screen dimensions ({width}x{height}) without camera lens EXIF")
        return "screenshot", 0.92, reasons

    # Any PNG without camera EXIF that has phone-like vertical aspect ratio (approx 19.5:9) is also a mobile screenshot
    aspect_ratio = max(width, height) / max(1.0, min(width, height))
    if ext == ".png" and not has_camera_exif and 2.05 <= aspect_ratio <= 2.25:
        reasons.append(f"Mobile screen aspect ratio PNG ({width}x{height}) with no camera EXIF")
        return "screenshot", 0.89, reasons

    # 2. If it HAS genuine Apple/Camera Lens EXIF:
    # It was physically shot with the iPhone camera lens. Only flag if it's severely blurry!
    if has_camera_exif:
        if is_blurry:
            reasons.append("Camera photo suffers from severe motion blur or out-of-focus lens")
            return "blurry", 0.85, reasons
        return None, 0.0, reasons

    # =========================================================================
    # Below here: `has_camera_exif` is FALSE!
    # On an iPhone 16 Pro Max / 18 Pro Max, any image in DCIM without camera EXIF
    # was NOT taken by the iPhone camera — it was received/saved from WhatsApp,
    # SMS, Web, or social forwards! Now we classify which type of clutter it is:
    # =========================================================================
    tiny_bgr = cv2.resize(small_bgr, (240, 240), interpolation=cv2.INTER_AREA)
    tiny_gray = cv2.cvtColor(tiny_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tiny_bgr, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(tiny_gray, 80, 200)
    edge_density = float(np.mean(edges > 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_box_count = 0
    for c in contours:
        _, _, cw, ch = cv2.boundingRect(c)
        if 12 < cw < 210 and 5 < ch < 48 and (cw / max(1, ch)) > 1.7:
            text_box_count += 1

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    high_sat_ratio = float(np.mean(sat > 140))
    very_bright_ratio = float(np.mean(val > 230))

    # 3. Banner / Flyer / Poster Check
    if any(k in fname_lower for k in ("banner", "flyer", "promo", "poster")):
        reasons.append("Filename matches banner/promotional graphic")
        return "banner", 0.94, reasons

    if aspect_ratio >= 2.05:
        reasons.append(f"Wide/Tall Banner or Flyer aspect ratio ({aspect_ratio:.1f}:1) with no camera EXIF")
        return "banner", 0.91, reasons

    if text_box_count >= 10 and very_bright_ratio > 0.22:
        reasons.append(f"Promotional Flyer / Banner layout ({text_box_count} text blocks over flat graphic background)")
        return "banner", 0.87, reasons

    # 4. Wish Messages ("Happy New Year", "Good Morning", Festival / Birthday Greeting Cards)
    wish_keywords = ("wish", "new_year", "newyear", "greeting", "morning", "diwali", "christmas", "eid", "bday", "happy")
    if any(k in fname_lower for k in wish_keywords):
        reasons.append("Filename matches greeting/wish card pattern")
        return "wish_greeting", 0.95, reasons

    # Greeting cards saved from WhatsApp have colorful/festive saturation + overlaid text blocks
    if high_sat_ratio > 0.26 and text_box_count >= 3 and edge_density > 0.045:
        reasons.append(
            f"Festive/Wish Greeting Card ({int(high_sat_ratio*100)}% vibrant saturation + {text_box_count} overlaid greeting text regions, no camera EXIF)"
        )
        return "wish_greeting", 0.88, reasons

    # 5. WhatsApp Forwards & Saved Social Media Images
    wa_pattern = re.compile(r"(img[-_]\d{8}[-_]wa\d+|whatsapp|wa\d{4}|^[0-9a-f]{8}-[0-9a-f]{4})", re.I)
    if wa_pattern.search(filename):
        reasons.append("Matches WhatsApp media naming pattern (IMG-YYYYMMDD-WAxxxx / UUID)")
        return "whatsapp_forward", 0.97, reasons

    # On iOS, WhatsApp saves images as IMG_xxxx.JPG with stripped camera EXIF and compressed dimensions (<= 2048px)
    max_dim = max(width, height)
    if max_dim <= 2560:
        reasons.append(
            f"WhatsApp / Social Forward ({width}x{height} {ext.upper()}, stripped Apple Camera EXIF)"
        )
        return "whatsapp_forward", 0.89, reasons

    reasons.append(f"External non-camera image ({width}x{height}, no iPhone camera EXIF)")
    return "whatsapp_forward", 0.80, reasons


def analyze_video_file(filepath: str, rel_path: str, file_id: str) -> MediaItem:
    stat = os.stat(filepath)
    file_size = stat.st_size
    capture_ts = stat.st_mtime
    width, height, fps, duration_sec = 0, 0, 0.0, 0.0
    phash_str, dhash_str = "", ""
    scene_vec: List[float] = []
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

            if frame_count > 5:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * 0.2))
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                scale = min(480.0 / max(h, w, 1), 1.0)
                small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                phash_str = str(imagehash.phash(pil_img))
                dhash_str = str(imagehash.dhash(pil_img))
                scene_vec = compute_spatial_scene_vector(small_frame)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
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
        mtime=stat.st_mtime,
        width=width,
        height=height,
        capture_time=capture_ts,
        capture_time_iso=datetime.fromtimestamp(capture_ts).strftime("%Y-%m-%d %H:%M:%S"),
        phash=phash_str,
        dhash=dhash_str,
        scene_vec=scene_vec,
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
        stat = os.stat(filepath)
        file_size = stat.st_size
        ext = os.path.splitext(filepath)[1].lower()

        pil_img = Image.open(filepath)
        orig_width, orig_height = pil_img.size
        capture_ts, camera_model, has_camera_exif, is_exif_screenshot, gps_lat, gps_lon = extract_exif_metadata(
            pil_img, filepath
        )

        try:
            pil_img.draft("RGB", (512, 512))
        except Exception:
            pass
        pil_img.thumbnail((512, 512), Image.Resampling.BILINEAR)
        rgb_small = pil_img.convert("RGB")

        phash_str = str(imagehash.phash(rgb_small))
        dhash_str = str(imagehash.dhash(rgb_small))
        colorhash_str = str(imagehash.colorhash(rgb_small))

        cv_rgb = np.array(rgb_small)
        cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        cv_gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

        scene_vec = compute_spatial_scene_vector(cv_bgr)
        q = score_image_quality_fast(cv_gray, orig_width, orig_height)
        junk_cat, junk_conf, junk_reasons = detect_junk_category(
            filename=os.path.basename(filepath),
            ext=ext,
            width=orig_width,
            height=orig_height,
            has_camera_exif=has_camera_exif,
            is_exif_screenshot=is_exif_screenshot,
            small_gray=cv_gray,
            small_bgr=cv_bgr,
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
            mtime=stat.st_mtime,
            width=orig_width,
            height=orig_height,
            capture_time=capture_ts,
            capture_time_iso=datetime.fromtimestamp(capture_ts).strftime("%Y-%m-%d %H:%M:%S"),
            camera_model=camera_model or ("iPhone Camera" if has_camera_exif else "WhatsApp / Saved Image"),
            has_camera_exif=has_camera_exif,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            is_live_photo=bool(live_video_path),
            live_video_path=live_video_path,
            live_video_size=live_size,
            phash=phash_str,
            dhash=dhash_str,
            colorhash=colorhash_str,
            scene_vec=scene_vec,
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


def compute_pair_similarity(
    a: MediaItem,
    b: MediaItem,
    max_time_window_sec: float = 120.0,
    max_gps_distance_meters: float = 50.0,
) -> Tuple[float, float, Optional[float], bool]:
    """
    Handheld-Calibrated Multi-Factor Similarity Score (0..100%):
    - Checks GPS distance (must be <= 50m if both have GPS)
    - Checks Timestamp proximity (within max_time_window_sec, or near-exact visual copy)
    - Combines translation-invariant HSV Scene Histogram similarity + Perceptual Hash (pHash/dHash)
      so two handheld takes of the same pose/scene taken seconds apart score 95% - 100%.
    """
    time_diff = abs(a.capture_time - b.capture_time)

    dist_m: Optional[float] = None
    if a.gps_lat is not None and a.gps_lon is not None and b.gps_lat is not None and b.gps_lon is not None:
        dist_m = round(haversine_meters(a.gps_lat, a.gps_lon, b.gps_lat, b.gps_lon), 1)
        if dist_m > max_gps_distance_meters:
            return 0.0, time_diff, dist_m, False

    p_dist = hamming_distance(a.phash, b.phash)
    d_dist = hamming_distance(a.dhash, b.dhash)
    best_hamming = min(p_dist, d_dist)

    # Calibrated curve for 64-bit perceptual hash:
    # 0..2 bits diff = 99-100% (identical)
    # 3..14 bits diff = 94-98% (same pose/scene with slight handheld movement)
    # >22 bits diff = <85% (different composition)
    if best_hamming <= 2:
        hash_sim_pct = 100.0 - best_hamming * 0.4
    elif best_hamming <= 16:
        hash_sim_pct = 99.0 - (best_hamming - 2) * 0.38
    else:
        hash_sim_pct = max(0.0, 93.5 - (best_hamming - 16) * 1.2)

    scene_sim_pct = hash_sim_pct
    if a.scene_vec and b.scene_vec and len(a.scene_vec) == len(b.scene_vec):
        va = np.array(a.scene_vec, dtype=np.float32)
        vb = np.array(b.scene_vec, dtype=np.float32)
        cos_sim = float(np.dot(va, vb))
        scene_sim_pct = max(0.0, min(100.0, cos_sim * 100.0))

    # Blend translation-invariant scene histogram (55%) + structural hash (45%)
    combined_sim = (scene_sim_pct * 0.55) + (hash_sim_pct * 0.45)

    # Bonus if taken within 15 seconds of each other at the exact same spot (burst multi-take)
    if time_diff <= 15.0 and best_hamming <= 16 and scene_sim_pct >= 92.0:
        combined_sim = min(99.8, combined_sim + 1.5)

    combined_sim = round(combined_sim, 1)
    context_ok = (time_diff <= max_time_window_sec) or (best_hamming <= 6 and combined_sim >= 97.5)

    return combined_sim, round(time_diff, 1), dist_m, context_ok


def cluster_similar_media(
    items: List[MediaItem],
    min_similarity_pct: float = 95.0,
    max_time_window_sec: float = 120.0,
    max_gps_distance_meters: float = 50.0,
) -> List[dict]:
    for it in items:
        it.cluster_id = None
        it.is_best_in_cluster = False
        it.similarity_pct = 100.0
        it.time_diff_sec = 0.0
        it.distance_meters = None
        if it.junk_category:
            it.recommended_action = "archive"
            it.action_reason = f"Detected {it.junk_category.replace('_', ' ').title()}: {it.junk_reasons[0]}"
        else:
            it.recommended_action = "keep"
            it.action_reason = "Unique photo — Keep on iPhone"

    photos = [it for it in items if it.media_type == "photo" and it.phash]
    videos = [it for it in items if it.media_type == "video" and it.phash]

    clusters: List[dict] = []

    def _build_clusters(media_list: List[MediaItem], prefix: str):
        n = len(media_list)
        if n < 2:
            return

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

        # 1. Exact/Near-exact copies anywhere in library via pHash prefix bucketing
        phash_buckets: Dict[str, List[int]] = {}
        for idx, it in enumerate(sorted_media):
            if len(it.phash) >= 4:
                phash_buckets.setdefault(it.phash[:2], []).append(idx)

        for bucket_indices in phash_buckets.values():
            if 1 < len(bucket_indices) <= 80:
                for bi in range(len(bucket_indices)):
                    for bj in range(bi + 1, len(bucket_indices)):
                        i1, i2 = bucket_indices[bi], bucket_indices[bj]
                        sim_pct, _, _, context_ok = compute_pair_similarity(
                            sorted_media[i1], sorted_media[i2], max_time_window_sec, max_gps_distance_meters
                        )
                        if context_ok and sim_pct >= min_similarity_pct:
                            union(i1, i2)

        # 2. Sliding time window over sorted timestamps
        for i in range(n):
            a = sorted_media[i]
            for j in range(i + 1, n):
                b = sorted_media[j]
                if (b.capture_time - a.capture_time) > max_time_window_sec:
                    break
                sim_pct, _, _, context_ok = compute_pair_similarity(
                    a, b, max_time_window_sec=max_time_window_sec, max_gps_distance_meters=max_gps_distance_meters
                )
                if context_ok and sim_pct >= min_similarity_pct:
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
            winner.similarity_pct = 100.0
            winner.time_diff_sec = 0.0
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

                sim_pct, t_diff, dist_m, _ = compute_pair_similarity(
                    winner,
                    runner_up,
                    max_time_window_sec=max_time_window_sec,
                    max_gps_distance_meters=max_gps_distance_meters,
                )
                runner_up.similarity_pct = sim_pct
                runner_up.time_diff_sec = t_diff
                runner_up.distance_meters = dist_m

                context_tags = [f"{sim_pct}% similar"]
                if t_diff <= 300:
                    context_tags.append(f"{int(t_diff)}s apart")
                if dist_m is not None:
                    context_tags.append(f"same GPS ({int(dist_m)}m)")
                context_str = " • ".join(context_tags)

                q_diff = round(winner.quality_score - runner_up.quality_score, 1)
                if runner_up.is_blurry:
                    runner_up.action_reason = f"{context_str} — Blurry take (Sharpness {runner_up.sharpness_score} vs {winner.sharpness_score})"
                elif q_diff > 0:
                    runner_up.action_reason = f"{context_str} — Lower quality (-{q_diff} pts vs {winner.filename})"
                else:
                    runner_up.action_reason = f"{context_str} — Duplicate take of {winner.filename}"

            clusters.append(
                {
                    "cluster_id": cid,
                    "media_type": prefix,
                    "count": len(group_items),
                    "winner_id": winner.id,
                    "winner_filename": winner.filename,
                    "capture_time_iso": winner.capture_time_iso,
                    "has_gps": winner.gps_lat is not None,
                    "reclaimable_bytes": reclaimable_bytes,
                    "items": [it.to_dict() for it in group_items],
                }
            )

    _build_clusters(photos, "photo")
    _build_clusters(videos, "video")

    clusters.sort(key=lambda c: c["reclaimable_bytes"], reverse=True)
    return clusters


def build_scan_response(
    root_path: Path,
    media_items: List[MediaItem],
    min_similarity_pct: float = 95.0,
    max_time_window_sec: float = 120.0,
) -> dict:
    clusters = cluster_similar_media(
        media_items,
        min_similarity_pct=min_similarity_pct,
        max_time_window_sec=max_time_window_sec,
    )

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
        "settings": {
            "min_similarity_pct": min_similarity_pct,
            "max_time_window_sec": max_time_window_sec,
        },
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


def _load_scan_cache(cache_file: Path) -> Dict[str, dict]:
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("_version") == CACHE_VERSION:
            return data.get("items", {})
        return {}
    except Exception:
        return {}


def _save_scan_cache(cache_file: Path, items: List[MediaItem]):
    try:
        payload = {}
        for it in items:
            key = f"{it.rel_path}|{it.file_size}|{int(it.mtime)}"
            payload[key] = it.to_dict(include_scene_vec=True)
        cache_file.write_text(json.dumps({"_version": CACHE_VERSION, "items": payload}), encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Could not write scan cache: {exc}")


def scan_Backup_directory(
    source_dir: str,
    min_similarity_pct: float = 95.0,
    max_time_window_sec: float = 120.0,
    return_raw_items: bool = False,
    progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
):
    start_time = time.time()
    root_path = Path(source_dir).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Source directory not found: {root_path}")

    cache_file = root_path / ".iphone_scan_cache.json"
    disk_cache = _load_scan_cache(cache_file)

    all_files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("PC_Archive_Moved", "_Archive")]
        for fn in filenames:
            if not fn.startswith("."):
                all_files.append(Path(dirpath) / fn)

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

    total_files = len(photo_tasks) + len(video_tasks)
    cpu_workers = min(16, max(4, (os.cpu_count() or 4)))
    print(
        f"\n[TURBO SCAN] Found {len(photo_tasks)} photos and {len(video_tasks)} videos in {root_path} (Using {cpu_workers} parallel i7 threads)",
        flush=True,
    )

    media_items: List[MediaItem] = []
    uncached_photos: List[Tuple[int, Path, Optional[Path], str]] = []
    idx_counter = 1

    for photo_path, live_mov in sorted(photo_tasks, key=lambda x: str(x[0])):
        rel_p = str(photo_path.relative_to(root_path))
        file_id = f"media_{idx_counter:05d}"
        idx_counter += 1
        try:
            st = photo_path.stat()
            live_sz = live_mov.stat().st_size if live_mov and live_mov.exists() else 0
            cache_key = f"{rel_p}|{st.st_size + live_sz}|{int(st.st_mtime)}"
            if cache_key in disk_cache:
                cached_dict = dict(disk_cache[cache_key])
                cached_dict["id"] = file_id
                cached_dict["filepath"] = str(photo_path)
                cached_dict["live_video_path"] = str(live_mov) if live_mov else None
                media_items.append(MediaItem(**cached_dict))
                continue
        except Exception:
            pass
        uncached_photos.append((idx_counter - 1, photo_path, live_mov, rel_p))

    cached_count = len(media_items)
    if cached_count > 0:
        print(f"  -> Loaded {cached_count} already-analyzed photos from instant cache!", flush=True)

    completed_count = cached_count
    if uncached_photos:
        def _worker_photo(args):
            fid_num, p_path, l_mov, r_path = args
            return analyze_photo_file(
                filepath=str(p_path),
                rel_path=r_path,
                file_id=f"media_{fid_num:05d}",
                live_video_path=str(l_mov) if live_mov else None,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            for item in pool.map(_worker_photo, uncached_photos):
                completed_count += 1
                if item:
                    media_items.append(item)
                elapsed = max(0.01, time.time() - start_time)
                rate = round(completed_count / elapsed, 1)
                if completed_count % 25 == 0 or completed_count == len(photo_tasks):
                    fname = item.filename if item else "..."
                    print(f"  -> [TURBO] {completed_count}/{total_files} ({rate} files/sec) - {fname}", flush=True)
                    if progress_callback:
                        progress_callback(completed_count, total_files, fname, rate)

    if video_tasks:
        def _worker_video(args):
            fid_num, v_path, r_path = args
            return analyze_video_file(filepath=str(v_path), rel_path=r_path, file_id=f"media_{fid_num:05d}")

        video_args = []
        for video_path in sorted(video_tasks):
            rel_p = str(video_path.relative_to(root_path))
            video_args.append((idx_counter, video_path, rel_p))
            idx_counter += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, cpu_workers)) as pool:
            for v_item in pool.map(_worker_video, video_args):
                completed_count += 1
                if v_item:
                    media_items.append(v_item)
                elapsed = max(0.01, time.time() - start_time)
                rate = round(completed_count / elapsed, 1)
                if progress_callback:
                    progress_callback(completed_count, total_files, v_item.filename, rate)

    _save_scan_cache(cache_file, media_items)

    elapsed_total = round(time.time() - start_time, 2)
    print(f"[TURBO SCAN] Analyzed {len(media_items)} files in {elapsed_total}s! Building >= {min_similarity_pct}% clusters...", flush=True)

    result = build_scan_response(
        root_path,
        media_items,
        min_similarity_pct=min_similarity_pct,
        max_time_window_sec=max_time_window_sec,
    )
    print(f"[TURBO SCAN] Complete! Found {len(result['clusters'])} similarity clusters.\n", flush=True)

    if return_raw_items:
        return result, media_items
    return result
