"""
Generates a realistic sample dataset simulating a combined folder from:
- iPhone 16 Pro Max
- iPhone 18 Pro Max

Includes:
1. Burst / Multi-take Portrait Group (4 shots: 1 razor-sharp well-exposed winner, 1 motion-blurred, 1 dark/underexposed, 1 slightly lower resolution crop)
2. Sunset Landscape Multi-take Group (3 shots across 16 Pro Max & 18 Pro Max: 1 crisp high-dynamic-range winner, 1 overexposed, 1 soft focus)
3. Food / Cafe Table Duplicate Group (2 near-identical shots + Live Photo .MOV companion)
4. Clutter & Junk Samples:
   - iPhone 18 Pro Max Screenshot (`IMG_8821_screenshot.PNG`, 1320x2868)
   - WhatsApp Forwarded Image (`IMG-20261104-WA0019.jpg`, 1280x960, no camera EXIF)
   - "Happy New Year / Good Morning" Wish Card (`happy_new_year_2027_wish.jpg`, high saturation + greeting text boxes)
   - Promotional Event Banner (`mega_sale_promo_banner.jpg`, wide 3:1 banner aspect ratio)
   - Blurry Pocket / Floor Camera Shot (`IMG_4099_blurry.jpg`, with Apple EXIF)
5. 4K Sample Video Clips (2 similar takes + 1 large standalone clip)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _create_apple_exif_bytes(model: str = "iPhone 18 Pro Max", dt_str: str = "2026:11:05 14:22:10") -> bytes:
    img = Image.new("RGB", (16, 16))
    exif = img.getexif()
    exif[0x010F] = "Apple"  # Make
    exif[0x0110] = model    # Model
    exif[0x0132] = dt_str   # DateTime
    exif[0x9003] = dt_str   # DateTimeOriginal
    return exif.tobytes()


def _draw_mountain_sunset(width: int, height: int, seed_shift: int = 0) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Warm sunset gradient sky (BGR)
    for y in range(height):
        ratio = y / float(height)
        b = int(60 + 140 * (1 - ratio))
        g = int(90 + 110 * np.sin(ratio * np.pi))
        r = int(200 + 55 * ratio)
        canvas[y, :] = (b, g, r)

    # Sun disc
    sun_x = int(width * 0.62) + seed_shift
    sun_y = int(height * 0.36)
    cv2.circle(canvas, (sun_x, sun_y), int(min(width, height) * 0.11), (180, 240, 255), -1, cv2.LINE_AA)

    # Detailed mountain ridges with fine texture (so sharp version has high Laplacian score)
    rng = np.random.default_rng(42)
    pts1 = []
    for x in range(0, width + 20, 16):
        y_m = int(height * 0.58 + 65 * np.sin((x + seed_shift) * 0.008) + rng.integers(-10, 10))
        pts1.append([x, y_m])
    pts1.extend([[width, height], [0, height]])
    cv2.fillPoly(canvas, [np.array(pts1, dtype=np.int32)], (55, 45, 70))

    # Add crisp pine trees & rock textures on foreground ridge
    for tx in range(40 + seed_shift, width - 40, 28):
        base_y = int(height * 0.78 + 20 * np.cos(tx * 0.015))
        cv2.line(canvas, (tx, base_y), (tx, base_y - 45), (25, 35, 30), 3)
        for branch in range(5):
            by = base_y - 10 - branch * 7
            bw = 18 - branch * 3
            cv2.line(canvas, (tx - bw, by), (tx + bw, by - 5), (30, 65, 40), 2, cv2.LINE_AA)

    return canvas


def _draw_portrait_scene(width: int, height: int, shift_x: int = 0, eyes_open: bool = True) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Soft studio background with detailed bookshelf / foliage texture
    for y in range(height):
        canvas[y, :] = (int(110 - y * 0.03), int(135 - y * 0.03), int(150 - y * 0.02))

    rng = np.random.default_rng(101)
    for _ in range(180):
        px = int(rng.integers(20, width - 20))
        py = int(rng.integers(20, height - 20))
        cv2.circle(canvas, (px, py), int(rng.integers(4, 18)), (90, 145, 110), 2, cv2.LINE_AA)

    cx, cy = width // 2 + shift_x, int(height * 0.45)
    # Shoulders / jacket with fine fabric grid texture
    cv2.ellipse(canvas, (cx, int(height * 0.88)), (int(width * 0.28), int(height * 0.25)), 0, 180, 360, (55, 70, 95), -1)
    for gx in range(cx - 220, cx + 220, 8):
        cv2.line(canvas, (gx, int(height * 0.68)), (gx + 15, height), (68, 85, 112), 1)

    # Head
    head_w, head_h = int(width * 0.14), int(height * 0.18)
    cv2.ellipse(canvas, (cx, cy), (head_w, head_h), 0, 0, 360, (175, 200, 230), -1, cv2.LINE_AA)

    # Hair details (fine strands for high sharpness measurement)
    for hx in range(cx - head_w, cx + head_w, 4):
        cv2.line(canvas, (hx, cy - head_h + 15), (hx + 3, cy - int(head_h * 0.55)), (35, 45, 65), 2, cv2.LINE_AA)

    # Eyes
    eye_y = cy - int(head_h * 0.12)
    eye_dx = int(head_w * 0.42)
    for ex in (cx - eye_dx, cx + eye_dx):
        if eyes_open:
            cv2.ellipse(canvas, (ex, eye_y), (22, 13), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.circle(canvas, (ex, eye_y), 9, (65, 50, 35), -1, cv2.LINE_AA)
            cv2.circle(canvas, (ex - 3, eye_y - 3), 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.line(canvas, (ex - 20, eye_y), (ex + 20, eye_y), (60, 50, 45), 3, cv2.LINE_AA)

    # Smile & nose
    cv2.line(canvas, (cx, eye_y + 10), (cx - 6, cy + 22), (135, 155, 185), 3, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy + int(head_h * 0.35)), (38, 18), 0, 15, 165, (90, 95, 185), 4, cv2.LINE_AA)

    return canvas


def _draw_cafe_food(width: int, height: int, shift: int = 0) -> np.ndarray:
    canvas = np.full((height, width, 3), (75, 95, 125), dtype=np.uint8)
    # Wood grain lines
    for y in range(0, height, 6):
        cv2.line(canvas, (0, y), (width, y + (y % 11)), (65, 85, 115), 2)

    cx, cy = width // 2 + shift, height // 2
    # Ceramic plate
    cv2.circle(canvas, (cx, cy), int(min(width, height) * 0.34), (242, 242, 240), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(min(width, height) * 0.28), (225, 228, 225), 3, cv2.LINE_AA)

    # Croissant / Berries / Garnish textures
    for bx, by, col in [(-65, -40, (60, 45, 180)), (55, -50, (50, 40, 195)), (20, 65, (85, 160, 65)), (-45, 55, (70, 145, 225))]:
        cv2.circle(canvas, (cx + bx, cy + by), 36, col, -1, cv2.LINE_AA)
        for dot in range(-20, 22, 7):
            cv2.circle(canvas, (cx + bx + dot, cy + by + (dot % 9)), 2, (240, 240, 250), -1)

    return canvas


def create_demo_workspace(base_dir: Path) -> Path:
    """
    Creates `/demo_workspace/Combined_iPhone_Backup` and populates it with realistic multi-phone photos,
    Live Photo pairs, screenshots, WhatsApp forwards, wish greetings, banners, and videos.
    """
    source_dir = base_dir / "Combined_iPhone_Backup"
    if source_dir.exists() and any(source_dir.iterdir()):
        return source_dir

    source_dir.mkdir(parents=True, exist_ok=True)
    base_ts = 1762380000.0  # Nov 2026

    # --- CLUSTER 1: Portrait Multi-Take (4 photos across iPhone 18 Pro Max) ---
    exif_18pm = _create_apple_exif_bytes("iPhone 18 Pro Max", "2026:11:05 16:10:05")

    # 1A: Sharp, well-exposed Winner
    p1_sharp = _draw_portrait_scene(1600, 1200, shift_x=0, eyes_open=True)
    p1_path = source_dir / "IMG_9001.JPG"
    Image.fromarray(cv2.cvtColor(p1_sharp, cv2.COLOR_BGR2RGB)).save(p1_path, quality=96, exif=exif_18pm)
    os.utime(p1_path, (base_ts, base_ts))

    # 1B: Motion-blurred take
    p1_blur = cv2.GaussianBlur(_draw_portrait_scene(1600, 1200, shift_x=6, eyes_open=True), (25, 25), 8.0)
    p2_path = source_dir / "IMG_9002.JPG"
    Image.fromarray(cv2.cvtColor(p1_blur, cv2.COLOR_BGR2RGB)).save(p2_path, quality=90, exif=exif_18pm)
    os.utime(p2_path, (base_ts + 4, base_ts + 4))

    # 1C: Underexposed / dark take with closed eyes
    p1_dark = np.clip(_draw_portrait_scene(1600, 1200, shift_x=-5, eyes_open=False).astype(np.float32) * 0.42, 0, 255).astype(np.uint8)
    p3_path = source_dir / "IMG_9003.JPG"
    Image.fromarray(cv2.cvtColor(p1_dark, cv2.COLOR_BGR2RGB)).save(p3_path, quality=88, exif=exif_18pm)
    os.utime(p3_path, (base_ts + 9, base_ts + 9))

    # --- CLUSTER 2: Mountain Sunset (3 takes across 16 Pro Max & 18 Pro Max) ---
    exif_16pm = _create_apple_exif_bytes("iPhone 16 Pro Max", "2026:11:05 17:45:12")

    s_best = _draw_mountain_sunset(1920, 1440, seed_shift=0)
    s1_path = source_dir / "IMG_5410.JPG"
    Image.fromarray(cv2.cvtColor(s_best, cv2.COLOR_BGR2RGB)).save(s1_path, quality=96, exif=exif_18pm)
    os.utime(s1_path, (base_ts + 3600, base_ts + 3600))

    # Slightly soft + overexposed take from 16 Pro Max
    s_over = cv2.GaussianBlur(
        np.clip(_draw_mountain_sunset(1920, 1440, seed_shift=4).astype(np.float32) * 1.35 + 30, 0, 255).astype(np.uint8),
        (9, 9),
        2.2,
    )
    s2_path = source_dir / "IMG_3108.JPG"
    Image.fromarray(cv2.cvtColor(s_over, cv2.COLOR_BGR2RGB)).save(s2_path, quality=89, exif=exif_16pm)
    os.utime(s2_path, (base_ts + 3615, base_ts + 3615))

    # Blurry take
    s_blur = cv2.GaussianBlur(_draw_mountain_sunset(1920, 1440, seed_shift=-3), (21, 21), 6.5)
    s3_path = source_dir / "IMG_3109.JPG"
    Image.fromarray(cv2.cvtColor(s_blur, cv2.COLOR_BGR2RGB)).save(s3_path, quality=85, exif=exif_16pm)
    os.utime(s3_path, (base_ts + 3622, base_ts + 3622))

    # --- CLUSTER 3: Cafe Brunch Live Photo Pair (2 takes + .MOV Live Photo video companion) ---
    c_best = _draw_cafe_food(1600, 1200, shift=0)
    c1_path = source_dir / "IMG_6201.JPG"
    Image.fromarray(cv2.cvtColor(c_best, cv2.COLOR_BGR2RGB)).save(c1_path, quality=95, exif=exif_18pm)
    os.utime(c1_path, (base_ts + 7200, base_ts + 7200))

    c_dup = cv2.GaussianBlur(_draw_cafe_food(1600, 1200, shift=3), (11, 11), 2.5)
    c2_path = source_dir / "IMG_6202.JPG"
    Image.fromarray(cv2.cvtColor(c_dup, cv2.COLOR_BGR2RGB)).save(c2_path, quality=86, exif=exif_18pm)
    os.utime(c2_path, (base_ts + 7205, base_ts + 7205))
    # Companion Live Photo .MOV for IMG_6202
    c2_mov = source_dir / "IMG_6202.MOV"
    _write_sample_video(c2_mov, width=640, height=480, frames=25, color_shift=20)
    os.utime(c2_mov, (base_ts + 7205, base_ts + 7205))

    # --- JUNK 1: iPhone 18 Pro Max Screenshot (1320 x 2868 PNG) ---
    scr = np.full((2868, 1320, 3), (245, 246, 250), dtype=np.uint8)
    cv2.rectangle(scr, (0, 0), (1320, 180), (255, 255, 255), -1)
    cv2.putText(scr, "9:41", (90, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4)
    cv2.putText(scr, "Order Confirmation #48291", (110, 420), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30, 30, 30), 4)
    for row in range(600, 2200, 260):
        cv2.rectangle(scr, (90, row), (1230, row + 190), (255, 255, 255), -1)
        cv2.putText(scr, "Item details & mobile receipt screenshot", (140, row + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 90), 2)
    scr_path = source_dir / "IMG_8821_Screenshot.PNG"
    Image.fromarray(cv2.cvtColor(scr, cv2.COLOR_BGR2RGB)).save(scr_path)

    # --- JUNK 2: WhatsApp Forwarded Image (No camera EXIF, WA naming + 1280x960) ---
    wa = np.full((960, 1280, 3), (230, 240, 230), dtype=np.uint8)
    cv2.rectangle(wa, (80, 80), (1200, 880), (255, 255, 255), -1)
    cv2.putText(wa, "Forwarded via WhatsApp", (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (40, 140, 60), 4)
    cv2.putText(wa, "10 Health Tips Everyone Should Read!", (180, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50, 50, 50), 3)
    cv2.putText(wa, "Share with 5 groups for good luck", (180, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
    wa_path = source_dir / "IMG-20261104-WA0019.jpg"
    Image.fromarray(cv2.cvtColor(wa, cv2.COLOR_BGR2RGB)).save(wa_path, quality=78)

    # --- JUNK 3: "Happy New Year / Good Morning" Wish Greeting Card ---
    wish = np.zeros((1080, 1080, 3), dtype=np.uint8)
    wish[:] = (20, 40, 220)  # Saturated festive red/gold background
    cv2.circle(wish, (540, 540), 440, (0, 195, 255), -1)
    cv2.putText(wish, "HAPPY NEW YEAR 2027!", (130, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (15, 15, 140), 6)
    cv2.putText(wish, "Wishing You & Your Family", (165, 510), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (25, 25, 120), 4)
    cv2.putText(wish, "Joy, Prosperity & Good Morning", (145, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (25, 25, 120), 4)
    cv2.putText(wish, "Best Wishes Always!", (260, 790), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 80, 10), 4)
    wish_path = source_dir / "Happy_New_Year_Wish_2027.jpg"
    Image.fromarray(cv2.cvtColor(wish, cv2.COLOR_BGR2RGB)).save(wish_path, quality=82)

    # --- JUNK 4: Promotional Web Banner (Extreme 3:1 Aspect Ratio, no EXIF) ---
    banner = np.full((600, 1800, 3), (250, 245, 235), dtype=np.uint8)
    cv2.rectangle(banner, (40, 40), (1760, 560), (35, 65, 175), 8)
    cv2.putText(banner, "MEGA FLASH SALE - 70% OFF EVERYTHING", (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (35, 65, 175), 6)
    cv2.putText(banner, "LIMITED TIME PROMO BANNER - USE CODE SAVE70", (220, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 60, 60), 4)
    banner_path = source_dir / "Promo_Sale_Banner_2026.jpg"
    Image.fromarray(cv2.cvtColor(banner, cv2.COLOR_BGR2RGB)).save(banner_path, quality=85)

    # --- JUNK 5: Accidental Blurry Pocket / Floor Camera Shot ---
    blurry_cam = cv2.GaussianBlur(_draw_cafe_food(1200, 900), (45, 45), 16.0)
    blurry_path = source_dir / "IMG_4099.JPG"
    Image.fromarray(cv2.cvtColor(blurry_cam, cv2.COLOR_BGR2RGB)).save(blurry_path, quality=85, exif=exif_16pm)

    # --- VIDEOS: 2 Similar Video Takes + 1 Standalone Clip ---
    v1_path = source_dir / "IMG_7101.MP4"
    _write_sample_video(v1_path, width=1280, height=720, frames=45, color_shift=0, sharp=True)
    os.utime(v1_path, (base_ts + 9000, base_ts + 9000))

    v2_path = source_dir / "IMG_7102.MP4"
    _write_sample_video(v2_path, width=1280, height=720, frames=35, color_shift=2, sharp=False)
    os.utime(v2_path, (base_ts + 9012, base_ts + 9012))

    return source_dir


def _write_sample_video(path: Path, width: int = 1280, height: int = 720, frames: int = 30, color_shift: int = 0, sharp: bool = True):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    base_frame = _draw_mountain_sunset(width, height, seed_shift=color_shift)
    if not sharp:
        base_frame = cv2.GaussianBlur(base_frame, (19, 19), 5.0)
    for f in range(frames):
        frame = base_frame.copy()
        cv2.circle(frame, (180 + f * 6, int(height * 0.28)), 22, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
