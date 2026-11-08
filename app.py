"""
FastAPI Backend for iPhone 16 Pro Max / 18 Pro Max Photo & Video Cleaner (Mode B: Full Backup + Smart Prune).
Starts listening on port 8090 immediately and runs the initial folder scan in a background thread.
Supports instant re-clustering when the user changes the Similarity % Threshold (default 95%) or Time Window!
"""

from __future__ import annotations

import io
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from analyzer import build_scan_response, scan_Backup_directory
from demo_generator import create_demo_workspace
from usb_archiver import execute_archive_plan

BASE_DIR = Path(__file__).parent.resolve()
DEMO_ROOT = BASE_DIR / "demo_workspace"

USER_PICTURES_BACKUP = Path.home() / "Pictures" / "Combined_iPhone_Backup"
USER_PICTURES_ARCHIVE = Path.home() / "Pictures" / "PC_Archive_Moved"

if USER_PICTURES_BACKUP.exists() and any(USER_PICTURES_BACKUP.iterdir()):
    DEFAULT_SOURCE_DIR = USER_PICTURES_BACKUP
    DEFAULT_ARCHIVE_DIR = USER_PICTURES_ARCHIVE
else:
    DEFAULT_SOURCE_DIR = DEMO_ROOT / "Combined_iPhone_Backup"
    DEFAULT_ARCHIVE_DIR = DEMO_ROOT / "PC_Archive_Moved"

app = FastAPI(title="iPhone 16/18 Pro Max Photo & Video Cleaner (Windows 11 Edition)")

SCAN_STATE: Dict = {
    "source_dir": str(DEFAULT_SOURCE_DIR),
    "archive_dir": str(DEFAULT_ARCHIVE_DIR),
    "min_similarity_pct": 95.0,
    "max_time_window_sec": 120.0,
    "scanning": False,
    "scan_error": None,
    "raw_items": [],
    "scan_result": None,
}


class ScanRequest(BaseModel):
    source_dir: str
    archive_dir: Optional[str] = None
    min_similarity_pct: float = 95.0
    max_time_window_sec: float = 120.0


class ReclusterRequest(BaseModel):
    min_similarity_pct: float = 95.0
    max_time_window_sec: float = 120.0


class ArchiveRequest(BaseModel):
    selected_ids: List[str]
    archive_dir: Optional[str] = None
    dry_run: bool = False
    prune_connected_iphone: bool = False


def _run_background_scan(source_dir: str, min_sim: float = 95.0, max_time: float = 120.0):
    SCAN_STATE["scanning"] = True
    SCAN_STATE["scan_error"] = None
    try:
        if source_dir == str(DEMO_ROOT / "Combined_iPhone_Backup"):
            create_demo_workspace(DEMO_ROOT)
        DEFAULT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        res, raw_items = scan_Backup_directory(
            source_dir,
            min_similarity_pct=min_sim,
            max_time_window_sec=max_time,
            return_raw_items=True,
        )
        SCAN_STATE["raw_items"] = raw_items
        SCAN_STATE["scan_result"] = res
    except Exception as exc:
        SCAN_STATE["scan_error"] = str(exc)
        print(f"[ERROR] Scan failed: {exc}", flush=True)
    finally:
        SCAN_STATE["scanning"] = False


@app.on_event("startup")
def startup_init():
    t = threading.Thread(
        target=_run_background_scan,
        args=(SCAN_STATE["source_dir"], SCAN_STATE["min_similarity_pct"], SCAN_STATE["max_time_window_sec"]),
        daemon=True,
    )
    t.start()


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/state")
def get_current_state():
    return {
        "source_dir": SCAN_STATE["source_dir"],
        "archive_dir": SCAN_STATE["archive_dir"],
        "min_similarity_pct": SCAN_STATE["min_similarity_pct"],
        "max_time_window_sec": SCAN_STATE["max_time_window_sec"],
        "scanning": SCAN_STATE["scanning"],
        "scan_error": SCAN_STATE["scan_error"],
        "data": SCAN_STATE["scan_result"],
    }


@app.post("/api/recluster")
def recluster_cached_items(req: ReclusterRequest):
    """
    Instantly re-runs the similarity clustering using cached photo features (no disk re-scan needed!).
    """
    if SCAN_STATE["scanning"] or not SCAN_STATE["raw_items"]:
        raise HTTPException(status_code=400, detail="Scan is still in progress or no items loaded.")

    SCAN_STATE["min_similarity_pct"] = req.min_similarity_pct
    SCAN_STATE["max_time_window_sec"] = req.max_time_window_sec

    updated = build_scan_response(
        Path(SCAN_STATE["source_dir"]),
        SCAN_STATE["raw_items"],
        min_similarity_pct=req.min_similarity_pct,
        max_time_window_sec=req.max_time_window_sec,
    )
    SCAN_STATE["scan_result"] = updated
    return {
        "source_dir": SCAN_STATE["source_dir"],
        "archive_dir": SCAN_STATE["archive_dir"],
        "min_similarity_pct": SCAN_STATE["min_similarity_pct"],
        "max_time_window_sec": SCAN_STATE["max_time_window_sec"],
        "scanning": False,
        "data": updated,
    }


@app.post("/api/scan")
def trigger_scan(req: ScanRequest):
    src = Path(req.source_dir).expanduser().resolve()
    if not src.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {src}")

    SCAN_STATE["source_dir"] = str(src)
    SCAN_STATE["min_similarity_pct"] = req.min_similarity_pct
    SCAN_STATE["max_time_window_sec"] = req.max_time_window_sec
    if req.archive_dir:
        SCAN_STATE["archive_dir"] = str(Path(req.archive_dir).expanduser().resolve())
    else:
        SCAN_STATE["archive_dir"] = str(src.parent / "PC_Archive_Moved")

    if not SCAN_STATE["scanning"]:
        t = threading.Thread(
            target=_run_background_scan,
            args=(str(src), req.min_similarity_pct, req.max_time_window_sec),
            daemon=True,
        )
        t.start()

    return {
        "source_dir": SCAN_STATE["source_dir"],
        "archive_dir": SCAN_STATE["archive_dir"],
        "min_similarity_pct": SCAN_STATE["min_similarity_pct"],
        "max_time_window_sec": SCAN_STATE["max_time_window_sec"],
        "scanning": True,
        "data": SCAN_STATE["scan_result"],
    }


@app.post("/api/reset-demo")
def reset_demo_workspace():
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    create_demo_workspace(DEMO_ROOT)
    DEMO_ARCHIVE = DEMO_ROOT / "PC_Archive_Moved"
    DEMO_ARCHIVE.mkdir(parents=True, exist_ok=True)
    SCAN_STATE["source_dir"] = str(DEMO_ROOT / "Combined_iPhone_Backup")
    SCAN_STATE["archive_dir"] = str(DEMO_ARCHIVE)
    res, raw_items = scan_Backup_directory(
        SCAN_STATE["source_dir"],
        min_similarity_pct=SCAN_STATE["min_similarity_pct"],
        max_time_window_sec=SCAN_STATE["max_time_window_sec"],
        return_raw_items=True,
    )
    SCAN_STATE["raw_items"] = raw_items
    SCAN_STATE["scan_result"] = res
    return {
        "source_dir": SCAN_STATE["source_dir"],
        "archive_dir": SCAN_STATE["archive_dir"],
        "min_similarity_pct": SCAN_STATE["min_similarity_pct"],
        "max_time_window_sec": SCAN_STATE["max_time_window_sec"],
        "scanning": False,
        "data": res,
    }


@app.post("/api/archive")
def archive_selected_media(req: ArchiveRequest):
    if not SCAN_STATE["scan_result"]:
        raise HTTPException(status_code=400, detail="No active scan. Please wait for scan to finish.")

    archive_dir = req.archive_dir or SCAN_STATE["archive_dir"]
    items_by_id = SCAN_STATE["scan_result"]["items"]

    res = execute_archive_plan(
        items_by_id=items_by_id,
        selected_ids=req.selected_ids,
        archive_dir_str=archive_dir,
        dry_run=req.dry_run,
        prune_connected_iphone=req.prune_connected_iphone,
    )

    if not req.dry_run:
        moved_set = set(req.selected_ids)
        SCAN_STATE["raw_items"] = [it for it in SCAN_STATE["raw_items"] if it.id not in moved_set]
        SCAN_STATE["scan_result"] = build_scan_response(
            Path(SCAN_STATE["source_dir"]),
            SCAN_STATE["raw_items"],
            min_similarity_pct=SCAN_STATE["min_similarity_pct"],
            max_time_window_sec=SCAN_STATE["max_time_window_sec"],
        )

    return {
        "archive_result": res,
        "updated_data": SCAN_STATE["scan_result"],
    }


@app.get("/api/thumbnail/{item_id}")
def get_media_thumbnail(item_id: str, size: int = Query(520, ge=120, le=1600)):
    if not SCAN_STATE["scan_result"]:
        raise HTTPException(status_code=404, detail="Not scanned")
    item = SCAN_STATE["scan_result"]["items"].get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Media item not found")

    filepath = item["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found on disk")

    try:
        if item["media_type"] == "video":
            cap = cv2.VideoCapture(filepath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > 5:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.2))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                img = Image.new("RGB", (400, 300), (30, 30, 40))
            else:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(filepath).convert("RGB")

        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=86)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
