# 📸 iPhone 16 Pro Max & 18 Pro Max — Smart Photo & Video Cleaner (Windows 11 Edition)

A local AI-powered desktop web tool designed for **Windows 11 (i7 / 16GB RAM)** to clean, deduplicate, and organize a **combined photo & video backup folder** from both your **iPhone 16 Pro Max** and **iPhone 18 Pro Max** over USB-C — **without needing iCloud**.

---

## ✨ Key Features (Mode B: Full Backup + Smart Prune)

1. **Unified Combined Folder Support (`16 Pro Max` + `18 Pro Max`):**
   - Put photos and videos from both iPhones into a single folder on your PC.
   - Automatically pairs **Live Photos** (`.HEIC` + `.MOV` with matching filenames) so both parts stay or move together.
2. **Cross-Phone Similarity & Burst Clustering:**
   - Uses perceptual hashing (`pHash`, `dHash`, `colorhash`) + EXIF capture timestamps to group burst shots, multi-takes, and duplicates across both phones.
3. **AI Photo Quality Scoring (`0–100`):**
   - **Sharpness / Blur Score:** Laplacian variance + Sobel edge gradients to catch motion blur or out-of-focus shots.
   - **Exposure & Dynamic Range:** Penalizes dark/underexposed or blown-out frames.
   - **Face & Open-Eye Detection:** OpenCV Haar cascades score face clarity and open eyes.
   - Automatically marks the highest-scoring photo in every group as **🌟 KEEP ON IPHONE** and marks the rest as **📦 MOVE TO PC ARCHIVE**.
4. **Smart Clutter & Social Junk Detector:**
   - 📱 **iPhone Screenshots:** Detects exact iPhone 16/18 Pro Max screen resolutions (`1320×2868`, `1344×2992`, etc.), PNG UI layouts, and EXIF `Screenshot` tags.
   - 💬 **WhatsApp Forwards:** Detects WhatsApp naming patterns (`IMG-YYYYMMDD-WAxxxx`), stripped camera EXIF, and standard WhatsApp compression dimensions.
   - 🎉 **Wish Messages ("Happy New Year", "Good Morning", Festival Cards):** Detects high-saturation greeting graphics with dense overlaid typography and no camera lens EXIF.
   - 🪧 **Promotional Banners & Flyers:** Detects extreme aspect ratios and poster text blocks over flat graphic backgrounds.
   - 🌫️ **Blurry Camera Shots:** Flags severely out-of-focus camera shots.
5. **Zero-Data-Loss SHA-256 Archive + Windows 11 USB-C iPhone Prune:**
   - Moves duplicates and clutter into organized subfolders inside `PC_Archive_Moved\` and **verifies SHA-256 checksums** before removing the source file.
   - Generates `prune_connected_iphone.ps1` + `iphone_delete_manifest.txt` so you can plug in your iPhone 16 Pro Max and 18 Pro Max via USB-C and automatically delete *only* the archived duplicates/junk from the iPhone's `DCIM` storage!

---

## 🚀 Quick Start on Windows 11

### Step 1: One-Time iPhone USB-C Setting
On both your **iPhone 16 Pro Max** and **iPhone 18 Pro Max**:
1. Open **Settings → Photos**.
2. Scroll to the bottom under **"Transfer to Mac or PC"** and select **"Keep Originals"**.
   *(This prevents Windows USB-C transfers from freezing and allows clean file deletion over USB-C.)*

### Step 2: Copy Photos from Both iPhones into One PC Folder
1. Connect each iPhone via USB-C, unlock the screen, and tap **Trust**.
2. Open **File Explorer → This PC → Apple iPhone → Internal Storage → DCIM**.
3. Copy the contents from both phones into one folder on your PC, e.g.:
   `C:\Users\YourName\Pictures\Combined_iPhone_Backup`

### Step 3: Run the App
Double-click **`start_windows.bat`** (or run in PowerShell/Terminal):
```powershell
git clone https://github.com/rammsec/iphone-photo-cleaner.git
cd iphone-photo-cleaner
.\start_windows.bat
```
Your browser will open `http://localhost:8090`.

### Step 4: Review, Archive & Prune iPhone over USB-C
1. Paste `C:\Users\YourName\Pictures\Combined_iPhone_Backup` into the **Combined Backup Source Folder** box and click **Scan Combined Folder**.
2. Review the **Similar Clusters**, **Clutter Detector**, and **Videos** tabs.
3. Click **"📦 Move Selected to PC Archive & Generate USB Prune"**.
4. To remove those exact archived files from your iPhones over USB-C, connect the iPhone and run the generated PowerShell script inside your archive folder:
   ```powershell
   powershell -ExecutionPolicy Bypass -File "C:\Users\YourName\Pictures\PC_Archive_Moved\prune_connected_iphone.ps1"
   ```
   *(Tip: Pass `-DryRun` first to preview every file it will delete on the iPhone!)*
