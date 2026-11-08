# =============================================================================
# 1-Click iPhone USB-C Photo & Video Puller for Windows 11 (Mode B)
# =============================================================================
# Automatically detects your connected iPhone (16 Pro Max or 18 Pro Max) in
# Windows 11 "This PC", walks Internal Storage -> DCIM, and copies all photos
# and videos into your combined local backup folder:
#   C:\Users\<You>\Pictures\Combined_iPhone_Backup
# =============================================================================

param (
    [string]$DestinationRoot = "$env:USERPROFILE\Pictures\Combined_iPhone_Backup"
)

$ErrorActionPreference = "Continue"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " iPhone USB-C Auto-Backup to Combined Folder (Windows 11)" -ForegroundColor Cyan
Write-Host " Target Folder: $DestinationRoot" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

if (-not (Test-Path $DestinationRoot)) {
    New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
}

$Shell = New-Object -ComObject Shell.Application
$ThisPC = $Shell.NameSpace(17) # This PC

$iPhoneItem = $null
foreach ($item in $ThisPC.Items()) {
    if ($item.Name -match "iPhone|Apple") {
        $iPhoneItem = $item
        break
    }
}

if (-not $iPhoneItem) {
    Write-Host ""
    Write-Host "[!] Could not see your iPhone in 'This PC' yet." -ForegroundColor Yellow
    Write-Host "Please check these 3 quick things on your iPhone:" -ForegroundColor Yellow
    Write-Host "  1. Unlock your iPhone screen with FaceID / Passcode."
    Write-Host "  2. If a popup says 'Trust This Computer?' or 'Allow this device to access photos and videos?', tap Trust / Allow."
    Write-Host "  3. Go to Settings -> Photos -> scroll to bottom ('Transfer to Mac or PC') -> select 'Keep Originals'."
    Write-Host ""
    exit 1
}

Write-Host "[OK] Connected to: $($iPhoneItem.Name)" -ForegroundColor Green
Write-Host "Scanning DCIM folders and copying new photos/videos to $DestinationRoot ..." -ForegroundColor Cyan

$DestNamespace = $Shell.NameSpace($DestinationRoot)

function Copy-FromMtpFolder($Folder) {
    if (-not $Folder) { return }
    $Items = @($Folder.Items())
    foreach ($item in $Items) {
        if ($item.IsFolder) {
            Copy-FromMtpFolder $item.GetFolder()
        } else {
            $FileName = $item.Name
            $ExistingPath = Join-Path $DestinationRoot $FileName
            # Check if file (or file with common extension) already exists in destination
            $AlreadyExists = (Test-Path $ExistingPath)
            if (-not $AlreadyExists) {
                foreach ($ext in @(".HEIC", ".JPG", ".JPEG", ".PNG", ".MOV", ".MP4")) {
                    if (Test-Path "$ExistingPath$ext") {
                        $AlreadyExists = $true
                        break
                    }
                }
            }
            if (-not $AlreadyExists) {
                Write-Host "  -> Copying: $FileName" -ForegroundColor Gray
                # 16 = Respond "Yes to All", 4 = Do not display progress dialog box for each single file
                $DestNamespace.CopyHere($item, 20)
            }
        }
    }
}

Copy-FromMtpFolder $iPhoneItem.GetFolder()

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host " Done! Your iPhone photos/videos are now in:" -ForegroundColor Green
Write-Host "   $DestinationRoot" -ForegroundColor White
Write-Host " You can now plug in your second iPhone and run this again, or" -ForegroundColor Green
Write-Host " scan that folder in the Web Dashboard at http://localhost:8090" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
