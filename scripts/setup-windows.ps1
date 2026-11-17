# First-run setup for the Windows release of SpeakMind server.
#
# Run once, from the folder you unzipped speakmind-server-windows-x64.zip into:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1
#
# What it does, in order:
#   1. ffmpeg   - needed to decode uploads. Installed with winget (Gyan.FFmpeg)
#                 if it is not already on PATH.
#   2. ONNX model (~1.2 GB) - exported with uv into models/, as in README.md.
#   3. Piper voices (optional, ~63 MB each) - downloaded into tts/.
#   4. espeak-ng - NOT installed automatically: copy espeak-ng.dll and
#                 espeak-ng-data/ into espeak/ (see the message at the end,
#                 and README.md). The server will not start without it.
#
# Nothing in this script needs an API key. AI chat stays off until you set
# SPEAKMIND_AI_API_KEY / SPEAKMIND_AI_BASE_URL, as described in README.md.

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

function Test-Command($name) {
  return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

# 1. ffmpeg ---------------------------------------------------------------
if (Test-Command ffmpeg) {
  Write-Host "ffmpeg: already on PATH"
} elseif (Test-Command winget) {
  Write-Host "ffmpeg: installing with winget (Gyan.FFmpeg)..."
  winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  # winget updates PATH for new shells; refresh it for this one.
  $env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
              [System.Environment]::GetEnvironmentVariable('Path', 'User')
  if (-not (Test-Command ffmpeg)) {
    Write-Warning "ffmpeg was installed but is not on PATH in this shell. Close and reopen PowerShell, then re-run this script."
  }
} else {
  Write-Warning "ffmpeg not found and winget is not available. Install it from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-essentials.zip), add it to PATH, then re-run this script."
}

# 2. ONNX model ------------------------------------------------------------
if ((Test-Path models/model.onnx) -and (Test-Path models/vocab.json) -and (Test-Path models/config.json)) {
  Write-Host "models/: already present"
} else {
  if (-not (Test-Command uv)) {
    Write-Host "uv: installing (https://docs.astral.sh/uv/)..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [System.Environment]::GetEnvironmentVariable('Path', 'User')
  }
  Write-Host "models/: exporting ONNX model (about 1.2 GB, this takes a while)..."
  uv run --no-project --python 3.12 --with torch==2.15.0 --with transformers==5.22.2 --with onnx python tools/export_onnx.py models
}

# 3. Piper voices (optional) ----------------------------------------------
New-Item -ItemType Directory -Force -Path tts | Out-Null
$voices = @(
  'en/en_US/lessac/medium/en_US-lessac-medium',
  'es/es_ES/davefx/medium/es_ES-davefx-medium'
)
foreach ($voice in $voices) {
  foreach ($suffix in @('.onnx', '.onnx.json')) {
    $file = Split-Path $voice -Leaf
    $dest = Join-Path tts "$file$suffix"
    if (Test-Path $dest) { continue }
    $url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/$voice$suffix"
    Write-Host "tts/: downloading $file$suffix ..."
    Invoke-WebRequest -Uri $url -OutFile $dest
  }
}

# 4. espeak-ng --------------------------------------------------------------
if ((Test-Path espeak/espeak-ng.dll) -and (Test-Path espeak/espeak-ng-data)) {
  Write-Host "espeak/: already present"
} else {
  Write-Host ""
  Write-Host "NEXT STEP - espeak-ng (required, not installed by this script):"
  Write-Host "  Download espeak-ng for Windows from https://github.com/espeak-ng/espeak-ng/releases"
  Write-Host "  and copy espeak-ng.dll and espeak-ng-data/ into the espeak/ folder next to speakmind-server.exe."
  Write-Host ""
}

Write-Host "Done. Start the server with: .\speakmind-server.exe"
Write-Host "Then check http://localhost:8000/health"
