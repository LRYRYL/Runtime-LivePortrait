<#
.SYNOPSIS
    One-click setup for Runtime-LivePortrait.

.DESCRIPTION
    Gets a machine from "nothing installed" to "running app" with as little manual work
    as possible, while keeping the repository free of anything that must not be
    redistributed.

    What it does:
      1. locate Python 3.10, or install it per-user if absent
      2. create a virtual environment at .venv inside the project
      3. fetch the dependency wheels (about 3 GB, once) -- or reuse an existing folder
      4. install the dependencies from that folder, with no further network access
      5. download the model weights into LivePortrait/pretrained_weights
      6. verify the result

    What it deliberately does NOT do:
      * ship or bundle model weights, Python installers, or PyPI wheels in the repo
      * install anything system-wide, or require administrator rights
      * touch any other Python on the machine

    The weights are fetched from Hugging Face at run time rather than committed, because
    the InsightFace models are licensed for non-commercial research only and are not ours
    to redistribute. See THIRD_PARTY_LICENSES.md.

.NOTES
    Environment variables:
      LPR_PYTHON    path to a python.exe to use, skipping discovery
      LPR_SOURCE    'cn' (default) or 'pypi' -- which package index to download from
      LPR_CUDA      'auto' (default), 'cu118' or 'cu128' -- force the CUDA stack
      LPR_OFFLINE   1 = never download; fail instead
#>
[CmdletBinding()]
param(
    [switch]$SkipWeights,
    [switch]$SkipWheels,      # use an existing wheel folder only
    [switch]$Reinstall,       # rebuild the venv from scratch
    [ValidateSet('auto', 'cu118', 'cu128')]
    [string]$Cuda = 'auto'    # force a CUDA stack instead of detecting the GPU
)

$ErrorActionPreference = 'Stop'
$Root    = Split-Path -Parent $PSScriptRoot          # project root
$Tools   = Join-Path $Root 'tools'
$Venv    = Join-Path $Root '.venv'
$VenvPy  = Join-Path $Venv 'Scripts\python.exe'
$Wt      = Join-Path $Root 'LivePortrait\pretrained_weights'

$Source  = if ($env:LPR_SOURCE) { $env:LPR_SOURCE } else { 'cn' }
$Offline = ($env:LPR_OFFLINE -eq '1')
# -Cuda wins if given explicitly; otherwise fall back to LPR_CUDA, then auto-detect
$Cuda    = if ($Cuda -ne 'auto') { $Cuda }
           elseif ($env:LPR_CUDA) { $env:LPR_CUDA }
           else { 'auto' }

function Head($n) { Write-Host ''; Write-Host "=== $n ===" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [ OK ] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Bad($m)  { Write-Host "  [FAIL] $m" -ForegroundColor Red }
function Info($m) { Write-Host "         $m" }

# Stops the run with something a non-expert can act on, instead of a stack trace or a
# vague failure several GB into the download.
function Abort-With([string]$problem, [string[]]$fixes) {
    Write-Host ''
    Write-Host '============================================================' -ForegroundColor Red
    Write-Host '  Cannot continue on this PC' -ForegroundColor Red
    Write-Host '============================================================' -ForegroundColor Red
    Write-Host ''
    Write-Host "  Problem: $problem" -ForegroundColor Yellow
    Write-Host ''
    if ($fixes) {
        Write-Host '  What to do:'
        foreach ($f in $fixes) { Write-Host "    - $f" }
        Write-Host ''
    }
    Write-Host '  Nothing was installed. Nothing was changed.' -ForegroundColor DarkGray
    Write-Host ''
    exit 1
}

Write-Host ''
Write-Host '============================================================' -ForegroundColor White
Write-Host '  Runtime-LivePortrait  -  one-click setup' -ForegroundColor White
Write-Host '============================================================' -ForegroundColor White
Info "project : $Root"
Info "venv    : $Venv"
if ($Offline) { Warn 'LPR_OFFLINE=1 - nothing will be downloaded' }

# ------------------------------------------------------------- preflight checks
# Everything here is checked BEFORE anything is downloaded, because the download is
# several GB and a beginner should not have to wait for it to discover their PC cannot
# run the app at all.
Head 'Preflight checks'

# 1) 64-bit Windows
$is64 = [Environment]::Is64BitOperatingSystem
if ($is64) {
    Ok "Windows 64-bit"
} else {
    Abort-With 'This is a 32-bit Windows installation.' @(
        'The GPU libraries this project needs are 64-bit only.',
        'Install 64-bit Windows 10 or 11, then run this again.'
    )
}

# 2) an NVIDIA GPU must exist -- there is no fallback, so say so plainly
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$gpuName = ''
$capMajor = $null
$capMinor = $null
$vramGB = 0.0

if (-not $smi) {
    Abort-With 'No NVIDIA graphics card was detected.' @(
        'This project needs an NVIDIA GPU. AMD and Intel graphics are not supported,',
        'and there is no CPU-only mode -- it would be far too slow to be usable.',
        'If you DO have an NVIDIA card, install its driver from nvidia.com and retry:',
        'the driver provides nvidia-smi, which this check looks for.'
    )
}

try {
    $line = (& $smi.Source --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>$null |
             Select-Object -First 1)
    if ($line -match '^\s*([^,]+),\s*(\d+)\.(\d+),\s*(\d+)\s*MiB') {
        $gpuName  = $Matches[1].Trim()
        $capMajor = [int]$Matches[2]
        $capMinor = [int]$Matches[3]
        $vramGB   = [double]$Matches[4] / 1024
    }
} catch { }

if (-not $gpuName) {
    Warn 'nvidia-smi is present but did not report a usable GPU.'
    Warn 'The graphics driver may be missing or too old -- this often still installs,'
    Warn 'but the app will not be able to run. Consider updating the driver first.'
} elseif ($capMajor -lt 6) {
    Abort-With "$gpuName is too old (compute capability sm_$capMajor$capMinor)." @(
        'PyTorch needs at least sm_60, and this project is tested from RTX 20 series up.',
        'A GTX 10-series or older card will not work.'
    )
} elseif ($vramGB -gt 0 -and $vramGB -lt 6) {
    Abort-With "$gpuName has only $([math]::Round($vramGB,1)) GB of VRAM." @(
        'At least 6 GB of VRAM is required.',
        'The models themselves need roughly 1.3 GB, but the camera pipeline and the',
        'browser need more on top of that.',
        'A 4 GB card is not supported.'
    )
} else {
    Ok "$gpuName  (sm_$capMajor$capMinor, $([math]::Round($vramGB,1)) GB)"
}

# 3) disk space -- measured need is roughly 10 GB in total
$needGB = 10
$drive = (Get-Item $Root).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
if ($freeGB -lt $needGB) {
    Warn "only $freeGB GB free on $($drive.Name): -- about $needGB GB is needed."
    Warn 'The dependency download alone is roughly 3 GB, plus about 5 GB for .venv'
    Warn 'and 0.6 GB for the model weights. Free up some space and run this again.'
} else {
    Ok "$freeGB GB free on $($drive.Name):  (about $needGB GB needed)"
}

Write-Host ''

# ---------------------------------------------------------- 0) pick the CUDA stack
#  No single PyTorch build covers every NVIDIA architecture: cu118 stops at sm_90 and
#  therefore cannot run on an RTX 50 card at all, while cu128 has no sm_89 cubin and is
#  measurably slower on an RTX 40 card. So the stack is chosen from the GPU here, once,
#  and everything downstream (which wheels to fetch, what to install) follows from it.
Head '0) GPU and CUDA stack'

$gpuScript = Join-Path $Tools 'gpu_choice.py'
$stack = 'cu118'

# detection needs a python; prefer one we already found, else any on PATH
$probePython = $null
if ($env:LPR_PYTHON -and (Test-Path $env:LPR_PYTHON)) { $probePython = $env:LPR_PYTHON }
elseif (Test-Path $VenvPy) { $probePython = $VenvPy }
else {
    $c = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($c) { $probePython = $c.Source }
}

if ($Cuda -ne 'auto') {
    $stack = $Cuda
    Warn "CUDA stack forced to $stack by LPR_CUDA"
} elseif ($probePython -and (Test-Path $gpuScript)) {
    & $probePython $gpuScript 2>&1 | ForEach-Object { Info $_ }
    $detected = (& $probePython $gpuScript --name 2>$null | Select-Object -Last 1)
    if ($detected -match '^cu(118|128)$') { $stack = $detected.Trim() }
} else {
    Warn 'cannot detect the GPU yet (no python available); assuming cu118'
}

$Wheels = Join-Path $Root ".venv-wheels\$stack"
Ok "CUDA stack: $stack"
Info "wheels  : $Wheels"
if ($stack -eq 'cu128') {
    Info 'cu128 is required for sm_90 and above (RTX 50 series).'
    Info 'Note: on an RTX 40 card cu118 would be about 12% faster.'
}

# ---------------------------------------------------------------- 1) Python 3.10
Head '1) Python 3.10'

function Test-Py310([string]$exe) {
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    try {
        $v = & $exe -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        return ($v -eq '3.10')
    } catch { return $false }
}

$python = $null
$why = ''

if ($env:LPR_PYTHON) {
    if (Test-Py310 $env:LPR_PYTHON) { $python = $env:LPR_PYTHON; $why = 'from LPR_PYTHON' }
    else { Warn "LPR_PYTHON is not Python 3.10: $env:LPR_PYTHON" }
}
if (-not $python -and (Test-Py310 $VenvPy)) { $python = $VenvPy; $why = 'existing .venv' }
if (-not $python) {
    $cands = @()
    foreach ($base in @("$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles",
                        "${env:ProgramFiles(x86)}", 'C:\')) {
        if ($base -and (Test-Path $base)) {
            $cands += (Get-ChildItem $base -Directory -Filter 'Python310*' -ErrorAction SilentlyContinue |
                       ForEach-Object { Join-Path $_.FullName 'python.exe' })
        }
    }
    $onPath = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($onPath) { $cands += $onPath.Source }
    foreach ($c in $cands) {
        if (Test-Py310 $c) { $python = $c; $why = 'already installed'; break }
    }
}

if ($python) {
    Ok "found Python 3.10 ($why)"
    Info $python
} else {
    Warn 'no Python 3.10 found - installing it per-user (no admin required)'
    if ($Offline) { Bad 'offline mode and no Python 3.10 available'; exit 2 }

    $target = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310'
    $url = 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe'
    $dl = Join-Path $env:TEMP 'python-3.10.11-amd64.exe'

    # python.org is slow or unreachable from some networks; try mirrors first
    $urls = @(
        'https://mirrors.huaweicloud.com/python/3.10.11/python-3.10.11-amd64.exe',
        'https://mirrors.tuna.tsinghua.edu.cn/python/3.10.11/python-3.10.11-amd64.exe',
        $url
    )
    $got = $false
    foreach ($u in $urls) {
        Info "trying $u"
        try {
            Invoke-WebRequest -Uri $u -OutFile $dl -UseBasicParsing -TimeoutSec 300
            if ((Get-Item $dl).Length -gt 20MB) { $got = $true; break }
            Warn 'file too small, trying the next mirror'
        } catch { Warn "failed: $($_.Exception.Message)" }
    }
    if (-not $got) {
        Bad 'could not download the Python installer from any mirror'
        Info 'Install Python 3.10 from https://www.python.org/downloads/ then re-run.'
        exit 2
    }

    Info "installing to $target (a minute or two)"
    $args = @('/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0',
              'Include_test=0', 'Include_doc=0', "TargetDir=$target")
    $p = Start-Process -FilePath $dl -ArgumentList $args -Wait -PassThru
    if ($p.ExitCode -ne 0) { Bad "installer exited with $($p.ExitCode)"; exit 2 }

    if (Test-Py310 (Join-Path $target 'python.exe')) {
        $python = Join-Path $target 'python.exe'
        Ok "installed Python 3.10"
    } else {
        Bad 'Python install finished but 3.10 was not found'
        exit 2
    }
}

# ---------------------------------------------------------------- 2) venv
Head '2) virtual environment'

if ($Reinstall -and (Test-Path $Venv)) {
    Warn 'removing the existing .venv (-Reinstall)'
    Remove-Item $Venv -Recurse -Force
}
if (Test-Py310 $VenvPy) {
    Ok 'already exists, reusing'
} else {
    Info "creating $Venv"
    & $python -m venv $Venv
    if (-not (Test-Py310 $VenvPy)) {
        Bad 'venv creation failed'
        Info 'If your python came from the Microsoft Store, install the full build'
        Info 'from python.org instead -- the Store shim cannot create venvs reliably.'
        exit 3
    }
    Ok 'created'
}
& $VenvPy -m pip --version 2>&1 | Select-Object -First 1 | ForEach-Object { Info $_ }

# ---------------------------------------------------------------- 3) wheels
Head '3) dependency wheels'

$wheelFiles = @()
if (Test-Path $Wheels) { $wheelFiles = @(Get-ChildItem $Wheels -Filter '*.whl' -ErrorAction SilentlyContinue) }

if ($wheelFiles.Count -ge 50) {
    Ok ("reusing {0} wheels already in .venv-wheels\{1} ({2:N2} GB)" -f `
        $wheelFiles.Count, $stack, (($wheelFiles | Measure-Object Length -Sum).Sum / 1GB))
} elseif ($SkipWheels) {
    Bad 'no wheel folder and -SkipWheels was given'
    exit 4
} elseif ($Offline) {
    Bad "offline mode but .venv-wheels\$stack is empty"
    exit 4
} else {
    Warn "downloading the $stack dependencies - a few GB, and it happens once"
    Info "index: $(if ($Source -eq 'pypi') { 'PyPI' } else { 'Tsinghua mirror' })"
    & $VenvPy (Join-Path $Tools 'fetch_wheels.py') --dest $Wheels --source $Source --cuda $stack
    if ($LASTEXITCODE -ne 0) {
        Bad 'wheel download failed'
        Info "Retry with:  `"$VenvPy`" `"$Tools\fetch_wheels.py`" --source pypi --cuda $stack"
        exit 4
    }
    $wheelFiles = @(Get-ChildItem $Wheels -Filter '*.whl')
    Ok ("{0} wheels ready" -f $wheelFiles.Count)
}

# ---------------------------------------------------------------- 4) install
Head '4) installing dependencies'

$local = @('--no-index', '--find-links', $Wheels)
Info 'installing from the local wheel folder (no network)'

# The CUDA stack first, so its torch/onnxruntime pair is what ends up installed.
& $VenvPy -m pip install --no-warn-script-location @local -r (Join-Path $Tools "requirements-$stack.txt")
if ($LASTEXITCODE -ne 0) { Bad "$stack install failed"; exit 5 }

# Then the shared dependencies. Installing these AFTER torch is what keeps numpy on
# 1.26.4: torch pulls numpy 2.x on its own, and LivePortrait is not compatible with it.
& $VenvPy -m pip install --no-warn-script-location @local -r (Join-Path $Tools 'requirements-common.txt')
if ($LASTEXITCODE -ne 0) { Bad 'dependency install failed'; exit 5 }
Ok 'dependencies installed'

# A wrong stack is silent otherwise: torch imports fine, and only the GPU is unusable.
$verify = & $VenvPy -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" 2>&1
Info "torch: $verify"
if ($verify -notmatch 'True') {
    Bad 'torch reports CUDA is NOT available.'
    Info "This machine was given the $stack stack. If that is wrong, re-run with:"
    Info "  `$env:LPR_CUDA='cu128'; .\tools\setup.ps1 -Reinstall"
    Info 'An RTX 50 (sm_120) card requires cu128; cu118 cannot target it.'
}

# ---------------------------------------------------------------- 5) weights
Head '5) model weights'

$required = @(
    'liveportrait\base_models\appearance_feature_extractor.pth',
    'liveportrait\base_models\motion_extractor.pth',
    'liveportrait\base_models\spade_generator.pth',
    'liveportrait\base_models\warping_module.pth',
    'liveportrait\retargeting_models\stitching_retargeting_module.pth',
    'liveportrait\landmark.onnx',
    'insightface\models\buffalo_l\2d106det.onnx',
    'insightface\models\buffalo_l\det_10g.onnx'
)
$missing = @($required | Where-Object { -not (Test-Path (Join-Path $Wt $_)) })

if ($missing.Count -eq 0) {
    Ok 'all 8 weight files present'
} elseif ($SkipWeights) {
    Warn "$($missing.Count) missing, -SkipWeights was given"
} elseif ($Offline) {
    Bad "offline and $($missing.Count) weight file(s) missing"
    exit 6
} else {
    Warn "downloading $($missing.Count) weight file(s) - about 667 MB"
    & $VenvPy (Join-Path $Tools 'download_weights.py')
    if ($LASTEXITCODE -ne 0) {
        Bad 'weight download failed'
        Info "Retry:  `"$VenvPy`" `"$Tools\download_weights.py`""
        exit 6
    }
    $still = @($required | Where-Object { -not (Test-Path (Join-Path $Wt $_)) })
    if ($still.Count -gt 0) { Bad "$($still.Count) weight file(s) still missing"; exit 6 }
    Ok 'weights ready'
}

# ---------------------------------------------------------------- 6) verify
Head '6) verification'

& $VenvPy -c @"
import sys
print('  python      ', sys.version.split()[0])
try:
    import torch
    print('  torch       ', torch.__version__)
    print('  cuda        ', torch.cuda.is_available())
    if not torch.cuda.is_available():
        print('                NOTE: no CUDA device - an NVIDIA GPU and driver are required')
except Exception as e:
    print('  torch       FAILED:', e)
for mod in ('numpy', 'cv2', 'onnxruntime', 'win32api'):
    try:
        m = __import__(mod)
        print('  %-12s%s' % (mod, getattr(m, '__version__', 'ok')))
    except Exception as e:
        print('  %-12sFAILED: %s' % (mod, e))
"@

Head 'done'
Ok 'setup finished'
Info 'start the app by double-clicking Start_WebUI.bat, or run:'
Info "  `"$VenvPy`" app.py"
Write-Host ''
exit 0
