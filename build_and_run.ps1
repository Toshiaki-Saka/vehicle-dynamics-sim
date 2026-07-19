<#
.SYNOPSIS
    Vehicle Dynamics Simulation — build, simulate, and animate in one command

.DESCRIPTION
    A quick-start script that runs the following steps in order:
      1. Build the C++ project        (CMake configure + build)
      2. Precompute the LQR gain      (python/compute_lqr_gain.py)
      3. Run the C++ simulation       (writes results.json)
      4. Plot the results             (visualize_results.py)
      5. Generate the animation GIF   (python/animation_demo.py)

    The Python-based steps (2, 4, 5) are skipped if Python is not found.

.PARAMETER SkipBuild
    Skip the C++ build (use when it is already built).

.PARAMETER SkipAnimation
    Skip generating the animation GIF (to save time).

.PARAMETER Python
    The Python command to use (default: py if available, otherwise python).

.EXAMPLE
    .\build_and_run.ps1

.EXAMPLE
    .\build_and_run.ps1 -SkipBuild -SkipAnimation
#>

[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipAnimation,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$scriptDir = $PSScriptRoot
$buildDir  = Join-Path $scriptDir 'build'

function Write-Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "==== [$n/$total] $msg ====" -ForegroundColor Cyan
}

# ── Resolve the Python command ───────────────────────────────────────
if (-not $Python) {
    if (Get-Command py     -ErrorAction SilentlyContinue) { $Python = 'py' }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $Python = 'python' }
}
$hasPython = [bool]$Python
if (-not $hasPython) {
    Write-Host "Warning: Python not found. Python-based steps will be skipped." -ForegroundColor Yellow
}

$TOTAL = 5

# ── [1/5] Build ──────────────────────────────────────────────────────
Write-Step 1 $TOTAL "C++ build (CMake)"
if ($SkipBuild) {
    Write-Host "  Skipping the build (-SkipBuild specified)."
} else {
    if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
    Write-Host "  configure... (the first run downloads Eigen / nlohmann-json and may take a few minutes)"
    cmake -S $scriptDir -B $buildDir -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
    Write-Host "  build..."
    cmake --build $buildDir --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed." }
}

# Resolve the executable path (varies by generator)
$exeCandidates = @(
    (Join-Path $buildDir 'Release\vehicle_dynamics.exe'),
    (Join-Path $buildDir 'vehicle_dynamics.exe'),
    (Join-Path $buildDir 'vehicle_dynamics')
)
$exe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { throw "Executable not found. Check that the build completed successfully." }

# ── [2/5] Precompute the LQR gain ────────────────────────────────────
Write-Step 2 $TOTAL "LQR gain computation (compute_lqr_gain.py)"
if ($hasPython) {
    & $Python (Join-Path $scriptDir 'python\compute_lqr_gain.py')
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Warning: LQR gain computation failed. The C++ build continues with zero gain." -ForegroundColor Yellow
    }
} else {
    Write-Host "  Skipped (no Python; the C++ build continues with zero gain)."
}

# ── [3/5] Run the simulation ─────────────────────────────────────────
Write-Step 3 $TOTAL "Run the simulation"
Write-Host "  $exe"
& $exe
if ($LASTEXITCODE -ne 0) { throw "The simulation failed to run." }

# ── [4/5] Plot the results ───────────────────────────────────────────
Write-Step 4 $TOTAL "Plot the results (visualize_results.py)"
$resultsJson = Join-Path $scriptDir 'results.json'
if ($hasPython) {
    if (Test-Path $resultsJson) {
        & $Python (Join-Path $scriptDir 'visualize_results.py') $resultsJson
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Warning: plotting failed." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  results.json not found. Skipping." -ForegroundColor Yellow
    }
} else {
    Write-Host "  Skipped (no Python)."
}

# ── [5/5] Generate the animation ─────────────────────────────────────
Write-Step 5 $TOTAL "Generate the animation (animation_demo.py)"
if ($SkipAnimation) {
    Write-Host "  Skipping (-SkipAnimation specified)."
} elseif ($hasPython) {
    & $Python (Join-Path $scriptDir 'python\animation_demo.py')
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Warning: animation generation failed (Pillow is required)." -ForegroundColor Yellow
    }
} else {
    Write-Host "  Skipped (no Python)."
}

# ── Done ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==== Done ====" -ForegroundColor Green
Write-Host "Generated files (project root):"
foreach ($f in @('results.json', 'cpp_results.png', 'cpp_results_ship_track.png', 'animation_demo.gif')) {
    $p = Join-Path $scriptDir $f
    if (Test-Path $p) { Write-Host ("  {0}" -f $f) -ForegroundColor Green }
}
