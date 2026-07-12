<#
.SYNOPSIS
    Vehicle Dynamics Simulation — ビルド・シミュレーション・アニメーションを一括実行

.DESCRIPTION
    以下を順に実行するクイックスタート用スクリプトです。
      1. C++ プロジェクトのビルド (CMake configure + build)
      2. LQR ゲインの事前計算              (python/compute_lqr_gain.py)
      3. C++ シミュレーションの実行         (results.json 出力)
      4. 結果のグラフ化                     (visualize_results.py)
      5. アニメーション GIF の生成          (python/animation_demo.py)

    Python を使うステップ (2, 4, 5) は Python が見つからない場合スキップします。

.PARAMETER SkipBuild
    C++ のビルドを省略します (既にビルド済みの場合)。

.PARAMETER SkipAnimation
    アニメーション GIF の生成を省略します (時間短縮)。

.PARAMETER Python
    使用する Python 実行コマンド (既定: py があれば py、なければ python)。

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

# ── Python 実行コマンドの解決 ────────────────────────────────────────
if (-not $Python) {
    if (Get-Command py     -ErrorAction SilentlyContinue) { $Python = 'py' }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $Python = 'python' }
}
$hasPython = [bool]$Python
if (-not $hasPython) {
    Write-Host "警告: Python が見つかりません。Python を使うステップはスキップします。" -ForegroundColor Yellow
}

$TOTAL = 5

# ── [1/5] ビルド ─────────────────────────────────────────────────────
Write-Step 1 $TOTAL "C++ ビルド (CMake)"
if ($SkipBuild) {
    Write-Host "  -SkipBuild 指定によりビルドをスキップします。"
} else {
    if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
    Write-Host "  configure... (初回は Eigen / nlohmann-json のダウンロードで数分かかります)"
    cmake -S $scriptDir -B $buildDir -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "CMake configure に失敗しました。" }
    Write-Host "  build..."
    cmake --build $buildDir --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "ビルドに失敗しました。" }
}

# 実行ファイルのパスを解決 (ジェネレータにより異なる)
$exeCandidates = @(
    (Join-Path $buildDir 'Release\vehicle_dynamics.exe'),
    (Join-Path $buildDir 'vehicle_dynamics.exe'),
    (Join-Path $buildDir 'vehicle_dynamics')
)
$exe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { throw "実行ファイルが見つかりません。ビルドが完了しているか確認してください。" }

# ── [2/5] LQR ゲインの事前計算 ──────────────────────────────────────
Write-Step 2 $TOTAL "LQR ゲイン計算 (compute_lqr_gain.py)"
if ($hasPython) {
    & $Python (Join-Path $scriptDir 'python\compute_lqr_gain.py')
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  警告: LQR ゲイン計算に失敗しました。C++ 版はゼロゲインで続行します。" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Python 無しのためスキップ (C++ 版はゼロゲインで続行します)。"
}

# ── [3/5] シミュレーション実行 ──────────────────────────────────────
Write-Step 3 $TOTAL "シミュレーション実行"
Write-Host "  $exe"
& $exe
if ($LASTEXITCODE -ne 0) { throw "シミュレーションの実行に失敗しました。" }

# ── [4/5] 結果のグラフ化 ────────────────────────────────────────────
Write-Step 4 $TOTAL "結果のグラフ化 (visualize_results.py)"
$resultsJson = Join-Path $scriptDir 'results.json'
if ($hasPython) {
    if (Test-Path $resultsJson) {
        & $Python (Join-Path $scriptDir 'visualize_results.py') $resultsJson
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  警告: グラフ化に失敗しました。" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  results.json が見つかりません。スキップします。" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Python 無しのためスキップ。"
}

# ── [5/5] アニメーション生成 ────────────────────────────────────────
Write-Step 5 $TOTAL "アニメーション生成 (animation_demo.py)"
if ($SkipAnimation) {
    Write-Host "  -SkipAnimation 指定によりスキップします。"
} elseif ($hasPython) {
    & $Python (Join-Path $scriptDir 'python\animation_demo.py')
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  警告: アニメーション生成に失敗しました (Pillow が必要です)。" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Python 無しのためスキップ。"
}

# ── 完了 ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==== 完了 ====" -ForegroundColor Green
Write-Host "生成物 (プロジェクトルート):"
foreach ($f in @('results.json', 'cpp_results.png', 'cpp_results_ship_track.png', 'animation_demo.gif')) {
    $p = Join-Path $scriptDir $f
    if (Test-Path $p) { Write-Host ("  {0}" -f $f) -ForegroundColor Green }
}
