#!/usr/bin/env bash
# ============================================================
#  Vehicle Dynamics CMake Build Script (Linux / macOS)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "========================================"
echo " Vehicle Dynamics CMake Build Script"
echo "========================================"
echo
echo "[1/2] CMake configure..."
echo "      (first run downloads Eigen and nlohmann/json; this may"
echo "       take a few minutes depending on your connection)"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo
echo "[2/2] Build..."
cmake --build "${BUILD_DIR}" --config Release --parallel

echo
echo "Build succeeded!"
echo "Executable: ${BUILD_DIR}/vehicle_dynamics"
echo
echo "Next steps:"
echo "  ./build/vehicle_dynamics                       # run the simulation"
echo "  python visualize_results.py results.json       # plot the results"
