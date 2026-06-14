@echo off
echo ========================================
echo  Vehicle Dynamics CMake Build Script
echo ========================================

set BUILD_DIR=%~dp0build

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo.
echo [1/2] CMake configure...
echo       (first run downloads Eigen and nlohmann/json;
echo        this may take a few minutes)
cmake -S "%~dp0" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo CMake configure failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Build...
cmake --build "%BUILD_DIR%" --config Release --parallel
if %ERRORLEVEL% neq 0 (
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Build succeeded!
echo Executable: %BUILD_DIR%\Release\vehicle_dynamics.exe
echo.
echo Next steps:
echo   python python\compute_lqr_gain.py            ^(optional: LQR gain^)
echo   build\Release\vehicle_dynamics.exe           ^(run the simulation^)
echo   python visualize_results.py results.json     ^(plot the results^)
pause
