@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PY=py"
set "SCRIPT_DIR=%~dp0"

REM Always run from project root so imports work
pushd "%SCRIPT_DIR%"

REM Share file path used by BOTH tracker + animation
set "TRACK_SHARE_FILE=%SCRIPT_DIR%logs\pupil_share.json"

REM Camera / tracker controls
set "CAMERA_ID=0"
set "ROTATE_90_CW=1"
set "TARGET_FPS=60"
set "DISABLE_VIZ=1"

REM Animation controls
set "TRACK_SENSITIVITY_X=2.8"
set "TRACK_SENSITIVITY_Y=2.4"
set "SMOOTHING=0.18"
set "MAX_OFFSET_FRAC=0.55"
set "DEADZONE=0.02"
set "CALIBRATE_ON_START=1"
set "FPS=60"

if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

echo =========================================
echo Eye Animation (Clean) + 2-pass Tracker
echo Share file: %TRACK_SHARE_FILE%
echo Camera ID:  %CAMERA_ID%
echo =========================================
echo.

REM Keep tracker window open so you can see errors/logs
start "PupilTracker2Pass" cmd /k "%PY% -m eye_pipeline.live_tracker_2pass"

REM Run animation in current window
"%PY%" "%SCRIPT_DIR%eye_animation_clean.py"

popd
endlocal
