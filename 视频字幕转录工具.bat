@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "CONDA_EXE="
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if not defined CONDA_EXE if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "CONDA_EXE=C:\ProgramData\miniconda3\Scripts\conda.exe"

if not defined CONDA_EXE (
  echo [ERROR] Conda was not found.
  echo Run setup_environment.ps1 after installing Miniconda.
  pause
  exit /b 1
)

"%CONDA_EXE%" run -n video-notebook python environment_check.py --backend sensevoice --device cuda
if errorlevel 1 (
  echo.
  echo [ERROR] The video-notebook environment is unavailable.
  echo Run: powershell -ExecutionPolicy Bypass -File setup_environment.ps1
  pause
  exit /b 1
)

start "" http://127.0.0.1:5000
"%CONDA_EXE%" run -n video-notebook --no-capture-output python web_app.py
