@echo off
setlocal
cd /d "%~dp0"
if exist ".detector-venv\Scripts\python.exe" goto run
for %%V in (3.12 3.14 3.13) do (
  py -%%V -m venv .detector-venv && goto run
)
echo Install 64-bit Python 3.12 for Windows with the Python launcher, then open this file again.
pause
exit /b 1
:run
".detector-venv\Scripts\python.exe" detector_local.py
set "detector_status=%errorlevel%"
pause
exit /b %detector_status%
