@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\start.ps1" %*
set "run_exit=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %run_exit%
