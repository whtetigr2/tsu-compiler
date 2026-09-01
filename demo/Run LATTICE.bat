@echo off
title LATTICE - stochastic world generator
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
"C:\Users\whtet\AppData\Local\Python\pythoncore-3.14-64\python.exe" "demo\lattice_app.py"
if errorlevel 1 (
  echo.
  echo LATTICE exited with an error. The message above is the real one.
  pause
)
