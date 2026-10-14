@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo Sanal ortam olusturuluyor...
  python -m venv .venv
  call .venv\Scripts\activate
  pip install -e .
) else (
  call .venv\Scripts\activate
)
start "" http://localhost:8000
python -m live_stream
pause
