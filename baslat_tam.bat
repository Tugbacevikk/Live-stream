@echo off
chcp 65001 > nul
rem ================================================================
rem  TAM MOD (sunucu, 4000 mesaj): API + dil tespiti + Türkçe + Vietnamca worker'ları + bildirim + bakım.
rem  Canlı altyazı bu modda yok; o, yerel modda (baslat.bat) çalışır.
rem  PostgreSQL ve Redis çalışıyor olmalı (DATABASE_URL, REDIS_URL: .env).
rem  Docker Desktop varsa ikisini şöyle açabilirsiniz:
rem    docker run -d --name asr-pg -e POSTGRES_USER=asr -e POSTGRES_PASSWORD=asr -e POSTGRES_DB=asr -p 5432:5432 postgres:16-alpine
rem    docker run -d --name asr-redis -p 6379:6379 redis:7-alpine redis-server --appendonly yes
rem ================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
if not exist ".env" copy ".env.example" ".env" > nul

start "Ses Analizi - API"        cmd /k "%PY% -m uvicorn live_stream.interface.api:app --host 127.0.0.1 --port 8000"
timeout /t 5 /nobreak > nul
start "Ses Analizi - Dil"        cmd /k "%PY% -m live_stream.interface.stream_worker --kuyruk lid"
start "Ses Analizi - Vietnamca"  cmd /k "%PY% -m live_stream.interface.stream_worker --kuyruk vi"
start "Ses Analizi - Turkce"     cmd /k "%PY% -m live_stream.interface.stream_worker --kuyruk tr --surec 1"
start "Ses Analizi - Bildirim"   cmd /k "%PY% -m live_stream.interface.dispatcher"
start "Ses Analizi - Bakim"      cmd /k "%PY% -m live_stream.interface.bakim"

echo.
echo   Tum servisler ayri pencerelerde baslatildi: http://127.0.0.1:8000/docs
echo   Durdurmak icin: durdur.bat
