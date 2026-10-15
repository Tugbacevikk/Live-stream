@echo off
chcp 65001 > nul
rem ================================================================
rem  YEREL MOD: tek pencere, Redis / PostgreSQL / Docker GEREKMEZ.
rem  Veritabanı: data\uploads\jobs.db (SQLite). Bilgisayarın kendisi için en basit yol.
rem  Tam mod (PostgreSQL + Redis + ayrı worker'lar) için: baslat_tam.bat
rem ================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)

if not exist ".env" (
    copy ".env.example" ".env" > nul
    echo .env olusturuldu ^(ayarlari buradan degistirebilirsiniz^)
)

echo [1/2] Modeller kontrol ediliyor ^(eksik varsa iner; ilk sefer ~2 GB^)...
%PY% scripts\download_models.py || goto :hata

echo [2/2] Ses Analizi ^(yerel mod^) baslatiliyor...
start "Ses Analizi - Yerel" cmd /k "%PY% -m live_stream --host 127.0.0.1 --port 8000"
echo.
echo   Hazir oldugunda:
echo     Dosya analizi : http://127.0.0.1:8000
echo     Canli altyazi : http://127.0.0.1:8000/canli
echo   Durdurmak icin: durdur.bat
goto :eof

:hata
echo Model indirme basarisiz. Internet baglantisini kontrol edin.
pause
