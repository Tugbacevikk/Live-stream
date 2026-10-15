@echo off
chcp 65001 > nul
echo Ses Analizi servisleri durduruluyor...
taskkill /F /FI "WINDOWTITLE eq Ses Analizi*" /T > nul 2>&1
echo Tum Ses Analizi servisleri kapatildi.
echo (Yarim kalan isler veritabaninda kalir; yeniden baslatinca kaldigi yerden devam eder.)
pause
