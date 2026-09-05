@echo off
setlocal
cd /d "%~dp0.."
set "STOCK_APP_PORT=8766"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv\Scripts\python.exe was not found.
    echo Create the virtual environment and install requirements first.
    exit /b 1
)

echo.
echo Active IPv4 addresses on this computer:
ipconfig | findstr /i "IPv4"
echo.
echo Share one IPv4 address from the active Wi-Fi or Ethernet adapter:
echo   Home:  http://YOUR_IPV4:%STOCK_APP_PORT%/
echo   About: http://YOUR_IPV4:%STOCK_APP_PORT%/about.html
echo.
echo Keep this window open during the demonstration.
echo If Windows Firewall asks, allow access on Private networks.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py ^
    --server.address 0.0.0.0 ^
    --server.port %STOCK_APP_PORT% ^
    --server.headless true
