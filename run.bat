@echo off
setlocal
cd /d "%~dp0"

set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

cargo build --release --target wasm32-unknown-unknown || exit /b 1
wasm-bindgen --target web --no-typescript --out-dir pkg target\wasm32-unknown-unknown\release\worldrenderer.wasm || exit /b 1
cargo fmt || exit /b 1

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":8080 .*LISTENING"') do taskkill /f /pid %%p >nul 2>&1

echo.
echo   http://localhost:8080/
echo.

python -m http.server 8080 --bind 127.0.0.1
