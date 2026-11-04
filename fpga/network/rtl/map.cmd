@echo off
call "E:\efinity-2026.1.132-windows-x64\efinity\2026.1\bin\setup.bat"
if errorlevel 1 exit /b 1
set "EFINITY_COLOR_PRINTING=0"
call efx_run.bat cnn_production_frame.xml --flow map --timeout 300
exit /b %errorlevel%
