@echo off
REM Launch the NQ Farm dashboard and open it in the browser.
cd /d C:\trading\nqorderflowbacktester
start "" http://localhost:8090
python live\farm\app.py
