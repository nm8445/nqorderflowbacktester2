@echo off
REM NQ Order Flow - Live Signal Generator Launcher
REM Analyzes NQ order flow, sends MNQ execution signals to NT8

echo ================================================================================
echo NQ ORDER FLOW LIVE SIGNAL GENERATOR
echo ================================================================================
echo.
echo Data Source: NQ.c.0 (continuous front month order flow)
echo Execution: MNQ (Micro NQ contracts)
echo Trading Hours: 9:30 AM - 11:00 AM ET
echo.
echo Make sure NinjaTrader 8 add-on is running first!
echo.
echo ================================================================================
echo.

REM Check if NT8 endpoint is reachable
echo Checking NT8 connection...
curl -s -o nul http://localhost:8080/
if errorlevel 1 (
    echo [WARNING] Cannot connect to NT8 on localhost:8080
    echo Make sure NinjaTrader add-on is running and server is started!
    echo.
    pause
)

echo Starting signal generator...
echo.

cd /d C:\trading\nqorderflowbacktester
python src/nqbt/live/signal_generator.py

echo.
echo Signal generator stopped.
pause
