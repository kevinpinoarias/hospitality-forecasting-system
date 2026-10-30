@echo off
rem Runs one step of the forecast log from the runner folder and keeps a console log.
rem Usage: run_forecast_log.cmd monday|evening|score [--dry-run]
rem Task Scheduler starts this; it can also be double-clicked or run in a terminal.
setlocal
cd /d "%~dp0.."
set PYTHONPATH=%CD%
set WANDB_MODE=disabled
if not exist logs mkdir logs
rem the date as yyyymmdd for the log file name (this computer prints dd/mm/yyyy)
set STAMP=%DATE:~-4%%DATE:~3,2%%DATE:~0,2%
echo [%DATE% %TIME%] start %1 %2 >> logs\forecast_log_%STAMP%.log
if /I "%1"=="score" (
    python -m src.forecast_log.score_forecasts %2 >> logs\forecast_log_%STAMP%.log 2>&1
) else (
    python -m src.forecast_log.run_forecasts --trigger %1 %2 >> logs\forecast_log_%STAMP%.log 2>&1
)
set CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] end %1 exit %CODE% >> logs\forecast_log_%STAMP%.log
exit /b %CODE%
