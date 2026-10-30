<#
Registers the forecast log's scheduled tasks in Windows Task Scheduler. NOT run automatically: you review it and run it yourself, once.

  powershell -ExecutionPolicy Bypass -File scripts\register_forecast_log_tasks.ps1 [-Remove]

Tasks (all run only while you are logged on, start as soon as possible after a missed start, need the network, give up after an hour):
  ForecastLog-Monday    every Monday 08:00   run_forecast_log.cmd monday    next week (NW) and this week (TW)
  ForecastLog-Evening   every day 18:00      run_forecast_log.cmd evening   tomorrow (TD)
  ForecastLog-Score     every day 21:00      run_forecast_log.cmd score     add actuals and rewrite the scores
A start missed because the computer was off runs when it is next on and you are logged on: once, however many starts were missed. A run issued after its deadline is logged as late.
#>
param([switch]$Remove)

$runner = Split-Path -Parent $PSScriptRoot
$cmd = Join-Path $runner "scripts\run_forecast_log.cmd"
$names = "ForecastLog-Monday", "ForecastLog-Evening", "ForecastLog-Score"

if ($Remove) {
    foreach ($n in $names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    "Removed: $($names -join ', ')"
    return
}

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

$jobs = @(
    @{ Name = "ForecastLog-Monday";  Arg = "monday";  Trigger = (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 08:00) },
    @{ Name = "ForecastLog-Evening"; Arg = "evening"; Trigger = (New-ScheduledTaskTrigger -Daily -At 18:00) },
    @{ Name = "ForecastLog-Score";   Arg = "score";   Trigger = (New-ScheduledTaskTrigger -Daily -At 21:00) }
)
foreach ($j in $jobs) {
    $action = New-ScheduledTaskAction -Execute $cmd -Argument $j.Arg -WorkingDirectory $runner
    Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $j.Trigger -Settings $settings -Principal $principal -Force | Out-Null
    "Registered $($j.Name): $cmd $($j.Arg)"
}
