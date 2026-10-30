<#
Registers the forecast log's scheduled tasks in Windows Task Scheduler. You review it and run it yourself, once, FROM THE RUNNER FOLDER (the tasks run the code next to this script):

  powershell -ExecutionPolicy Bypass -File D:\Projectos\DRG\forecasting-runner\scripts\register_forecast_log_tasks.ps1 [-Remove]

Tasks (they run only while you are logged on, start as soon as possible after a missed start, need the network, and give up after an hour):
  ForecastLog-Monday    every Monday 08:00   run_forecast_log.cmd monday    next week (NW) and this week (TW)
  ForecastLog-Evening   every day 18:00      run_forecast_log.cmd evening   tomorrow (TD)
  ForecastLog-Score     every day 21:00      run_forecast_log.cmd score     add actuals and rewrite the scores
Each task's first start is its NEXT occurrence, so registering never triggers a run by itself. A start missed because the computer was off runs when it is next on and you are logged on:
once, however many starts were missed. A run issued after its deadline is logged as late. -Remove deletes the three tasks and nothing else.
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
if (-not (Test-Path (Join-Path $runner ".git")) -or -not (Test-Path $cmd)) { throw "Run this script from the runner folder (no scripts\run_forecast_log.cmd or .git next to it): $runner" }

function Next-Occurrence([int]$hour, [int]$minute, [string]$weekday) {
    $t = (Get-Date).Date.AddHours($hour).AddMinutes($minute)
    while ($t -le (Get-Date) -or ($weekday -and $t.DayOfWeek.ToString() -ne $weekday)) { $t = $t.AddDays(1) }
    return $t
}

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

$jobs = @(
    @{ Name = "ForecastLog-Monday";  Arg = "monday";  Start = (Next-Occurrence 8 0 "Monday"); Trigger = { param($s) New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At $s } },
    @{ Name = "ForecastLog-Evening"; Arg = "evening"; Start = (Next-Occurrence 18 0 "");      Trigger = { param($s) New-ScheduledTaskTrigger -Daily -At $s } },
    @{ Name = "ForecastLog-Score";   Arg = "score";   Start = (Next-Occurrence 21 0 "");      Trigger = { param($s) New-ScheduledTaskTrigger -Daily -At $s } }
)
foreach ($j in $jobs) {
    $trigger = & $j.Trigger $j.Start
    $action = New-ScheduledTaskAction -Execute $cmd -Argument $j.Arg -WorkingDirectory $runner
    Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    "Registered $($j.Name): $cmd $($j.Arg) - first start $($j.Start.ToString('yyyy-MM-dd HH:mm'))"
}
