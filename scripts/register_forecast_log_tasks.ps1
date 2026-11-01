<#
Registers the forecast log's scheduled tasks in Windows Task Scheduler. You review it and run it yourself, once, FROM THE RUNNER FOLDER (the tasks run the code next to this script):

  powershell -ExecutionPolicy Bypass -File D:\Projectos\DRG\forecasting-runner\scripts\register_forecast_log_tasks.ps1 [-Remove]

Task (it runs only while you are logged on, starts as soon as possible after a missed start, needs the network, and gives up after an hour):
  ForecastLog-Weekly    every Monday 12:30   run_forecast_log.cmd weekly    process any new export in data\raw_private, update the scores, then make the forecasts:
                                                                            next week (NW) and this week (TW), Monday to Sunday each
Its first start is its NEXT occurrence, so registering never triggers a run by itself. A start missed because the computer was off runs when it is next on and you are logged on:
once, however many starts were missed. A run issued after its deadline is logged as late. -Remove deletes this task (and the four older ForecastLog tasks, if any are registered) and nothing else.
#>
param([switch]$Remove)

$runner = Split-Path -Parent $PSScriptRoot
$cmd = Join-Path $runner "scripts\run_forecast_log.cmd"
$names = "ForecastLog-Weekly"
$old = "ForecastLog-Drop", "ForecastLog-Monday", "ForecastLog-Evening", "ForecastLog-Score"  # an earlier design: removed if present

if ($Remove) {
    foreach ($n in $names + $old) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
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

foreach ($n in $old) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
$jobs = @(
    @{ Name = "ForecastLog-Weekly";  Arg = "weekly";  Start = (Next-Occurrence 12 30 "Monday"); Trigger = { param($s) New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At $s } }
)
foreach ($j in $jobs) {
    $trigger = & $j.Trigger $j.Start
    $action = New-ScheduledTaskAction -Execute $cmd -Argument $j.Arg -WorkingDirectory $runner
    Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    "Registered $($j.Name): $cmd $($j.Arg) - first start $($j.Start.ToString('yyyy-MM-dd HH:mm'))"
}
