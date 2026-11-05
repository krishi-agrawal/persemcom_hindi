# Run Stage 1 then Stage 2 of Algorithm 1 for AWGN and Rayleigh, back to back, stopping on the first failure.
# Progress: runs\channel_stages.status ; per-job logs: runs\<channel>_<stage>.log / .err
param(
    [string]$Python = "C:\Users\Test\miniconda3\envs\persemcom\python.exe",
    [string[]]$Channels = @("awgn", "rayleigh"),
    [int]$Workers = 4
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force runs | Out-Null
$status = Join-Path $root "runs\channel_stages.status"

foreach ($ch in $Channels) {
    foreach ($st in @("stage1", "stage2")) {
        "RUNNING $ch $st since $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
        # Start-Process instead of '&' + redirection: in Windows PowerShell 5.1 redirected native stderr
        # (e.g. warnings) is turned into error records.
        $p = Start-Process -FilePath $Python -NoNewWindow -Wait -PassThru `
            -ArgumentList "-X utf8 scripts/05_train_channel.py --config configs/channel_$ch.yaml --stage $st --workers $Workers" `
            -RedirectStandardOutput "runs\${ch}_${st}.log" -RedirectStandardError "runs\${ch}_${st}.err"
        if ($p.ExitCode -ne 0) {
            "FAILED $ch $st (exit $($p.ExitCode)) at $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
            exit 1
        }
    }
}
"DONE at $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
