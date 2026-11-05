# After run_channel_stages.ps1 reports DONE: evaluate the Rayleigh model, build Rayleigh corrector pairs,
# fine-tune the corrector, train the speaker encoder + knowledge base, then train Tacotron 2.
# Jobs run one at a time (each needs the GPU to itself) and the chain stops on the first failure.
# Progress: runs\after_channels.status ; per-job logs: runs\<job>.log / .err
param([string]$Python = "C:\Users\Test\miniconda3\envs\persemcom\python.exe")
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$status = Join-Path $root "runs\after_channels.status"
$snrs = "-10 -6 -2 0 2 4 6 8 10 12 14 16 18 20"

$jobs = @(
    @("eval_rayleigh_matched", "-X utf8 scripts/eval_s2t.py --ckpt checkpoints/rayleigh/stage2/best.pt --channel rayleigh --snrs $snrs --out results/s2t_trainRayleigh_testRayleigh.csv"),
    @("eval_rayleigh_mismatch", "-X utf8 scripts/eval_s2t.py --ckpt checkpoints/rayleigh/stage2/best.pt --channel awgn --snrs $snrs --out results/s2t_trainRayleigh_testAWGN.csv"),
    @("pairs_rayleigh", "-X utf8 -m persemcom.corrector.make_pairs --ckpt checkpoints/rayleigh/stage2/best.pt --channel rayleigh --passes 3 --out data/corrector/pairs_rayleigh.jsonl"),
    @("train_corrector", "-X utf8 -m persemcom.corrector.train_indicbart --pairs data/corrector/pairs_awgn.jsonl data/corrector/pairs_rayleigh.jsonl --out checkpoints/corrector"),
    @("train_speaker", "-X utf8 scripts/07_train_speaker.py"),
    @("train_tts", "-X utf8 scripts/08_train_tts.py")
)

# wait for the channel chain
$chain = Join-Path $root "runs\channel_stages.status"
while ($true) {
    $s = if (Test-Path $chain) { Get-Content $chain -Raw } else { "" }
    if ($s -match "DONE") { break }
    if ($s -match "FAILED") { "NOT STARTED: channel chain failed ($s)" | Out-File -Encoding utf8 $status; exit 1 }
    "WAITING for channel chain since $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
    Start-Sleep -Seconds 60
}

foreach ($j in $jobs) {
    $name, $args_ = $j
    "RUNNING $name since $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
    $p = Start-Process -FilePath $Python -NoNewWindow -Wait -PassThru -ArgumentList $args_ `
        -RedirectStandardOutput "runs\$name.log" -RedirectStandardError "runs\$name.err"
    if ($p.ExitCode -ne 0) {
        "FAILED $name (exit $($p.ExitCode)) at $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
        exit 1
    }
}
"DONE at $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
