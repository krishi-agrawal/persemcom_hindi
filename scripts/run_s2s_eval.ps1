# After make_corrected_train.py: fine-tune the synthesizer on corrected transcripts (Algorithm 1, l.12-15),
# then evaluate PerSemCom end to end on both channels and the Txt2Vid speech-to-speech baseline.
# Jobs run one at a time and the chain stops on the first failure.
# Progress: runs\s2s_eval.status ; per-job logs: runs\<job>.log / .err
# Resume after a failure with -From <job> (earlier jobs are skipped); -ReuseS2T makes the PerSemCom jobs
# load their saved speech-to-text transcripts instead of repeating that sweep.
param(
    [string]$Python = "C:\Users\Test\miniconda3\envs\persemcom\python.exe",
    [string]$From = "",
    [switch]$ReuseS2T
)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$status = Join-Path $root "runs\s2s_eval.status"
$tts = "checkpoints/tts_finetune/best.pt"
$reuse = if ($ReuseS2T) { " --reuse-s2t" } else { "" }

$jobs = @(
    @("finetune_tts", "-X utf8 scripts/08_train_tts.py --finetune data/corrector/train_corrected.jsonl"),
    @("eval_persemcom_awgn", "-X utf8 scripts/10_eval_all.py --channel awgn --tts-ckpt $tts$reuse"),
    @("eval_persemcom_rayleigh", "-X utf8 scripts/10_eval_all.py --channel rayleigh --tts-ckpt $tts$reuse"),
    @("eval_txt2vid_awgn", "-X utf8 scripts/09_baselines.py --channel awgn --systems txt2vid --tts-ckpt $tts"),
    @("eval_txt2vid_rayleigh", "-X utf8 scripts/09_baselines.py --channel rayleigh --systems txt2vid --tts-ckpt $tts")
)

# wait for the corrected transcripts
$log = Join-Path $root "runs\make_corrected_train.log"
while (-not ((Test-Path $log) -and ((Get-Content $log -Raw) -match "wrote "))) {
    "WAITING for make_corrected_train since $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
    Start-Sleep -Seconds 30
}

$started = ($From -eq "")
foreach ($j in $jobs) {
    $name, $args_ = $j
    if (-not $started) {
        if ($name -ne $From) { continue }
        $started = $true
    }
    "RUNNING $name since $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
    $p = Start-Process -FilePath $Python -NoNewWindow -Wait -PassThru -ArgumentList $args_ `
        -RedirectStandardOutput "runs\$name.log" -RedirectStandardError "runs\$name.err"
    if ($p.ExitCode -ne 0) {
        "FAILED $name (exit $($p.ExitCode)) at $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
        exit 1
    }
}
"DONE at $(Get-Date -Format s)" | Out-File -Encoding utf8 $status
