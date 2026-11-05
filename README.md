# PerSemCom — Hindi

Reimplementation of *PerSemCom: A Personalized Semantic Communication Framework for Speech
Transmission* (Tan et al., IEEE TCOM 2026) on the IndicVoices-R Hindi corpus.

## Setup (Windows)

```powershell
conda create -n persemcom python=3.10
# This machine's C:\ProgramData\pip\pip.ini adds a dead NVIDIA index; bypass all pip config files.
# Must be lowercase 'nul' (pip compares against os.devnull exactly).
$env:PIP_CONFIG_FILE = 'nul'
C:\Users\Test\miniconda3\envs\persemcom\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
C:\Users\Test\miniconda3\envs\persemcom\python.exe -m pip install -e ".[dev]"
C:\Users\Test\miniconda3\envs\persemcom\python.exe -m pytest
```

## Data

IndicVoices-R Hindi: 72.8 h available, 72.2 h kept (clips 1–25 s, speakers with ≥ 20 clips).
Speaker-shared split, as in the paper: 376 speakers appear in every split.

| Split | Clips | Hours |
|---|---|---|
| train | 23,889 | 65.0 |
| val | 1,326 | 3.6 |
| test | 1,326 | 3.6 |

## Pipeline

```powershell
python scripts/01_download.py                     # 44 GiB of parquet shards (48 kHz stereo)
python scripts/02_prepare.py                      # mono 24 kHz FLAC, speaker-shared split, tokenizer
python scripts/02_prepare.py --tokenizer-only --tokenizer artifacts/spm_hi_1000 --vocab-size 1000
python scripts/03_features.py                     # 80-d fbank @16 kHz + Vocos 100-d mel @24 kHz, CMVN
python scripts/04_train_asr.py                    # pretrain = "Baseline w/o channel"
powershell -File scripts/run_channel_stages.ps1   # Stage 1 + Stage 2 for AWGN, then Rayleigh
powershell -File scripts/run_after_channels.ps1   # Rayleigh eval, corrector pairs + IndicBART, GE2E + KB, Tacotron 2
python scripts/eval_s2t.py --ckpt <ckpt> --channel none|awgn|rayleigh --snrs ...   # WER/CER vs SNR -> results/
powershell -File scripts/run_s2s_eval.ps1         # TTS fine-tune on corrected text, full PerSemCom + Txt2Vid eval
python scripts/09_baselines.py --channel awgn --systems amrwb   # AMR-WB + LDPC (also --channel rayleigh)
python scripts/11_complexity.py                   # params + latency per module (Table III)
python scripts/12_speaker_tsne.py                 # speaker t-SNE (Fig. 15)
python scripts/13_mos_set.py                      # listening-test clips -> results/mos/ ; page: mos/listening_test.html
```

## Results (test split)

Full write-up with charts: the "PerSemCom in Hindi — Results" doc (https://claude.ai/artifact/6rLe1y5dE4xFTgucbGnvne).

| | AWGN -10 / -2 / 2 / 20 dB | Rayleigh -10 / -2 / 2 / 20 dB |
|---|---|---|
| S2T WER, PerSemCom + corrector | 50.0 / 21.9 / 20.3 / 19.3% | 75.9 / 36.8 / 27.3 / 19.4% |
| S2T WER, Txt2Vid-style | lost / lost / lost / 20.4% | lost / lost / lost / 20.4% |
| WER of delivered speech, PerSemCom | 64.2 / 46.2 / 45.7 / 44.8% | 85.9 / 56.5 / 50.0 / 46.0% |
| WER of delivered speech, AMR-WB + LDPC | lost / lost / lost / 21.3% | lost / lost / lost / 21.3% |

- No-channel ASR: 20.4% WER, 8.5% CER. Synthesizer from the true text: 41-42% WER of speech (the PerSemCom floor).
- Mismatch at 0 dB: Rayleigh-trained on AWGN 23.8% (matched 22.0%); AWGN-trained on Rayleigh 79.1% (matched 34.2%).
- Speaker cosine to the original: PerSemCom 0.53-0.59, averaged KB embedding 0.015; t-SNE nearest-speaker match 96% vs 8%.
- Symbols per second of speech: Txt2Vid 131, PerSemCom 795, AMR-WB + LDPC 17,940.
- Size / latency (A2000, 6.7 s clip): transmitter 34 MB / 88 ms; receiver 1.15 GB / 0.77 s (IndicBART 976 MB / 325 ms, Tacotron 2 123 MB / 423 ms).

CSV sources: `results/s2t_*.csv`, `s2s_*.csv`, `baseline_*.csv`, `complexity.csv`, `tsne_*.csv`.

## Deviations from the paper (and why)

| What | Paper | Here | Reason |
|---|---|---|---|
| CTC vocabulary | 5000 subwords | 1000 | With 65 h, 68% of 5000 pieces occur < 50 times; training collapsed to all-blank output (WER 97% at epoch 9). `runs/asr_failed_vocab5000.log` |
| Variational noise | used, std not given | std 0.01 | Common recipe value 0.075 exceeds the weight std (0.066) and collapsed training at step 10k; measured WER 37.5% → 38.6% at 0.01 vs 100% at 0.075 |
| Rayleigh equalizer | perfect CSI, unspecified | MMSE | Zero forcing's deep-fade noise blow-ups capped the Stage-1 codec at MSE 0.40 vs 0.10 for MMSE |
| Semantic corrector | 6-layer transformer from scratch | fine-tuned IndicBART (244M) | Pretrained Hindi knowledge; user decision |
| Corrector training pairs | ASR errors + synthetic | Stage-2 decodes of the train split with input time masking (7 × ≤ 4%) | The ASR model makes ~37% fewer errors on train audio; masking matches test-level WER |
| Tacotron 2 | r = 1 | r = 3, input frame dropout 0.3, guided attention, 800 s batches | Decoder loop is launch-bound on the A2000 (~6 days → ~1 day); without frame dropout the free-running decoder droned |
| Speaker encoder input | 40-d mel | the cached 80-d ASR fbank | Same features for original and generated speech |
| Similarity score (Eq. 8) | BERT | LaBSE | Mean-pooled MuRIL scored an unrelated sentence 0.993 vs 1.000 for identical; LaBSE: 0.466 vs 1.000 |

Stage 1 alone gets only ~74% WER (AWGN, 10 dB): the trained encoder's features need ~96 principal components
for 88% of their variance, but the paper's codec sends 64 real numbers per frame. Stage 2 (joint) fixes it:
20.9% after 5 epochs.
