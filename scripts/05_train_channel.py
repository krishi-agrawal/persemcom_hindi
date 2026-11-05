"""Stage 1 (channel codec only, MSE) and Stage 2 (joint, CTC) of Algorithm 1 for one channel type.

    python scripts/05_train_channel.py --config configs/channel_awgn.yaml --stage stage1
    python scripts/05_train_channel.py --config configs/channel_awgn.yaml --stage stage2
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.train.s2t import S2TTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-config", default="configs/asr.yaml")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    base = yaml.safe_load(open(args.asr_config, encoding="utf-8"))
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    stage_cfg = cfg[args.stage]
    model_cfg = {**base["model"], **cfg["model"]}
    train_cfg = {
        "optimizer": base["train"]["optimizer"],
        "lr": stage_cfg["lr"],  # constant LR for the channel stages
        "amp": cfg.get("amp", True),
        "grad_clip": base["train"].get("grad_clip", 5.0),
        "spec_augment": base["train"].get("spec_augment"),  # ignored in stage1 (frozen encoder)
        "variational_noise": {"std": 0.0, "start_step": 0},
        "ckpt_dir": stage_cfg["ckpt_dir"],
    }
    # stage1 starts from the pretrained ASR; stage2 from the best stage1 model
    init = cfg["init_from"] if args.stage == "stage1" else str(Path(cfg["stage1"]["ckpt_dir"]) / "best.pt")
    snr = (cfg["snr_db"]["low"], cfg["snr_db"]["high"])
    trainer = S2TTrainer(base["data"], model_cfg, train_cfg, stage=args.stage, snr_range=snr,
                         select_snr=cfg["stage2"]["select_snr_db"], init_from=init, num_workers=args.workers)
    trainer.fit(stage_cfg["max_epochs"], max_steps=args.max_steps)


if __name__ == "__main__":
    main()
