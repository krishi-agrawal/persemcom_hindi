"""ASR pretraining of the semantic codec (f_alpha, m^-1_delta) with the channel bypassed.

The resulting model is also the "Baseline w/o channel" upper bound.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.train.s2t import S2TTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asr.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    trainer = S2TTrainer(cfg["data"], cfg["model"], cfg["train"], stage="pretrain", num_workers=args.workers)
    trainer.fit(cfg["train"]["max_epochs"], max_steps=args.max_steps)


if __name__ == "__main__":
    main()
