"""Train the GE2E speaker encoder, then build the speaker knowledge base (paper Fig. 6).

    python scripts/07_train_speaker.py               # train + build KB
    python scripts/07_train_speaker.py --kb-only     # rebuild KB from checkpoints/speaker/best.pt
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.kb.build_kb import build_kb  # noqa: E402
from persemcom.train.ge2e import train_ge2e  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spk.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--kb-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if not args.kb_only:
        best = train_ge2e(cfg, max_steps=args.max_steps)
        print(f"best val EER {best:.4f}")
    build_kb(cfg, str(Path(cfg["train"]["ckpt_dir"]) / "best.pt"))


if __name__ == "__main__":
    main()
