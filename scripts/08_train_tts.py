"""Train the speaker-conditioned Tacotron 2, or fine-tune it on corrected transcripts (Algorithm 1, l.12-15).

    python scripts/08_train_tts.py
    python scripts/08_train_tts.py --finetune data/corrector/train_corrected.jsonl
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persemcom.train.tts import TTSTrainer, load_text_override  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tts.yaml")
    ap.add_argument("--finetune", default=None, help="JSONL of {id, text} corrected transcripts for train utts")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--init-from", default=None, help="warm-start weights (shape-mismatched tensors are re-initialized)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.finetune:
        ft = cfg["finetune"]
        base = Path(cfg["train"]["ckpt_dir"]) / "best.pt"
        # guided attention stays on: alignment on this corpus only held with it (runs 1-3)
        cfg["train"].update(lr=ft["lr"], steps=ft["steps"], ckpt_dir=ft["ckpt_dir"], warmup_steps=1)
        trainer = TTSTrainer(cfg, text_override=load_text_override(args.finetune), init_from=str(base))
    else:
        trainer = TTSTrainer(cfg, init_from=args.init_from)
    trainer.fit(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
