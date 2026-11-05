"""Semantic corrector h(.) (paper Sec. IV-D, Eq. 5) built on IndicBART instead of a from-scratch transformer.

IndicBART formats: source "<text> </s> <2hi>", target "<2hi> <text> </s>", decoder start "<2hi>".
"""
from typing import List

import torch
from transformers import AlbertTokenizer, AutoModelForSeq2SeqLM

BASE_MODEL = "ai4bharat/IndicBART"
LANG_TAG = "<2hi>"


def load_tokenizer(path: str = BASE_MODEL) -> AlbertTokenizer:
    return AlbertTokenizer.from_pretrained(path, do_lower_case=False, keep_accents=True)


class Corrector:
    def __init__(self, path: str = BASE_MODEL, device=None, max_len: int = 160):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = load_tokenizer(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device).eval()
        self.max_len = max_len
        ids = self.tok._convert_token_to_id_with_added_voc
        self.bos, self.eos, self.pad, self.lang = ids("<s>"), ids("</s>"), self.tok.pad_token_id, ids(LANG_TAG)

    def encode_src(self, texts: List[str]) -> dict:
        # AlbertTokenizer would also return token_type_ids, which BART does not accept
        return self.tok([f"{t} </s> {LANG_TAG}" for t in texts], add_special_tokens=False, padding=True,
                        truncation=True, max_length=self.max_len, return_tensors="pt",
                        return_token_type_ids=False).to(self.device)

    def encode_tgt(self, texts: List[str]) -> torch.Tensor:
        return self.tok([f"{LANG_TAG} {t} </s>" for t in texts], add_special_tokens=False, padding=True,
                        truncation=True, max_length=self.max_len, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def correct(self, texts: List[str], num_beams: int = 4, max_len_ratio: float = 1.5, batch_size: int = 32) -> List[str]:
        """Correct a batch of transcripts. Falls back to the input when the output length (in words)
        differs from the input by more than a factor of max_len_ratio, a sign of hallucination."""
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.encode_src(chunk)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                gen = self.model.generate(
                    **enc, num_beams=num_beams, max_length=self.max_len, early_stopping=num_beams > 1, use_cache=True,
                    pad_token_id=self.pad, bos_token_id=self.bos, eos_token_id=self.eos,
                    decoder_start_token_id=self.lang,
                )
            for src, ids in zip(chunk, gen):
                hyp = self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                n_src, n_hyp = max(len(src.split()), 1), len(hyp.split())
                ok = src.strip() == "" or (n_hyp > 0 and 1 / max_len_ratio <= n_hyp / n_src <= max_len_ratio)
                out.append(hyp if ok else src)
        return out
