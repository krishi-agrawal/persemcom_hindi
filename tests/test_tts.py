import torch

from persemcom.data.tts_text import GraphemeVocab
from persemcom.models.tacotron2_spk import (SpeakerTacotron2, attention_coverage, attention_focus_rate,
                                            guided_attention_loss)
from persemcom.train.tts import make_collate, tts_loss


def _tiny(r=2):
    return SpeakerTacotron2(n_symbol=20, n_mels=16, speaker_dim=8, frames_per_step=r, symbol_embedding_dim=32,
                            encoder_dim=32, decoder_rnn_dim=64, attention_rnn_dim=64, attention_hidden_dim=16,
                            prenet_dim=32, postnet_embedding_dim=32, decoder_max_step=30)


def _batch(r=2):
    items = []
    for L, T in ((7, 23), (5, 16)):
        items.append({"id": f"u{L}", "tokens": torch.randint(2, 20, (L,)), "mel": torch.randn(16, T),
                      "spk": torch.randn(8), "speaker_id": "s", "text": "x"})
    return make_collate(r)(items)


def test_collate_pads_to_r_and_builds_gate():
    b = _batch(r=2)
    assert b["mel"].shape == (2, 16, 24)       # 23 -> 24
    assert b["steps"].tolist() == [12, 8]
    assert b["gate"].shape == (2, 12)
    assert b["gate"][0].tolist() == [0.0] * 11 + [1.0]
    assert b["gate"][1, :7].sum() == 0 and b["gate"][1, 7:].min() == 1


def test_collate_is_picklable_for_windows_workers():
    import pickle

    c = pickle.loads(pickle.dumps(make_collate(2)))
    assert c(_batch_items())["mel"].shape[-1] % 2 == 0


def _batch_items():
    return [{"id": "a", "tokens": torch.tensor([3, 4, 5]), "mel": torch.randn(16, 9), "spk": torch.randn(8),
             "speaker_id": "s", "text": "x"}]


def test_forward_shapes_and_loss_backprop():
    m = _tiny()
    b = _batch()
    out = m(b["tokens"], b["token_lens"], b["mel"], b["mel_lens"], b["spk"])
    mel_before, mel_after, gate, align = out
    assert mel_before.shape == mel_after.shape == (2, 16, 24)
    assert gate.shape == (2, 12) and align.shape == (2, 12, 7)
    assert mel_after[1, :, 16:].abs().max() == 0  # padding masked
    loss, parts = tts_loss(out, b, ga_weight=1.0, gate_pos_weight=5.0)
    loss.backward()
    assert m.embedding.weight.grad is not None and torch.isfinite(loss)


def test_speaker_embedding_changes_output():
    torch.manual_seed(0)
    m = _tiny().eval()
    b = _batch()
    a = m(b["tokens"], b["token_lens"], b["mel"], b["mel_lens"], b["spk"])[1]
    c = m(b["tokens"], b["token_lens"], b["mel"], b["mel_lens"], b["spk"] + 1.0)[1]
    assert not torch.allclose(a, c)


def test_infer_lengths_are_multiples_of_r():
    m = _tiny().eval()
    b = _batch()
    mel, lens, align = m.infer(b["tokens"], b["token_lens"], b["spk"])
    assert mel.shape[0] == 2 and mel.shape[1] == 16 and mel.shape[2] % 2 == 0
    assert (lens % 2 == 0).all() and (lens <= 60).all()
    assert align.shape == (2, mel.shape[2] // 2, 7)  # (B, decoder steps, max tokens)
    assert attention_focus_rate(align, b["token_lens"], lens // 2) <= 1.0


def test_infer_accepts_unsorted_batch_and_keeps_order():
    torch.manual_seed(0)
    m = _tiny().eval()
    # Tacotron 2's prenet keeps dropout on at inference; make it deterministic for this comparison
    prenet = m.decoder.prenet

    def no_dropout(x):
        for layer in prenet.layers:
            x = torch.relu(layer(x))
        return x

    prenet.forward = no_dropout
    b = _batch()  # sorted: lengths 7, 5
    flip = torch.tensor([1, 0])
    mel_s, lens_s, _ = m.infer(b["tokens"], b["token_lens"], b["spk"])
    mel_u, lens_u, _ = m.infer(b["tokens"][flip], b["token_lens"][flip], b["spk"][flip])
    assert torch.equal(lens_u, lens_s[flip])
    assert torch.allclose(mel_u, mel_s[flip], atol=1e-5)


def test_guided_attention_prefers_diagonal():
    S, L = 10, 10
    diag = torch.eye(S)[None]
    anti = torch.eye(S).flip(1)[None]
    lens = torch.tensor([L])
    assert guided_attention_loss(diag, lens, torch.tensor([S])) < guided_attention_loss(anti, lens, torch.tensor([S]))
    assert abs(attention_focus_rate(diag, lens, torch.tensor([S])) - 1.0) < 1e-6


def test_guided_attention_penalizes_stuck_attention_strongly():
    # attention stuck on the first character for a long utterance must cost a lot, independent of text length
    S, L = 200, 80
    stuck = torch.zeros(1, S, L)
    stuck[..., 0] = 1.0
    loss = guided_attention_loss(stuck, torch.tensor([L]), torch.tensor([S]))
    assert loss > 0.5
    assert attention_coverage(stuck, torch.tensor([L]), torch.tensor([S])) == 1 / L
    assert attention_coverage(torch.eye(L)[None], torch.tensor([L]), torch.tensor([L])) == 1.0


def test_input_frame_dropout_only_changes_training_inputs():
    torch.manual_seed(0)
    m = SpeakerTacotron2(n_symbol=20, n_mels=16, speaker_dim=8, frames_per_step=2, symbol_embedding_dim=32,
                         encoder_dim=32, decoder_rnn_dim=64, attention_rnn_dim=64, attention_hidden_dim=16,
                         prenet_dim=32, postnet_embedding_dim=32, decoder_max_step=30, input_frame_dropout=0.5)
    seen = []
    orig = m.decoder.forward
    m.decoder.forward = lambda memory, mel_in, memory_lengths: (seen.append(mel_in.clone()) or orig(memory, mel_in, memory_lengths))
    b = _batch()
    m.train()
    m(b["tokens"], b["token_lens"], b["mel"], b["mel_lens"], b["spk"])
    m.eval()
    m(b["tokens"], b["token_lens"], b["mel"], b["mel_lens"], b["spk"])
    assert not torch.equal(seen[0], b["mel"])     # some input frame groups replaced in training
    assert torch.equal(seen[1], b["mel"])         # untouched in eval
    changed = (seen[0] != b["mel"]).any(1)        # (B, T) frames that changed
    assert torch.equal(changed[:, 0::2], changed[:, 1::2])  # whole r-frame groups at a time


def test_partial_load_across_reduction_factors():
    from persemcom.train.tts import load_partial

    src, dst = _tiny(r=2), _tiny(r=3)
    fresh = load_partial(dst, src.state_dict())
    assert fresh and all("prenet.layers.0" in k or "linear_projection" in k for k in fresh)
    assert torch.equal(dst.encoder.convolutions[0][0].weight, src.encoder.convolutions[0][0].weight)


def test_frame_dynamics_detects_drone():
    from persemcom.train.tts import frame_dynamics

    lens = torch.tensor([50])
    speech = torch.randn(1, 16, 50)
    drone = speech[:, :, :1].expand(-1, -1, 50) + 0.01 * torch.randn(1, 16, 50)
    assert frame_dynamics(drone, lens) < 0.1 * frame_dynamics(speech, lens)


def test_grapheme_vocab_roundtrip(tmp_path):
    v = GraphemeVocab.build(["नमस्ते दुनिया", "क़लम"])
    p = tmp_path / "v.json"
    v.save(str(p))
    w = GraphemeVocab.load(str(p))
    assert w.symbols == v.symbols
    ids = w.encode("नमस्ते")
    assert ids[-1] == 1 and 0 not in ids
    assert w.encode("नमस्ते X") == ids  # unknown Latin char and trailing space are dropped by normalization
