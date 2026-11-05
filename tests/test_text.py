from persemcom.data.text_norm import is_clean_hindi, normalize
from persemcom.metrics.wer import cer, wer


def test_normalize_strips_punctuation_and_spaces():
    assert normalize("  नमस्ते,   दुनिया।  कैसे हो?॥ ") == "नमस्ते दुनिया कैसे हो"


def test_normalize_unifies_nukta_forms():
    precomposed = "क़"          # क़
    decomposed = "क़"     # क + nukta
    assert normalize(precomposed) == normalize(decomposed)


def test_normalize_drops_zero_width():
    assert normalize("क्‍ष") == "क्ष"


def test_is_clean_hindi():
    assert is_clean_hindi("मैं घर जा रहा हूँ।")
    assert not is_clean_hindi("मेरा phone खो गया")
    assert not is_clean_hindi("मेरे पास 5 रुपये हैं")
    assert not is_clean_hindi("मेरे पास ५ रुपये हैं")
    assert not is_clean_hindi("।।")


def test_wer_cer():
    refs = ["मैं घर जा रहा हूँ"]
    assert wer(refs, ["मैं घर जा रहा हूँ।"]) == 0.0
    assert wer(refs, ["मैं घर जा रही हूँ"]) == 0.2
    assert 0 < cer(refs, ["मैं घर जा रही हूँ"]) < 0.2
