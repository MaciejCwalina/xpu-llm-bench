from xpu_llm_bench.prompts import corpus


def test_seeded_and_typed() -> None:
    a = corpus("code_edits", seed=20261111).user_message(128)
    b = corpus("code_edits", seed=20261111).user_message(128)
    c = corpus("code_edits", seed=99).user_message(128)
    d = corpus("long_form", seed=20261111).user_message(128)
    assert a == b
    assert a != c
    assert a != d
    assert "def " in a
    assert "Paragraph" in d or "journal" in d.lower() or "instrument" in d.lower()
    chat = corpus("multi_turn_chat", seed=1).user_message(64)
    assert "Turn" in chat or "User:" in chat


def test_grows_to_target() -> None:
    corp = corpus("long_form", seed=1, chars_per_token=4.0)
    text = corp.fill(2000)
    assert len(text) >= 2000 * 4 - 200
