from xpu_llm_bench.client import parse_prompt_logprobs
from xpu_llm_bench.quality import (
    CODE_TASKS,
    cut_at_eos,
    extract_code,
    first_divergence,
    format_check,
    logprob_probes,
    mean_answer_logprob,
    needle_pass,
    repetition_ratio,
    run_code_test,
)


def test_eos_cut_and_format() -> None:
    text, eos = cut_at_eos("Fine answer.<|im_end|>garbage after eos")
    assert text == "Fine answer." and eos
    ok = format_check("The meeting moved to Tuesday. Notes follow below.<|im_end|>xx", "long_form")
    assert ok.ok and ok.eos_seen
    loop = format_check("the cat sat on the mat " * 40, "long_form")
    assert not loop.ok and any("repetition" in p for p in loop.problems)
    leak = format_check("hello <|im_start|>user", "multi_turn_chat")
    assert not leak.ok
    bleed = format_check("Sure.\nUser: and another thing", "multi_turn_chat")
    assert not bleed.ok
    assert repetition_ratio("a b c d e f g h i j k l m n o p", 6) == 0.0


def test_needle() -> None:
    assert needle_pass("teal-4821<|im_end|>", "teal-4821")
    assert not needle_pass("I don't know", "teal-4821")


def test_code_edit_unit_test_pass_and_fail() -> None:
    task = CODE_TASKS[0]
    good = "```python\ndef median(xs):\n    s = sorted(xs)\n    n = len(s)\n    m = n // 2\n    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2\n```"
    ok, detail = run_code_test(extract_code(good), task["test"])
    assert ok, detail
    bad = "```python\ndef median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]\n```"
    ok, _ = run_code_test(extract_code(bad), task["test"])
    assert not ok
    spin = "```python\ndef median(xs):\n    while True:\n        pass\n```"
    ok, detail = run_code_test(extract_code(spin), task["test"], timeout_s=2)
    assert not ok


def test_first_divergence() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_prompt_logprob_parse_and_mean() -> None:
    data = {"choices": [{"prompt_logprobs": [None, {"11": {"logprob": -1.0}}, {"12": {"logprob": -0.5}}, {"13": {"logprob": -0.25}}]}]}
    lps = parse_prompt_logprobs(data, [10, 11, 12, 13])
    assert lps == [None, -1.0, -0.5, -0.25]
    assert mean_answer_logprob(lps, 2) == -0.375
    echo = {"choices": [{"logprobs": {"token_logprobs": [None, -2.0, -1.0]}}]}
    assert parse_prompt_logprobs(echo, [1, 2, 3]) == [None, -2.0, -1.0]


def test_logprob_probes_are_fixed() -> None:
    a = logprob_probes(20261111, 2000)
    b = logprob_probes(20261111, 2000)
    assert [p.context for p in a] == [p.context for p in b]
    assert len({p.probe_id for p in a}) == len(a)
