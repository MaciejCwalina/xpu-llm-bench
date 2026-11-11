from xpu_llm_bench.bootlog import classify_failure, lines_since, parse_boot

PROD = """2026-11-09T17:22:51.000000000Z (EngineCore pid=162) INFO 11-09 17:22:51 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='/model', speculative_config=SpeculativeConfig(method='mtp', model='/model', num_spec_tokens=6), tokenizer='/model', dtype=torch.float16, max_seq_len=262144, download_dir=None, kv_cache_dtype=int8_k_int4_v, compilation_config={'cudagraph_mode': <CUDAGraphMode.FULL_DECODE_ONLY: 2>, 'cudagraph_capture_sizes': [7, 14, 21, 28]}
2026-11-09T17:23:38.000000000Z (Worker_TP0 pid=267) INFO 11-09 17:23:38 [xpu.py:152] Using Xe2 K8/V4 attention backend.
2026-11-09T17:24:00.000000000Z (EngineCore pid=162) INFO 11-09 17:24:00 [kv_cache_utils.py:2395] XPU KV cache size: 753,004 tokens, Maximum concurrency for 262,144 tokens per request: 2.87x
2026-11-09T17:24:10.000000000Z (Worker_TP0 pid=267) Capturing CUDA graphs (FULL):   0%|          | 0/4 [00:00<?, ?it/s]Capturing CUDA graphs (FULL): 100%|##########| 4/4 [00:01<00:00,  3.43it/s]
2026-11-09T17:24:12.000000000Z (Worker_TP0 pid=267) INFO 11-09 17:24:12 [model_runner.py:1066] Graph capturing finished in 2 secs, took 0.05 GiB
"""

QSA = """(Worker_TP0 pid=281) INFO 11-11 10:54:49 [xpu.py:152] Using Xe2 K8/V4 attention backend.
(EngineCore pid=173) INFO 11-11 10:55:34 [kv_cache_utils.py:2706] QSA_MEMORY_PLAN {'blocks': 412, 'main_bytes': 11715618816}
(Worker_TP0 pid=281) QSA_GRAPH skip starts=(0, 7, 14) n=14 pos=True
(Worker_TP0 pid=281) QSA_WRAP Qwen3NextAttention
(Worker_TP0 pid=281) INFO 11-11 10:57:07 [model_runner.py:1067] Graph capturing finished in 5 secs, took 0.13 GiB
"""


def test_production_boot_is_dense_with_graphs() -> None:
    ev = parse_boot(PROD)
    assert ev.attention_boot == "dense"
    assert ev.vllm_version == "0.30.0"
    assert ev.spec_method == "mtp" and ev.spec_k == 6
    assert ev.max_model_len == 262144
    assert ev.kv_dtype == "int8_k_int4_v"
    assert ev.kv_tokens == 753004
    assert ev.capture_sizes_logged == [7, 14, 21, 28]
    assert ev.cudagraph_mode == "FULL_DECODE_ONLY"
    assert ev.capture_counts == {"FULL": "4/4"}
    assert ev.graphs_on is True


def test_banner_alone_is_not_qsa_and_qsa_needs_wrap_and_plan() -> None:
    ev = parse_boot(QSA)
    assert ev.attention_boot == "qsa"
    assert ev.banner_lines  # the same banner as dense
    half = parse_boot("QSA_WRAP Qwen3NextAttention\nUsing Xe2 K8/V4 attention backend.")
    assert half.attention_boot == "unknown"


def test_enforce_eager_means_graphs_off() -> None:
    ev = parse_boot("non-default args: {'enforce_eager': True, 'model': '/model'}\nUsing Xe2 K8/V4 attention backend.")
    assert ev.graphs_on is False and ev.enforce_eager


def test_lines_since_filters_by_timestamp() -> None:
    keep = lines_since(PROD, since_epoch=1794245050.0)  # 2026-11-09T17:24:10Z
    assert len(keep) == 2 and "Capturing" in keep[0]


def test_failure_classification_order() -> None:
    assert classify_failure([], ["x"], ram_floor_hit=True).flag == "RAM_FLOOR"
    spec = "AssertionError: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1)"
    f = classify_failure(["Traceback (most recent call last):", spec], ["HTTP 500"])
    assert f.flag == "SPEC_ASSERT"
    assert classify_failure(["UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY"], ["500"]).flag == "OOM"
    assert classify_failure([], ["500"], oom_killed=True).flag == "OOM"
    assert classify_failure(["vllm.v1.engine.exceptions.EngineDeadError"], ["500"]).flag == "ENGINE_ERROR"
    assert classify_failure([], [], timed_out=True).flag == "TIMEOUT"
    assert classify_failure([], ["connection reset"]).flag == "CLIENT_ERROR"
    assert classify_failure([], []) is None
