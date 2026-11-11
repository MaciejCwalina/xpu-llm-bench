"""xpu-llm-bench: consistent, comparable LLM-serving benchmarks on Intel XPU (vLLM)."""

__version__ = "0.1.0"

# Second timestamp on every record ("*_local"): $XPU_BENCH_TZ, else the site
# config's `timezone`, else the system's local zone. UTC is always recorded too.
TZ_ENV = "XPU_BENCH_TZ"
