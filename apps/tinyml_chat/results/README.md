# TinyML Chat: raw benchmark logs

Run on 2026-11-12 with `HF_HUB_OFFLINE=1 python tinychat.py --bench` (SmolLM2-360M-Instruct, CPU only, torch 2.14.0+cpu,
6 threads). Machine in `environment.txt` (i5-11400H, 15.7 GB RAM, Windows, other programs open).

- `bench.log`, `bench_run2.log`: two independent runs of the same command.
- `test_tinychat.log`: `python test_tinychat.py` (self-check vs transformers' `LlamaForCausalLM`), "all checks passed".
- `token_counts.txt`: the perplexity text is **113 tokens**; the AWQ calibration text (a different paragraph) is 140 tokens.

| precision | weights | perplexity (both runs) | tok/s run 1 | tok/s run 2 | tok/s in report |
| --- | --: | --: | --: | --: | --: |
| fp32 | 1447 MB | 33.19 | 4.9 | 12.3 | 15.4 |
| bf16 | 724 MB | 33.22 | 14.3 | 23.1 | 20.1 |
| int8 | 363 MB | 33.22 | 6.2 | 5.8 | 6.0 |
| int4 | 244 MB | 40.58 | 2.1 | 2.2 | 2.3 |
| awq4 | 244 MB | 34.84 | 3.0 | 2.5 | 2.9 |

Perplexity is deterministic and identical across runs and to the report. Speed is a single greedy 64-token decode, so it is
noisy: fp32 and bf16 (memory-bandwidth bound) swing by 2-3x between runs on a shared laptop, while int8/int4/awq4
(compute bound in PyTorch's integer kernels) stay within ~20%. The ordering bf16 > fp32 > int8 > awq4 > int4 held in the
three measurements; absolute fp32/bf16 numbers should be read as a range, not a point.
