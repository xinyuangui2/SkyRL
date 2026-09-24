# VLM support gaps

Running log of gaps found while exercising VLM training in SkyRL.
Starting point: `examples/train/geometry3k` (Qwen3-VL-8B-Instruct, FSDP, 8xH100).

Format: one entry per gap — symptom, where, proposed fix, status.

## 1. Stale vLLM prereq in VLM docs/scripts
- **Symptom:** `vision_language_rl.mdx` and both geometry3k run scripts say VLM needs a
  local vLLM source override because the repo pins `vllm==0.19.0`.
- **Reality:** repo pins `vllm==0.30.0` (#2271), which already contains the required
  commit `80b18230e` (v0.30.0 is 5518 commits ahead of it).
- **Where:** `docs/content/docs/tutorials/vision_language_rl.mdx`,
  `examples/train/geometry3k/run_geometry3k.sh`, `run_geometry3k_lora.sh`,
  `docs/content/docs/examples/geometry3k.mdx`.
- **Fix:** drop the override note once a stock run is confirmed.
- **Status:** pending verification on 8xH100.
