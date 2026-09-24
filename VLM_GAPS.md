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

## 2. Recipe assumes the driver and GPU workers share `$HOME` (multi-node Ray)
- **Symptom:** on a cluster with a CPU-only head and a GPU worker (e.g. Anyscale), the run fails with
  `FileNotFoundError: Unable to find '/home/ray/data/geometry_3k/train.parquet'`. The script
  generates the dataset on the driver's local disk, but `geometry3k_entrypoint` is a Ray task that
  gets scheduled on the GPU node.
- **Where:** `examples/train/geometry3k/run_geometry3k.sh` (defaults for `DATA_DIR`, `EXPORT_PATH`,
  and the hard-coded `trainer.ckpt_path=$HOME/...`), same in `run_geometry3k_lora.sh`.
  Not VLM-specific, but it's the first thing a VLM user on a multi-node box hits.
- **Workaround:** `DATA_DIR=/mnt/cluster_storage/geo3k/data EXPORT_PATH=... bash run_geometry3k.sh trainer.ckpt_path=/mnt/cluster_storage/geo3k/ckpts`.
- **Proposed fix:** make `CKPT_PATH` an env var like the others, and add a doc note that
  data, ckpt and export paths must be on storage every node can see.
- **Status:** worked around (not changed in-tree).

## 3. Docs say the dataset script writes `val.parquet`, but the recipe evaluates on `test.parquet`
- **Symptom:** `geometry_3k_dataset.py` writes train, val (300) and test (601). The run scripts use
  `test.parquet` for `data.val_data`, and `val.parquet` is never used. `geometry3k.mdx` only mentions
  `train.parquet` / `val.parquet`.
- **Where:** `docs/content/docs/examples/geometry3k.mdx:41`, `run_geometry3k*.sh`.
- **Proposed fix:** make the docs name `test.parquet` as the eval set (or change the scripts to use
  val for periodic eval and keep test for the final number).
- **Status:** open (doc nit).

## 4. Mixed image and text-only rollouts in one batch would crash
- **Symptom (by code reading, not hit in geometry3k):** `SkyRLGymGenerator` builds
  `pixel_values` / `image_grid_thw` lists with `None` for text-only rollouts whenever *any*
  rollout has images (`skyrl/train/generators/skyrl_gym_generator.py:1085-1091`). The trainer then does
  `TensorList(pixel_values)` (`skyrl/train/trainer.py:894`), and `TensorList.__init__` calls
  `tensors[0].device` on every element (`skyrl/backends/skyrl_train/training_batch.py:100`), so it
  raises `AttributeError` on `None`. The FSDP forward `torch.cat(pixel_values.tensors)` would fail too.
- **Proposed fix:** have `TensorList` accept `None` or zero-size entries (e.g. an empty
  `[0, dim]` tensor with a `[0, 3]` grid) and skip them when concatenating. Add a unit test with a mixed batch.
- **Status:** open (proposal).

## 5. `WANDB_ENTITY` isn't forwarded to Ray workers
- **Symptom:** `wandb.errors.errors.CommError: the provided API key cannot access this resource`
  at `Tracking.__init__`. The key's default entity (`anyscale-llm-forge`) isn't writable by it. The
  only entity it can write to is a team (`sky-posttraining-uc-berkeley`), but setting `WANDB_ENTITY`
  in the launching shell had no effect: `prepare_runtime_environment` forwards `WANDB_API_KEY` and
  an allowlist of vars, and `WANDB_ENTITY` wasn't on that list. The tracker runs inside the Ray
  task, so it never saw the entity.
- **Where:** `skyrl/train/utils/utils.py` (`prepare_runtime_environment` forward list).
  Not VLM-specific.
- **Fix (minimal, in this branch):** add `WANDB_ENTITY` and `WANDB_BASE_URL` to the forwarded env vars.
  Longer term: a `trainer.wandb_entity` config field, or forward all `WANDB_*` like `SKYRL_*`.
- **Status:** fixed on branch.
