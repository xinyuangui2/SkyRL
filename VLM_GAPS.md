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
- **Verification (2026-09-24, 8xH100 80GB, driver 580.178.04 / CUDA 13.0):** the pyproject pin is
  `vllm==0.30.0`, and `uv run --isolated --extra fsdp` installs vllm 0.30.0, torch 2.13.0+cu130 and
  transformers 5.16.1. The stock `run_geometry3k.sh` (no `[tool.uv.sources]` override) completes eval_before_train
  and 10+ training steps with the VLM generator.
- **Status:** verified. The override notes are removed from `vision_language_rl.mdx`,
  `geometry3k.mdx`, `visgym.mdx` and both geometry3k run scripts on this branch.

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

## 6. Checkpointing dominates checkpoint steps (165 s per save)
- **Symptom:** with `ckpt_interval=5`, step 5 took 259 s against about 95 s for a normal step:
  105 s eval plus 165 s `save_checkpoints` (FSDP policy with optimizer state, 8B, written to
  `/mnt/cluster_storage`). That's roughly 2x wall-clock overhead amortized over 5 steps.
- **Where:** `run_geometry3k.sh` (`trainer.ckpt_interval=5`, `eval_interval=5`). Not VLM-specific.
- **Proposed fix:** default the recipe to a larger `ckpt_interval` (e.g. 20) or `max_ckpts_to_keep`,
  and note that shared/NFS storage is slow for full optimizer checkpoints.
- **Status:** open (recipe tuning).

## 7. Eval metrics only reach the tracker, and the per-dataset key is `unknown`
- **Symptom:** eval accuracy never appears in the console log, only in wandb and
  `dumped_evals/*/aggregated_results.jsonl`. The per-dataset keys are `eval/unknown/*` because
  `geometry_3k_dataset.py` doesn't set `data_source` on its records.
- **Where:** `examples/train/geometry3k/geometry_3k_dataset.py` (record schema),
  `skyrl/train/evaluate.py` (no console summary).
- **Proposed fix:** add `data_source: "geometry3k"` to the dataset records, and log a one-line
  `eval/all/pass_at_1` summary to the console after `log_eval_results`.
- **Status:** open (small).

## 8. transformers v5 `mm_token_type_ids` workaround: the comment is stale and the fix is image-only
- **Symptom:** `model_wrapper.py:385-390` builds `mm_token_type_ids = (seq == config.image_token_id)`
  because "vLLM doesn't support transformers v5 yet". The installed stack is now vllm 0.30.0 plus
  transformers 5.16.1, so the rationale is out of date. The workaround also assumes images only
  (no video tokens) and that the config has `image_token_id`.
- **Where:** `skyrl/backends/skyrl_train/workers/model_wrapper.py:380-403`. The same 3D-position
  skipping (`position_ids=None`) appears in `megatron_model_wrapper.py:489`.
- **Proposed fix:** check whether vLLM's render endpoint now returns `mm_token_type_ids`. If it does,
  thread it through `GeneratorOutput`; if not, build it via the HF processor's own helper so video tokens are covered.
- **Status:** open (needs investigation).

## 9. The VLM generator is a fork of `agent_loop`, not hooks into the base generator
- **Symptom:** `SkyRLVLMGymGenerator.agent_loop` (`skyrl/train/generators/skyrl_vlm_generator.py:73-237`)
  is a full re-implementation. It has already drifted from the base: the `max_tokens` argument is
  accepted but never used (`:78`), and there's no custom chat template support. The generator also
  only decodes `pixel_values` from the *last* render (`:215`), and it relies on each turn's tokens
  being a prefix of the next render. Its own NOTE (`:134-137`) says that assumption fails for
  thinking models such as Qwen3-Thinking, and nothing checks for it at runtime.
- **Unsupported with VLM (hard errors):** `generator.batched=true` (`:56`, `:239`), step-wise
  trajectories (`:58`, `generators/utils.py:1005`), `use_conversation_multi_turn=false` (`:60`),
  `remove_microbatch_padding` / sequence packing and sequence parallelism
  (`model_wrapper.py:333-336`; Megatron `megatron_model_wrapper.py:304-308`), sample-support
  capture and R3 replay (`config.py:1861,1865`). Fully-async training has no explicit guard or test.
- **Proposed fix:** (a) assert a prefix match between consecutive renders, or fall back to
  per-turn token re-extraction; (b) honor `max_tokens`; (c) longer term, fold the
  render-based path into the base generator behind a renderer interface so the text and VLM paths share code.
  Packing for VLM needs per-model 3D mRoPE position ids (`get_rope_index`), which is the biggest perf item,
  because `remove_microbatch_padding=false` currently pads every sequence to the batch max (3061 tokens
  observed against a ~1300 mean response).
- **Status:** open (proposal).

## 10. VLM detection is duplicated three times
- **Symptom:** `hasattr(config, "vision_config")` is repeated in `model_wrapper.py:182`,
  `megatron_worker.py:194` and `fsdp_worker.py:64`, while `skyrl/utils/tok.py` already has `check_is_vlm`.
- **Proposed fix:** use the shared helper everywhere.
- **Status:** open (cleanup).

## 11. No Megatron VLM recipe, and the Megatron VLM SP guard checks the wrong knob
- **Symptom:** the geometry3k scripts are FSDP-only. Megatron VLM is covered only by unit tests
  (`test_megatron_vlm_init.py`, Qwen3-VL-2B, TP2/PP1 and TP1/PP2 forward).
- **Where:** `examples/train/geometry3k/`. Guard: `megatron_model_wrapper.py:_assert_vlm_supported`.
- **Observation:** Megatron turns on its own sequence parallelism whenever TP>1
  (`megatron_worker.py:291`, `provider.sequence_parallel = tp > 1`). The VLM guard only checks the
  FSDP/Ulysses knob `trainer.policy.sequence_parallel_size`, so a TP=2 VLM run passes the guard
  with Megatron SP on. The guard's docstring says SP is unsafe for VLMs, but the TP2 forward test and
  this run both work. Either the guard message is stale for Megatron SP, or it silently allows
  something it means to block.
- **Fix (in this branch):** added `examples/train/geometry3k/run_geometry3k_megatron.sh`. It's the
  same recipe with `trainer.strategy=megatron` and `--extra megatron`, with `MEGATRON_TP` (default 2),
  `MEGATRON_PP` and `CKPT_PATH` as env vars.
- **Proposed:** make the guard's intent explicit (check `provider.sequence_parallel`, or document
  that Megatron SP is fine for Qwen3-VL), and add a docs row for the Megatron VLM recipe.
- **Status:** recipe added; guard question open. See the Megatron run results below.

## 12. Greedy eval isn't reproducible: ~10% of answers flip with identical weights
- **Symptom:** eval uses `temperature=0` (greedy). The FSDP and Megatron runs start from identical
  weights and had the same step-0 pass@1 (0.536 on 591 matched questions), yet they disagreed on 56
  questions (28 each way). vLLM batched greedy decoding isn't bitwise-deterministic, and multi-turn
  tool use (`calc_score`, up to 3 turns) amplifies the divergence.
- **Impact:** a single greedy eval of 601 questions carries roughly ±1.5-2 pts of pure decode noise, which makes
  backend or config A/B comparisons from one eval point unreliable (see the Run 2 analysis).
- **Proposed fix:** for comparisons, evaluate with `n>1` samples at temperature>0 and report mean pass@1,
  or enable vLLM batch-invariant / deterministic mode for eval. Document the noise floor in the recipe docs.
- **Status:** open.

## 13. Critic path ignores `pixel_values`
- **Symptom:** the critic forward (`worker.py:1629-1634`) passes only `sequences`/`attention_mask`; no
  vision inputs. `CriticModel` is built from `AutoModel._model_mapping[type(config)]`
  (`model_wrapper.py:632`), i.e. the multimodal base for a VLM config, so PPO/GAE with a critic on a VLM
  would score image-placeholder tokens with no image attached.
- **Proposed fix:** reject `critic.model.path` + `vision_language_generator` in config validation
  until `pixel_values` is plumbed through `CriticWorkerBase`.
- **Status:** verified 2026-09-25. **The run fails cleanly at startup, earlier than predicted.** The missing-`pixel_values` path
  can't be reached yet. Setup: Qwen3-VL-2B policy and critic, `advantage_estimator=gae`, `critic_mini_batch_size=64`
  (the default critic mini-batch fails `validate_batch_sizes` against `train_batch_size=128`). Log: `/tmp/geo3k_verify_13.log`.
  - Crash: `FSDPCriticWorkerBase.init_model` (`fsdp_worker.py:276`) calls `get_llm_for_sequence_regression`, which builds the value head with
    `nn.Linear(config.hidden_size, 1)` (`model_wrapper.py:503`). That raises
    `AttributeError: 'Qwen3VLConfig' object has no attribute 'hidden_size'`, because for VLMs it lives under
    `config.text_config.hidden_size`. It happens in `build_models`, before any rollout.
  - So today a VLM critic is a hard error, not a silent one. But a one-line `hidden_size` fix would expose the
    silent path this entry describes (the critic forward passes no `pixel_values`).
  - **Proposed fix (unchanged, now more urgent):** reject `critic.model.path` with a VLM config (or
    `vision_language_generator=true`) in config validation, with a clear message, until `pixel_values`/`image_grid_thw`
    are plumbed through `CriticWorkerBase` and the value head uses `text_config.hidden_size`.

## 14. LoRA lands on the vision tower; rollout never sees those deltas
- **Symptom:** default `target_modules="all-linear"` with `exclude_modules=None` (`config.py:110-116`);
  `run_geometry3k_lora.sh` sets no exclusion, so adapters are trained on `visual.*`/merger and exported
  (`fsdp_worker.py:159-180`). vLLM applies LoRA only to the language model of multimodal models, so the
  trainer's policy diverges from the rollout policy.
- **Proposed fix:** default `exclude_modules` to vision modules when `is_vlm`; document the choice.
- **Status:** **confirmed** 2026-09-25 (Qwen3-VL-2B, `run_geometry3k_lora.sh`, 2 steps; logs
  `/tmp/geo3k_verify_14{a,c}.log`; W&B https://wandb.ai/sky-posttraining-uc-berkeley/geometry3k-verify/runs/x3lg11bn and
  https://wandb.ai/sky-posttraining-uc-berkeley/geometry3k-verify/runs/cap8edt0).
  - (a) The exported adapter (`lora_sync_path/adapter_model.safetensors`) has **600 tensors: 392 on
    `language_model`, 208 on `visual`** (16 on `visual.merger`, 12 on `deepstack_merger_list`). PEFT expands
    `all-linear` to include the vision names `qkv`, `attn.proj`, `linear_fc1`, `linear_fc2`. After 2 steps all
    104 visual `lora_B` matrices are nonzero (mean norm 0.0168 vs 0.0194 on the LM), so training does update the vision tower.
  - (b) vLLM 0.30 drops them silently. All 8 engines log only an INFO line at startup:
    `Qwen3VLForConditionalGeneration supports adding LoRA to the tower modules. If needed, please set
    enable_tower_connector_lora=True`. With that off, `LoRAModelManager` never wraps tower modules
    (`vllm/lora/model_manager.py:209-216`). The adapter still loads without an unexpected-module error, so the visual
    deltas are discarded with no warning. The trainer's policy (LoRA'd vision tower) and the rollout policy (base
    vision tower) diverge.
  - (c) With `trainer.policy.model.lora.exclude_modules='.*visual.*'` (a PEFT regex), the export has 392 tensors, all
    `language_model`, and both steps train normally.
  - **Fix options:** default `exclude_modules` to `.*visual.*` when the model is a VLM (the minimal change), or pass
    `enable_tower_connector_lora=True` to vLLM (marked experimental in vLLM) to keep training the tower.

## 15. Prompt-length filter undercounts image tokens; oversized prompts silently waste samples
- **Symptom:** the dataset filter uses `tokenizer.apply_chat_template`, which emits one placeholder per
  image instead of the processor-expanded count (hundreds for geometry3k). Oversized prompts pass the
  filter, then hit `skyrl_vlm_generator.py:155` (`len(input_ids) > max_input_length`) before any
  generation and return an empty response with reward 0. No error, no metric.
- **Prior art:** verl/EasyR1 `filter_overlong_prompts` runs the processor in `doc2len`.
- **Proposed fix:** measure prompt length via the processor (or vLLM render) in the filter, and log a
  `truncated_at_turn_0` count.
- **Status:** verified 2026-09-25. **The undercount is real, but it doesn't bite geometry3k at 1024.** The actual waste comes from
  `generator.max_input_length` on later turns. Log: `/tmp/geo3k_verify_15.log`.
  - Tokenizer length (what the dataset filter uses), train split: p50 140, max 239. Processor length with
    images expanded: p50 245, p90 406, p99 576, max 844 (test split: max 686). Images add 63-699 tokens that
    the filter doesn't see. Rows with (b) > 1024 while (a) <= 1024: **0** (train and test). At a 512 limit it
    would be 54 train and 20 test rows.
  - Eval dumps (Run 1/2): **0** empty responses, so no trajectory is cut at turn 0.
  - **But** `generator.max_input_length` defaults to `trainer.max_prompt_length` (`config.py:1826`), so it's
    also 1024, and the VLM generator checks it against the *whole* conversation before every turn
    (`skyrl_vlm_generator.py:155`). With `max_generate_length=2048` per turn, a wrong first answer
    followed by the env's "try again" observation usually already exceeds 1024 tokens, so the loop breaks with
    `stop_reason=length` and reward 0 before the retry turn. These trajectories end in the env's
    retry prompt plus `<|im_start|>assistant\n` and have no final answer. Count of eval trajectories cut this way:
    **27/601 at step 0 (FSDP), 77/601 at step 30 (FSDP), 93/601 at step 90 (Megatron)**. The count grows as
    the policy learns to use `calc_score`. The remaining length stops are real 2048-token turns (215, 104 and 56 respectively).
  - **Proposed recipe fix:** set `generator.max_input_length` explicitly in the geometry3k scripts
    (e.g. 4096 = prompt <= 844 + a 2048-token turn + observations), and log a per-batch count of
    trajectories stopped by `max_input_length`. The filter fix (measure with the processor) is still worth doing
    for image-heavy datasets.

## 16. `mm_processor_cache_gb=0` is hard-coded, so images are re-processed every render
- **Symptom:** `inference_servers/utils.py:200` disables vLLM's multimodal processor cache (since #1494).
  The VLM generator re-renders the full conversation each turn (`skyrl_vlm_generator.py:142`), so every
  image goes through the HF image processor O(turns x images x n_samples) times, and `pixel_values`
  travel as base64 on every render and every generate (`remote_inference_client.py:300,819`).
- **Prior art:** verl only zeroes this for vLLM < 0.22 (pause/resume cache desync); we're on 0.30.
- **Proposed fix:** re-enable the cache (config knob, default > 0) once pause/resume is confirmed safe.
- **Status:** verified 2026-09-25. **Safe to enable, but no measurable speedup on geometry3k.** Logs:
  `/tmp/geo3k_verify_16{a,b}.log`. W&B: baseline https://wandb.ai/sky-posttraining-uc-berkeley/geometry3k-verify/runs/3q1mk7me,
  cache=4 GB https://wandb.ai/sky-posttraining-uc-berkeley/geometry3k-verify/runs/0qci777k. Setup: 4 steps each, Qwen3-VL-8B FSDP, stock recipe.
  - The override works. `build_vllm_cli_args` gives `mm_processor_cache_gb=0` by default, and
    `engine_init_kwargs={"mm_processor_cache_gb": 4}` produces 4 (it's applied after the hard-coded overrides). vLLM
    doesn't log its engine args to the Ray worker logs, so this was confirmed by calling the builder with the run's config.
  - Per-step generate time: baseline 32.2 / 30.2 / 30.3 / 30.6 s, cache 32.4 / 29.7 / 30.2 / 31.2 s. Step time:
    109.5 / 91.1 / 95.3 / 89.5 s vs 111.5 / 92.1 / 91.9 / 91.8 s. Rewards (0.488 / 0.533 / 0.451 / 0.549 vs
    0.471 / 0.541 / 0.436 / 0.518) and response lengths (1114-1322 vs 1183-1333) are in the same range.
  - No errors across 4 weight-sync pause/resume cycles with the cache on, so the #1494 desync didn't reproduce on vLLM 0.30.
  - Why no gain here: geometry3k has one small diagram per prompt (63-699 image tokens, p50 95) and at most 3 renders
    per trajectory, so HF image preprocessing is a negligible part of the ~30 s generate phase. It may matter
    for image-heavy multi-turn envs such as VisGym (a new image every step), which is worth an A/B there before changing the default.

## 17. Microbatch cost is rows x batch-max-len under `remove_microbatch_padding=False`
- **Symptom:** the token budget counts `attention_mask.sum()` (`worker_utils.py:286`), but
  `_create_microbatch_from_indices` (`:299-309`) keeps the padded `seq_len`, so the real cost of a VLM
  microbatch is rows x max-len. Vision-encoder cost (4 patches per placeholder on Qwen3-VL) is not
  counted at all. Follow-on of #9.
- **Status:** open.

## 18. Placeholder-token validity masking is missing
- **Symptom:** rows that contain `image_token_id` but no attached media are not masked, so a
  mixed/degenerate batch would make the model demand missing features (relates to #4).
- **Prior art:** NeMo-RL `build_media_token_validity_mask`; EasyR1 maps `image_token_id -> -100`.
- **Status:** open.

## 19. Qwen3-VL deepstack under packing / CP
- **Symptom:** any packing or CP implementation for #9 must also slice `deepstack_visual_embeds` /
  `visual_pos_masks`; verl, AReaL, ROLL and EasyR1 all carry this patch. Not a bug today (packing is
  blocked), but a prerequisite for #9.
- **Status:** open (design note).

## 20. No VLM trainer-vs-inference logprob parity test
- **Symptom:** #2276 added the text parity check; nothing exercises it with `pixel_values`. ROLL has a
  full VLM suite (vLLM -> FSDP2 vs HF, incl. CP2).
- **Proposed fix:** extend the #2276 example to Qwen3-VL and add a CPU mRoPE position-id test.
- **Status:** open.

## 21. Eval dumps are image-blind; parquet has no image dedup or size cap
- **Symptom:** `trainer_utils.py:337` decodes `prompt_token_ids`, so dumps are runs of `<|image_pad|>`
  with no image reference. `geometry_3k_dataset.py:42-49` stores one base64 JPEG per row at default
  quality with no resize cap, and the same bytes are re-sent per sample per turn (#16).
- **Proposed fix:** dump the original prompt with data URIs replaced by an image hash; cap image size
  in prep.
- **Status:** open.

## 22. Docs say Megatron VLM is "not wired"; visgym doc still cites the removed vLLM override
- **Where:** `vision_language_rl.mdx` (Limitations + broken `skyrl-train/` SFT link),
  `docs/content/docs/examples/visgym.mdx:29-33`.
- **Status:** open (docs).

## Recheck notes (2026-09-25)
Corrections to the earlier framework comparison: verl's `freeze_vision_tower` is config-only (nothing
reads it); SkyRL can already reach `limit_mm_per_prompt` / `mm_processor_cache_gb` via
`engine_init_kwargs` (not first-class knobs); `min/max_pixels` and a vision-tower freeze remain
missing. A future freeze flag needs a name filter in `FsdpWeightSource` (`sources.py:66-77`) or the
frozen tower is re-synced every step. Verified OK: image observation tokens are loss-masked, the ref
model receives `pixel_values`, the processor is saved on HF export for both backends.

## Run 1 summary (2026-09-24)
- **Setup:** 8xH100 80GB (driver 580.178.04, CUDA 13.0) on a Ray GPU worker; the head node is CPU-only.
  Stock `run_geometry3k.sh` (Qwen3-VL-8B-Instruct, FSDP, colocated, GRPO, bs=128 x n=4) with
  vllm 0.30.0, torch 2.13.0+cu130 and transformers 5.16.1. No vLLM override (entry #1 verified).
- **Launch:** `WANDB_ENTITY=sky-posttraining-uc-berkeley LOGGER=wandb DATA_DIR=/mnt/cluster_storage/geo3k/data
  EXPORT_PATH=/mnt/cluster_storage/geo3k/exports bash examples/train/geometry3k/run_geometry3k.sh
  trainer.ckpt_path=/mnt/cluster_storage/geo3k/ckpts` (needs the entry #5 fix for `WANDB_ENTITY` to reach the workers).
- **wandb:** https://wandb.ai/sky-posttraining-uc-berkeley/geometry3k/runs/hqyvet2l
- **Eval (test split, 601 examples, pass@1):** step 0 **0.534**, 5 0.526, 10 0.539, 15 0.571,
  20 0.589, 25 0.632, 30 **0.654**, 35 0.652 (+12 pts by step 30). Stopped by hand at step 38 of ~98
  to free the box for the Megatron comparison run.
- **Step time:** about 90-97 s per normal step: generate ~31 s (33%), forward logprobs ~13 s (14%),
  policy_train ~40 s (43%), weight sync ~7 s (7%). Eval adds ~105 s and checkpointing adds
  125-165 s every 5 steps (entry #6).
- **Peak GPU memory:** ~66.6-67.7 GiB of 80 GiB on each GPU (nvidia-smi, sampled 1 Hz over ~3 steps).
- **Train reward (avg_final_reward, steps 1-13):** 0.463 0.512 0.424 0.531 0.451 0.539 0.537 0.453
  0.512 0.533 0.461 0.535 0.473, noisy at first. It trends up from step ~14, reaching 0.55-0.66 around steps 30-38. Mean response is ~1.2-1.35k
  tokens, and batches pad to ~3.0k because packing is off for VLM (entry #9).
- **Blockers hit:** multi-node data path (#2), wandb entity not forwarded (#5). Both have workarounds or fixes.
  No VLM-specific crashes.

## Run 2 summary: Megatron backend (2026-09-24, completed)
- **Launch:** `WANDB_ENTITY=sky-posttraining-uc-berkeley LOGGER=wandb DATA_DIR=/mnt/cluster_storage/geo3k/data
  EXPORT_PATH=/mnt/cluster_storage/geo3k/exports_megatron CKPT_PATH=/mnt/cluster_storage/geo3k/ckpts_megatron
  bash examples/train/geometry3k/run_geometry3k_megatron.sh` (TP=2, PP=1, DP=4; otherwise identical to Run 1).
- **wandb:** https://wandb.ai/sky-posttraining-uc-berkeley/geometry3k/runs/v5xewvd2
- **Result:** the Megatron VLM path trains Qwen3-VL-8B out of the box. No code changes were needed
  beyond the recipe. The full recipe (6 epochs, 96 steps) completed with exit code 0 in 3 h 52 min
  (16:17 to 20:10), with no errors.
- **Final eval pass@1:** 0.539 at step 0, peak **0.729** at step 90, 0.720 at step 95, 0.697 at the final step 96
  (+16 to +19 pts). Full series: 0 0.539, 5 0.534, 10 0.557, 15 0.569, 20 0.596, 25 0.589, 30 0.619, 35 0.636,
  40 0.627, 45 0.657, 50 0.622, 55 0.659, 60 0.661, 65 0.677, 70 0.672, 75 0.664, 80 0.676, 85 0.694, 90 0.729,
  95 0.720, 96 0.697.
- **Train reward (16-step epoch means):** 0.507, 0.568, 0.610, 0.625, 0.662, 0.685. It rises steadily through all
  6 epochs. Mean response length fell from ~1300 to ~770 tokens.
- **Eval pass@1 (Megatron vs FSDP):** step 0 0.539 / 0.534, 5 0.534 / 0.526, 10 0.557 / 0.539,
  15 0.569 / 0.571, 20 0.596 / 0.589, 25 0.589 / 0.632, 30 0.619 / 0.654, 35 0.636 / 0.652
  (FSDP was stopped at step 38; Megatron then surpassed FSDP's best at step 45).
- **Is the step 25-30 gap real?** Per-question McNemar tests put FSDP's step-25 and step-30 checkpoints
  ahead (72 vs 46 and 73 vs 49 discordant questions, p about 0.02-0.03), but that compares single
  checkpoints from one seed each, not backends. Train-side metrics match over steps 21-34: reward 0.582 vs 0.593,
  response length 1020 vs 998, grad norm 0.233 vs 0.237, and the trainer-vs-vLLM logprob gap is lower on Megatron
  (0.0116 vs 0.0133), so there's no sign of a numerics bug. By step 35 the gap had narrowed to 1.6 pts,
  inside the noise floor (#12). One consistent difference: **Megatron's policy entropy falls faster**
  (0.236 vs 0.273 averaged over steps 21-34, lower at almost every step since step 3). Candidate causes are DP=4 vs
  DP=8 micro-batch grouping under `token_mean_legacy`, or different vision-tower handling. It's worth a
  seed-controlled rerun before drawing conclusions.
- **Step time:** 88-104 s per normal step (FSDP: 90-97 s). The split is generate ~32 s, forward logprobs ~13 s,
  policy_train ~37-41 s, weight sync 6-13 s. Checkpoint saves take 141-154 s (FSDP: 125-165 s), so gap #6 applies here too.
- **Peak GPU memory:** ~70.7-72.1 GiB of 80 GiB per GPU, about 4 GiB more than FSDP.
- **Step-1 metrics vs FSDP:** entropy 0.355 vs 0.343, grad_norm 0.226 vs 0.179, clip_ratio 4.3e-4 vs 3.8e-4.
