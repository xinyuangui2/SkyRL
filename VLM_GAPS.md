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
- **Status:** open (proposal). **Confirmed for Tinker on FSDP** (Run 5, test 2d): a `forward_backward` with 2 image and 2 text-only
  datums on Qwen3-VL-8B returns a 400 whose body is the raw worker traceback, ending in transformers
  `qwen3_vl.get_image_features` -> `vision_utils.get_vision_interpolation_indices_and_weights`:
  `RuntimeError: repeats must have the same size as input along dim, but got repeats.size(0) = 0 and input.size(0) = 1`.
  **Not reproduced on Megatron** (colocated, `merge_lora=false`): the same mixed batch runs and per-row NLL matches each row run
  alone (`[0.0, 0.0, 0.0, 0.0091]` both ways).

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
- **Status:** **measured** 2026-09-25 (stock `run_geometry3k.sh`, Qwen3-VL-8B, step-0 weights, eval only; logs
  `/tmp/geo3k_verify_12{a,b,c}.log`, dumps under `/mnt/cluster_storage/geo3k/verify12/`).
  - Two greedy evals with the identical config: pass@1 **0.534 vs 0.526**. Per question, **530/601 agree (88.2%)**
    and 71 flip (38 one way, 33 the other). This matches the 56 flips between the Run 1 and Run 2 step-0 evals. The per-eval noise floor
    for greedy is about ±1-1.5 pts, and single-point differences up to ~3 pts aren't meaningful.
  - `generator.eval_n_samples_per_prompt=4`, `generator.eval_sampling_params.temperature=0.6`: mean pass@1 **0.459**
    over 2404 samples (pass@4 0.614). The per-draw accuracies are 0.468 / 0.439 / 0.463 / 0.466 (std 1.3 pts), so the 4-sample mean
    carries about ±0.7 pts. Temperature-0.6 sampling scores ~7 pts below greedy on this model, so
    sampled and greedy numbers aren't comparable with each other.
  - **Recommendation:** for backend or config A/B comparisons, use `eval_n_samples_per_prompt>=4` and compare mean pass@1
    (or run greedy eval at least twice). Don't read differences under ~3 pts from one greedy eval.

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
  **Also confirmed on the Tinker side** (Run 5, test 6): the r=32 LoRA sampler exports from both FSDP and Megatron
  (`merge_lora=false`) contain 232 of 736 tensors under `model.visual.*` (27 ViT blocks x `attn.qkv`/`attn.proj`/`mlp.linear_fc1`/
  `linear_fc2`, plus 16 `deepstack_merger_list.*` tensors). Megatron's trainable-param log for a Tinker LoRA model:
  `language_model=73.1M/8263.9M, vision_model=17.7M/594.1M`.
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
- **Status:** open. Mixed-batch crash confirmed for Tinker on FSDP (see #4, Run 5 test 2d).

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

## 23. VLM generator feeds `<|im_end|>` back into the conversation, so every turn boundary gets a doubled `<|im_end|>`
- **Symptom:** in the eval dumps, every trajectory with more than one assistant turn contains
  `...</tool_call><|im_end|><|im_end|>\n<|im_start|>user\n...` (75/75 multi-turn at FSDP step 0, 192/192 at
  Megatron step 90). No single-turn trajectory has it.
- **Root cause (runtime probe, 2026-09-25):** `gen_text` **ends with a literal `<|im_end|>`**. `RemoteInferenceClient.generate`
  doesn't use vLLM's output text. It re-detokenizes `response_ids` via `self.detokenize()` →
  `self.tokenizer.batch_decode(token_ids)` **without `skip_special_tokens=True`**
  (`remote_inference_client.py:618-624, 1016-1017`). So the `skip_special_tokens=True` in the sampling params (`engine_utils.py:59`)
  never applies to `responses`, which contradicts the contract documented in `base.py:43-47`. The text generator
  works around this by stripping a trailing eos before appending the assistant message (`skyrl_gym_generator.py:838-840`).
  The VLM generator appends `gen_text` as-is (`skyrl_vlm_generator.py:~190`), and the chat template adds its own `<|im_end|>`.
- **Probe data** (temporary instrumentation, Qwen3-VL-2B, 1 step, 3 multi-turn boundaries; log `/tmp/geo3k_verify_23.log`):
  - `gen_ids[-4:]` = `['}}', '</tool_call>', '<|im_end|>']` (one eos); `gen_text[-40:]` ends in `</tool_call><|im_end|>`; no `<think>`.
  - Prefix check: `rendered[:prev_len] == prev_input_ids` in 3/3. The rendered assistant region up to its first `<|im_end|>` is exactly
    `len(gen_ids)-1` tokens, so `pending_obs_offset = len(input_ids)+len(gen_ids)` is arithmetically correct, and the obs slice
    starts at the template's second `<|im_end|>`: `['</tool_call>', '<|im_end|>', '<|im_end|>', 'Ċ', '<|im_start|>', 'user']`.
  - **vLLM's turn-2 prompt contains the double too** (3/3), so rollout and trainer see the same doubled token. The harm
    is an off-template context for every turn >= 2 (and an extra masked token), not a logprob mismatch.
  - **Separate, smaller mismatch (1/3):** the render re-tokenizes the assistant text non-identically to what the model generated,
    with the same length but different ids (generated `')'`, `'"}}'` vs rendered `')"'`, `'}}'`). There the trainer sequence has
    the generated ids while vLLM's turn-2 prompt had the re-tokenized ones, so turn-2+ tokens are scored on a slightly
    different context than they were sampled from. This is the BPE non-canonical-tokenization case of the NOTE at `skyrl_vlm_generator.py:134-137`.
- **Proposed fix:** (1) minimal: strip a trailing `tokenizer.eos_token` from `gen_text` before appending the assistant message in the
  VLM generator, mirroring `skyrl_gym_generator.py:838-840`. (2) Better: make `RemoteInferenceClient.generate` honor its contract
  (`batch_decode(..., skip_special_tokens=True)` for `responses`; the HTTP `/detokenize` fallback needs the same). This needs a check that
  no model relies on special tokens surviving in `responses`, e.g. models whose `<think>` is a special token.
  (3) For the re-tokenization case, the robust fix from #9(a): assert `input_ids[:pending_obs_offset] == prev_input_ids + gen_ids`,
  and on mismatch compute obs tokens by diffing the renders of `[..., assistant]` and `[..., assistant, obs]` instead of trusting the offset.
- **Status:** root cause identified; fix not applied.

## 24. No large-model VLM recipe; Qwen3-VL MoE on Megatron (EP) is untested
- **Symptom:** every VLM recipe and test uses Qwen3-VL-2B/8B on one node. Nothing exercises
  expert parallelism with a VLM, multi-node colocated vLLM with images, or the memory profile of a
  30B-class VLM with full optimizer state. Megatron-Bridge does ship a registered `Qwen3VLMoEBridge`
  (`models/qwen_vl/qwen3_vl_bridge.py`, source `Qwen3VLMoeForConditionalGeneration`, with
  `vision_expert_{model,tensor}_parallel_size` provider knobs), and SkyRL dispatches via
  `AutoBridge`, so the path exists but has never been run here. geometry3k also saturates for 8B
  (0.73 pass@1), so a larger model needs a harder dataset to show signal.
- **Fix (in this branch):** `examples/train/geometry3k/run_geometry3k_30b_a3b_megatron.sh`:
  Qwen3-VL-30B-A3B-Instruct, 2x8 H100, TP=2/PP=1/EP=8/ETP=1, 4 vLLM engines x TP=4, n=8, full
  recompute, `generator.max_input_length=8192` (#15). Fallbacks documented in the script header.
- **Open:** run it; then swap in a harder multimodal-reasoning set (e.g. ViRL39K or MMK12) once the
  pipeline is proven. Check whether `vision_expert_*_parallel_size` needs exposing in
  `megatron_config` for the vision tower under EP.
- **Status:** run to completion on 2x8 H100 (Run 3, 2026-09-26): 0.577 -> 0.807 pass@1. Needed TP=4/PP=2 (#49).
  `vision_expert_*_parallel_size` wasn't needed; the vision tower worked under EP=8 as-is.

## 25. Tinker: non-colocated and external-inference sampling silently drop images
- **Symptom:** colocated `sample` forwards raw chunks to the engine client, which renders images
  (`skyrl_train_backend.py:1309-1322`, `remote_inference_client.py:673-757`). With
  `colocate_all=false` or `external_inference_url`, requests go through
  `SkyRLTrainInferenceForwardingClient` / `ExternalInferenceClient` (`api.py:347-362`), and both call
  the text-only `render_model_input` (`skyrl_train_inference_forwarding.py:160`,
  `external_inference.py:111`; `renderer.py:31-38` keeps only chunks with `.tokens`). Image chunks
  vanish with no error; the model answers a prompt with no image.
- **Proposed fix:** route those clients through `VLLMRenderer`, or reject image chunks with a 400
  until they do.
- **Status:** **confirmed** (Run 5, test 4; FSDP, `trainer.placement.colocate_all=false`, Qwen3-VL-8B). After
  `save_weights_and_get_sampling_client`, the cats image prompt is 321 tokens on the client, but `prompt_logprobs` shows the
  server saw 21 tokens: all 300 image-pad tokens were dropped and only `<|vision_start|><|vision_end|>` remained. The model answers
  "The image shows a close-up of a person's hand holding a small, dark-colored object..." (the same kind of hallucination as the
  no-image control). Two images: 623 -> 23 tokens, "The image is blank". A wrong `expected_tokens` also passes. There is no
  error and no log line. `external_inference_url` was not run (same `render_model_input` call, `external_inference.py:111`).
  Related, same mode: base-model sampling returns 400 `inference engine not ready: no proxy URL published to EngineStateDB`
  until the first sampler-weight sync, and `create_lora_training_client` returning does not mean samples will succeed
  (colocated mode queues them on the engine instead).

## 26. Tinker: JAX / skyrl-tx backend ignores images entirely
- **Symptom:** `forward_backward` and `sample` both call `render_model_input` (`skyrl/backends/jax.py:644,
  851`), so images are dropped and targets misalign. `skyrl/tx/models/` has text models only.
  Nothing rejects the request, so the failure is silent.
- **Proposed fix:** reject `ImageChunk` / `ImageAssetPointerChunk` on the JAX backend with a clear
  error now; VLM support in tx is a separate project.
- **Status:** open; **not run on GPU**: the `jax` extra fails to import (#56). By code reading,
  `get_model_class` (`skyrl/tx/utils/models.py:65-82`) has no Qwen3-VL class, so Qwen3-VL fails loudly at `create_model`.
  But `Qwen3_5ForConditionalGeneration = Qwen3_5ForCausalLM` (`skyrl/tx/models/qwen3_5.py:849`), so Qwen3.5 VL
  checkpoints load text-only and images are dropped silently, which is exactly this gap.

## 27. Tinker: `expected_tokens` is enforced on training but not on sampling; no server-side image token count
- **Symptom:** the training render path enforces `expected_tokens` (`renderer.py:81-182`), but
  `_render_for_sample` does not (`remote_inference_client.py:673-757`). The prompt-length /
  `max_tokens` check ignores image tokens (`api.py:767-778`). The SDK's `ImageChunk.length` raises
  when `expected_tokens` is unset, and the reference server returns 400 "Expected N tokens, got M
  from image" on mismatch (tinker-cookbook `image_token_count_test.py`), so clients must compute
  counts from the processor config, which `get_info` / `get_sampler` do not expose (`api.py:1285-1305,
  1380-1390`).
- **Proposed fix:** validate `expected_tokens` on the sample path with the same 400; account for image
  tokens in the length check; expose processor min/max pixels (or a token-count endpoint) in `get_info`.
- **Status:** **confirmed** (Run 5, test 3; FSDP colocated, Qwen3-VL-8B, 300-token image). `forward_backward` with
  `expected_tokens=290` or `350` gives a clear 400: `Image 0: expected_tokens=290 but render returned 300 placeholder tokens`
  (resp. `350`). `sample` with `expected_tokens=350` **succeeds silently** and answers correctly. Non-colocated sampling also
  ignores it (#25). The SDK computes `ModelInput.length` from `expected_tokens`, so a client with a wrong count builds
  misaligned `target_tokens`/`weights` without knowing. Side finding: a base-model `sample` sent before any `create_model` on
  FSDP returns 400 `'NoneType' object has no attribute 'trainer'` (#58).

## 28. Tinker: training and sampling can render images differently
- **Symptom:** `CPURenderServer` loads only tokenizer + processor (`inference_servers/render_server.py:86-106`)
  and does not receive the engine's `mm_processor_kwargs` (e.g. max_pixels). Training renders
  (CPU server before engines start, engine client after: `skyrl_train_backend.py:445-501, 695-705`)
  and sampling renders can disagree on image token counts. Same class of issue as #16/#21.
- **Proposed fix:** pass the engine's processor kwargs to the render server; add a parity test.
- **Status:** open; not exercised in Run 5 (default processor kwargs only). At defaults, training and sampling agree: the
  cookbook `get_number_of_image_patches` count (300 for 640x480) matched the render placeholder length on every path.

## 29. Tinker: VLM needs `remove_microbatch_padding=false`, but the server default is `True`
- **Symptom:** `config.py:1525` defaults to `True`; the VLM asserts (`model_wrapper.py:334`,
  `megatron_model_wrapper.py:304`) fire on the first `forward_backward` with images. The cookbook doc
  tells users to override it by hand (`cookbook.mdx`).
- **Proposed fix:** flip it automatically when the loaded model `is_vlm`, with a log line.
- **Status:** **confirmed** (Run 5; Qwen3-VL-2B, FSDP, server started without the override). The first
  `forward_backward` returns 400 `AssertionError: remove_microbatch_padding is not supported with VLM vision inputs`
  (`model_wrapper.py:334`), and so does a **text-only** datum on the same VLM server: the assert fires for any VLM model, not
  just rows with images. Every Tinker VLM user has to know the override.

## 30. Tinker: sampler export and adapter sync are untested for VLMs; Megatron export lacks the processor
- **Symptom:** FSDP HF export saves the processor (`fsdp_strategy.py:595-596`); no processor save was
  found in the Megatron sampler export path, and adapter-only tarballs (`merge_lora=false`,
  `skyrl_train_backend.py:390-401, 1555-1565`) carry only adapter files. LoRA target selection
  (`megatron_worker.py:413-431`) has no vision-tower handling (same root as #14). Training-proto
  chunks accept only `encoded_text` and `image` (`proto_serialization.py:72-89`), so
  `ImageAssetPointerChunk` is sample-only and there is no asset storage backend. Batch padding copies
  row 0's pixels into padded rows (`training_batch.py:631-634`).
- **Proposed fix:** save the processor on every export path; exclude vision modules from Tinker LoRA;
  document asset pointers as sample-only; add GPU tests for VLM forward_backward, mixed batches
  through `_to_training_batch`, and LoRA + adapter sync with images.
- **Status:** **partially confirmed** (Run 5, test 6; Qwen3-VL-8B, r=32, 6 steps teaching the answer "Zebrafish" on 2 images).
  Weight sync with images works on FSDP (ephemeral and persistent) and Megatron `merge_lora=false` (ephemeral, and persistent
  once #57 is worked around): the trained sampler answers "Zebrafish", the base still says "cats". The **Megatron adapter-only
  export lacks the processor**, as predicted. Its tarball is `adapter_config.json` + `adapter_model.safetensors` only (201 MB);
  the FSDP one also has `processor_config.json`, `chat_template.jinja`, tokenizer and `config.json` (412 MB). Both exports put LoRA on the
  vision tower (#14). Both `.tar.gz` files are uncompressed POSIX tar. Megatron with the default `merge_lora=true` cannot
  sample a trained LoRA at all (#55). First sync is slow: FSDP 174-187 s, Megatron 154 s; later FSDP syncs 9-60 s.

## 31. Tinker docs claim VLM is "Supported" without caveats; no RL-with-images example
- **Where:** `docs/content/docs/tinker/overview.mdx:42`, `limitations.mdx` (no VLM section),
  `cookbook.mdx:112-139` (only the SFT `vlm_classifier` recipe; its "needs newer vLLM" callout is
  stale per #1). `examples/tinker/` has no VLM example. A cookbook RL loop with images should work in
  colocated fsdp/megatron but is neither tested nor documented.
- **Proposed fix:** limitations section listing #25-#30; an `examples/tinker/` RL-with-images recipe.
- **Status:** **confirmed** (Run 5, test 1). The documented `vlm_classifier` client command fails as written. Four fixes are
  needed: (1) `experiment_dir=...` is required (`Missing required arguments for parameter(s): experiment_dir`); (2)
  `renderer_name=qwen3_vl_instruct`, because the recipe defaults to `qwen3_5_disable_thinking` for `Qwen/Qwen3.5-397B-A17B`
  and the doc doesn't override it; (3) pin the SDK, `--with 'tinker>=0.25,<0.26'`, because unpinned `--with tinker` resolves 0.30.4
  and dies at step 0 (#54); (4) `run_nll_evaluator=false` (or a subset), because the step-0 NLL evaluator sends the full
  caltech101 test split (19 forward requests, about 6.1k images, 6.5 MB each in SQLite). That took 952 s on the doc's server, which
  uses 1 GPU (FSDP `world_size=1` by default). With those fixes the recipe trains as described (numbers in Run 5 summary). The
  "needs newer vLLM" callout is stale (vllm 0.30.0 worked).

## 32. Megatron-Bridge freezes the LM and vision tower of Qwen3-VL MoE by default; SkyRL never overrides it
- **Symptom:** `Qwen3VLMoEModelProvider` defaults `freeze_language_model=True, freeze_vision_model=True`
  (bridge `models/qwen_vl/qwen3_vl_provider.py:272-273`, applied in `provide()` at `:334-338`); the dense
  `Qwen3VLModelProvider` defaults both to `False` (`:109-111`). SkyRL sets no `freeze_*` on the provider
  (`megatron_worker.py:290-310`; repo-wide grep for `freeze_language_model` is empty). Any Megatron run
  of a Qwen3-VL MoE (RL via #24's recipe, or SFT) silently trains only the vision projector, with no
  warning, while loss/reward/ckpt logging look normal.
- **Fix (in this branch):** `run_geometry3k_30b_a3b_megatron.sh` now passes
  `transformer_config_kwargs.freeze_language_model=false` and `freeze_vision_model=false` (the
  `setattr` loop at `megatron_worker.py:308-310` forwards them to the provider).
- **Proposed:** in `megatron_worker.init_configs`, set the three `freeze_*` fields explicitly from a new
  `trainer.policy.model.freeze_vision_tower` (+ language) config (this is also the P2 "freeze vision
  tower" feature), and log the trainable-parameter count by submodule at startup so a frozen tower is
  visible. Same applies to the SFT path (`sft_config` has no freeze knob either).
- **Status:** confirmed from source; recipe patched before its first run.

## 33. SFT: `image_url` content parts are treated as text-only
- **Symptom:** `has_images` checks only `{"type": "image"}` (`sft_trainer.py:513-517`); OpenAI-style
  `image_url` parts (the format the RL datasets use, e.g. `geometry_3k_dataset.py`) are silently
  ignored while the Qwen template still emits an image placeholder with no pixels (#18).
- **Fix:** normalize `image_url` / `video` parts in `_normalize_chat_messages` (`:385-403`).
- **Status:** open (small; verl, NeMo-RL, ms-swift, prime-rl all accept OpenAI content blocks).

## 34. SFT: mixed text/image rows crash mid-epoch instead of at load
- **Symptom:** the modality check is in `collate_sft_batch` (`:722-727`); the pretokenized loader admits
  mixed stores (`pretokenized.py:320-322`). Fix: validate homogeneity after tokenization (`:1132`).
- **Status:** open (small).

## 35. SFT: last-assistant-turn loss only; no image-token loss mask
- **Symptom:** `ALL_ASSISTANT_MESSAGES` + images raises (`:518-519`). Loss window is "tokens after the
  prompt" (`:613`), so an image inside the last assistant turn would be in-loss. Peers (verl, LLaMA-Factory,
  NeMo-RL, ms-swift, prime-rl, tinker) train on all assistant turns; only ms-swift masks image tokens explicitly.
- **Fix:** build the mask from per-turn spans via the processor, and zero `image_token_id` positions.
- **Status:** open (medium; the main SFT feature gap).

## 36. SFT: over-length VLM rows are dropped, never truncated
- **Symptom:** `:600-604` / `pretokenized.py:247-255`. Peers either truncate image-safely (ms-swift,
  prime-rl `_find_image_safe_cut`, tinker drops whole image chunks) or filter like SkyRL.
- **Fix:** right-truncate text only; drop a row only if an image span would be cut.
- **Status:** open.

## 37. SFT: no processor kwargs (min/max pixels) and the tokenization cache key ignores them
- **Where:** `get_processor(**tokenizer_kwargs)` (`:922-932`), `_compute_cache_key` (`:175-220`).
- **Fix:** `processor_kwargs` in SFTConfig, forwarded and hashed. Same knob as the RL P2 item.
- **Status:** open (small).

## 38. SFT: sequential tokenization, images encoded twice, cache materializes pixels as Python floats
- **Symptom:** VLM forces `num_workers=0` (`:1096-1098`; the spawn worker takes no processor, `:90-100`);
  each sample runs the processor twice (prompt `:578`, full `:587`); the cache path round-trips
  `pixel_values` through `Dataset.from_list` / `to_list()` (`:262,287`, ~30x float32) and the collator
  re-tensorizes nested lists per batch (`:742`).
- **Prior art:** tinker-cookbook pickles renderers by name and rebuilds the processor in the worker;
  NeMo-RL reconstructs from the model path; ms-swift lazy-tokenizes multimodal by default.
- **Fix:** rebuild the processor in workers from `model.path`; tokenize once; keep VLM rows arrow-backed.
- **Status:** open (perf; blocks any VLM SFT dataset beyond a few thousand rows).

## 39. SFT: packing is gated on `is_vlm`, not on "rows carry images"; `language_model_only` not exposed
- **Symptom:** text-only SFT of a VLM checkpoint (e.g. Qwen3.5) loses packing (`:937-941`); `build_skyrl_config_for_sft`
  does not map `language_model_only` (`sft_config.py:630-700`); `use_sequence_packing=True` + VLM leaves
  `micro_train_batch_size_per_gpu=1` (`:676`) after packing is flipped off (`:939`).
- **Fix:** expose `language_model_only` in SFTConfig; gate packing on tokenized rows having `pixel_values`;
  move the VLM override into `validate_sft_cfg`.
- **Status:** open.

## 40. SFT + RL: no freeze-vision-tower / vision LR; Megatron LoRA hits the ViT
- **Symptom:** no config field on either backend; only reachable as undocumented
  `megatron_config.transformer_config_kwargs.freeze_vision_model=true`. Megatron LoRA `exclude_modules`
  defaults `[]` (`megatron_worker.py:441`) and the bridge ViT shares `linear_qkv/linear_proj/linear_fc1/fc2`
  names, so adapters land on the vision tower on both backends (extends #14).
- **Peers:** freeze flags in 7 frameworks; `vit_lr`/`aligner_lr` in ms-swift; LoRA excludes the tower by
  default in LLaMA-Factory and prime-rl.
- **Fix:** `model.freeze_vision_tower` (+ projector) on both backends, default LoRA exclusion for VLMs, and a
  name filter in `FsdpWeightSource` so a frozen tower is not re-synced.
- **Status:** open (P2 feature; #32 is the MoE-specific symptom).

## 41. SFT: FSDP VLM SFT is untested and undocumented; CP guard has no test row
- **Symptom:** wiring exists (`model_wrapper.py:329-403`) but no test or recipe; `sft/overview.mdx:218` says
  Megatron only. `test_megatron_vlm_unsupported_parallelism_raises` has no `cp=2` row, and `validate_sft_cfg:600`
  only asserts padding removal for CP, which `setup()` later flips off.
- **Fix:** parametrize `test_vlm_train` over `strategy=fsdp`; add `run_sft_fsdp_vlm.sh`; reject CP>1 with VLM.
- **Status:** open (small).

## 42. Tinker SFT: SFT-trainer and Tinker render paths agree on image token counts only at default processor kwargs
- **Symptom:** SFT uses the HF processor template (inline `<vision_start><image_pad>xN<vision_end>`); Tinker's vLLM
  render supplies only the `<image_pad>` run and the client supplies the surrounding tokens
  (`renderer.py:109-118, 163-180`). N matches by construction only when the engine's `mm_processor_kwargs`
  are default (#28). tinker-cookbook's `image_to_chunk` computes `expected_tokens` from the HF processor
  (`get_number_of_image_patches // merge_size**2`); the cookbook's generic chat dataset builders never pass an
  `image_processor`, so multi-turn image SFT there needs a custom dataset (`vlm_classifier` is single-turn).
- **Fix:** a cross-path parity test (SFT tokenization vs `/render` vs cookbook `image_to_chunk`) on one image set.
- **Status:** open.

## 43. Qwen3.5 as a full VLM is untested (SkyRL runs it text-only today)
- **Symptom:** Qwen3.5 checkpoints are unified VL models (`Qwen3_5ForConditionalGeneration`, `vision_config`
  present). SkyRL's Qwen3.5 support is `language_model_only` via custom LM-only bridges
  (`megatron/model_bridges.py:92-164`); nothing exercises the full-VL path, although Megatron-Bridge ships
  `Qwen35VLModelProvider` (freeze defaults `False`) and vLLM 0.30 serves the VL model.
- **Fix (in this branch):** `examples/train/sft/run_sft_megatron_vlm_qwen3.5.sh` (1 node: Qwen3.5-9B; 2 nodes:
  35B-A3B with EP=8 or 27B with TP4/PP2). Untested.
- **Open:** RL with Qwen3.5 as a VLM (the generator's `image_token_id` / `mm_token_type_ids` assumptions,
  `model_wrapper.py:389-390`, need checking against the Qwen3.5 config); GDN layers + VL forward under PP.
- **Status:** recipe written, not run.

## 44. Gemma 4 (and any non-Qwen VLM) is blocked by Qwen-shaped multimodal plumbing
- **Symptom:** vLLM 0.30 registers `Gemma4ForConditionalGeneration` and Megatron-Bridge has
  `gemma4_vl_bridge/provider`, but SkyRL hard-codes Qwen's layout end to end: `image_grid_thw` in ~60 places
  across 12 files (`trainer.py`, `sft_trainer.py`, `worker.py`, both model wrappers, `renderer.py`,
  generators, `pretokenized.py`, `replay_buffer.py`), `mm_token_type_ids` derived from `config.image_token_id`
  (`model_wrapper.py:389-390`), the `language_model.` prefix (`worker_utils.py:46`), and `decode_mm_kwargs`
  expecting exactly `pixel_values` + `image_grid_thw` (`renderer.py:51-62`). Gemma 4 has `pixel_values`
  (+ audio) and no grid tensor.
- **Fix:** carry multimodal kwargs as an opaque per-sample dict (`mm_kwargs: Dict[str, TensorList]`) from
  render through `TrainingInputBatch` to `model.forward(**mm_kwargs)`, with per-model position-id handling
  delegated to HF / the bridge. That is the same refactor #9 and #19 need. Gemma 4 recipes (RL + SFT, 1 and
  2 nodes) follow once it lands; writing them now would produce scripts that cannot run.
- **Status:** open (design; prerequisite for Gemma 4, Kimi-VL, InternVL).

## 45. No end-to-end VLM CI job
- **Symptom:** GPU CI has unit-level VLM tests (`test_vlm_model_wrapper.py`, `test_skyrl_vlm_gym_generator.py`,
  `test_megatron_vlm_init.py`) but the nightly e2e family (`gpu_e2e_ci*.yaml`) is text-only, so regressions
  like #23 or #32 are invisible to CI.
- **Fix (in this branch):** `SkyRL-GPU-E2E-CI-VLM`: `.github/workflows/gpu_e2e_ci_vlm.yaml` ->
  `ci/anyscale_gpu_e2e_test_vlm.yaml` -> `ci/gpu_e2e_test_run_vlm.sh` -> `tests/train/gpu_e2e_test/geometry3k_colocate.sh`
  (Qwen3-VL-2B-Instruct, FSDP, 512-sample subset, 8 steps on l4_ci, asserts eval/train accuracy, token count and
  logprob diff via `get_summary.py`). Thresholds are initial and loose; recalibrate after ~10 nightlies.
- **Status:** written, not run (needs the `run_gpu_ci`-style Anyscale submission to validate).

## 46. SFT on Megatron: MoE VLM provider freeze defaults also apply (see #32)
- **Symptom:** `run_sft_megatron_vlm.sh` with a Qwen3-VL MoE checkpoint would train only the projector.
- **Fix:** same explicit `freeze_*=false` overrides; the Qwen3.5 SFT recipe already passes them.
- **Status:** open (recipe-level mitigation only).

## 47. Media transfer: images and pixel tensors round-trip through the client on every hop
- **Symptom:** an image enters as base64 JSON (`ImageChunk`, `api.py:585`; RL datasets as data URIs), goes to vLLM
  `/render` as a data URI (`renderer.py:136-146`), comes back as base64 msgpack `pixel_values` (`kwargs_data`,
  `renderer.py:41-62`), and is **re-sent as `features` on every `generate`** (`remote_inference_client.py:300, 819`).
  The VLM generator re-renders the full conversation each turn, and `mm_processor_cache_gb=0` (#16) prevents vLLM
  from returning `kwargs_data=None` on a cache hit, which its render protocol supports. Cost per image per hop is
  roughly 4-10 MB (1,200 patches x 1,536 values for a 640x480 Qwen3-VL image), x turns x n_samples. Tinker's
  `ImageAssetPointerChunk.location` is passed through as a URL; there is no asset store, and the reference Tinker
  SDK has no public upload API either (its server dedups internally).
- **Peers:** every surveyed framework keeps pixel tensors engine/trainer-side and ships them once per prompt
  (verl, slime, NeMo-RL, prime-rl, AReaL, ROLL); NeMo-RL shares one processor output across the n repeats
  (`share_immutable_media`); AReaL caches processor output per rollout group; ROLL down-casts mm features to bf16
  for Ray transfer; vLLM offers content hashing (`mm_hashes`, `multi_modal_uuids`) so media bytes can be omitted on
  expected cache hits; SGLang EPD / vLLM EC connectors move vision embeddings instead of pixels.
- **Proposed fix, in order:** (1) enable the vLLM processor cache and pass `mm_hashes` / `multi_modal_uuids` on
  `generate` instead of tensors, so pixel tensors never leave the engine; (2) render once per prompt and reuse across
  turns and n samples (only new env images are rendered); (3) server-side content-addressed dedup for Tinker: hash
  `ImageChunk` bytes, store once, accept `image_asset_pointer` with `location=asset://<hash>` (client unchanged);
  (4) ship `pixel_values` to the trainer once per prompt in bf16; (5) later, a vision-embedding cache or encoder
  disaggregation for multi-image / video workloads.
- **Status:** open (design). **Measured** (Run 5, test 7; Qwen3-VL-8B, 640x480 COCO image, 300 tokens):
  - Client to server: the SDK sends the JPEG as base64. The stored request is 85 KB for a 1-image `sample`, 88 KB for a
    1-image `forward_backward`, and 366 KB for 4 images. The source JPEG is 173 KB; the cookbook helper re-encodes it with PIL.
  - `/v1/chat/completions/render`: request 231 KB (source JPEG as a data URI), response **4.92 MB**
    (`kwargs_data` 4,915,436 bytes), about 0.04 s per call. That is about 28x the JPEG and about 58x the re-encoded wire image.
  - Pixel tensors are re-sent on every `generate`, by code (`remote_inference_client.py:751-754, 818-819` put
    `features.kwargs_data` into the generate payload); this was not intercepted on the wire.
  - With `generator.inference_engine.engine_init_kwargs.mm_processor_cache_gb=4` (the override does replace the hard-coded 0,
    `inference_servers/utils.py:200,255-262`), `/render` still returns the full 4.9 MB `kwargs_data` on 3 identical calls:
    never `kwargs_data=None`. So enabling the cache alone does not shrink the render hop. Training and sampling stay correct with
    it on (forward NLL identical over 5 calls, samples identical).
  - Render cost is small: py-spy on the engine during cookbook training shows render + `_to_training_batch` at about 1% of
    wall time (about 0.3 s per 32-image step).

## 48. Qwen3-VL-30B-A3B loops under greedy eval; ~25% of eval trajectories hit the 4096-token turn cap
- **Symptom:** in Run 3, 154/601 eval trajectories at step 0 end with `stop_reason=length` at
  `max_generate_length=4096`, all with reward 0. None are cut by `max_input_length` (so #15 is fixed). 129/154 stop in
  turn 1, before any tool call (only 1 capped final turn contains a `<tool_call>`). 114/154 are degenerate repetition
  loops (the second half of the turn has under 50% unique lines, e.g. "Let me consider the angle ... But I don't know
  that." repeated). The 8B Megatron run at the same prompt format (2048 cap) has more capped trajectories (238/601,
  183 in turn 1) but only 41 loops. Its caps are mostly long genuine reasoning.
- **Trend:** RL shrinks it. Capped count: 154 (step 0), 142 (20), 132 (40), 66 (64); loops: 114, 99, 107, 45.
- **Where:** eval sampling is greedy (`eval_sampling_params.temperature=0.0`, `repetition_penalty=1.0`). Train
  sampling is temperature 1.0, top_p 1.0, top_k -1, min_p 0, repetition_penalty 1.0. Train rollouts aren't
  dumped, so the train-side loop rate is unknown.
- **Proposed fix:** not a larger cap, which would only buy longer loops. Options: eval at T=0.6-0.7 with n>1 (which
  also addresses #12), or `repetition_penalty` about 1.05 for eval, or a prompt/format change that asks for the tool
  call earlier. Also log per-batch `stop_reason=length` and loop counts for train rollouts.
- **Status:** open (measured; recipe unchanged).

## 49. Qwen3-VL-30B-A3B at TP=2/PP=1/EP=8 OOMs in the first policy_train step on 80 GB H100s
- **Symptom:** `torch.OutOfMemoryError` in `forward_backward_mini_batch` -> Megatron `forward_step` -> the logprob
  computation in `megatron_utils`: "Tried to allocate 5.68 GiB ... 52.68 GiB is allocated by PyTorch", with the
  sleeping vLLM holding about 6 GiB on the same GPU. nvidia-smi peaked at 76-81 GB on all 16 GPUs. The run had
  already passed eval (step 0: 0.594), generate (89 s) and fwd logprobs (50 s). W&B `97wwyvrm`, log
  `/tmp/geo3k_30b_4.log`. The rank-0 shard was 4.67B params.
- **Where:** recipe default `MEGATRON_TP=2 MEGATRON_PP=1` with `micro_train_batch_size_per_gpu=2` and sequences up to
  about 10k tokens (fp32 logits over a 151k vocab).
- **Fix (in this branch):** recipe defaults are now `MEGATRON_TP=4 MEGATRON_PP=2` (DP=2, rank-0 shard 2.15B). This ran
  64 steps with policy_train peaks of 59.9 / 68.4 GiB (PP stage 0 / 1) and no OOM. Untested alternative:
  TP=2 with `micro_train_batch_size_per_gpu=1`, which might avoid the PP bubble (policy_train is 58% of step time).
- **Status:** fixed (recipe).

## 50. `max_ckpts_to_keep=-1` default keeps every checkpoint (4.0 TB for one 30B-A3B run)
- **Symptom:** Run 3 saved 10 Megatron checkpoints (every 10 steps plus each epoch end) at 406 GB each, 4.0 TB
  on shared storage. `cleanup_old_checkpoints` runs every save but deletes nothing.
- **Where:** `config.py:1465` and `sft_config.py:234` default to `max_ckpts_to_keep=-1`. The 30B recipe didn't set it.
- **Fix (in this branch):** the 30B recipe now sets `trainer.max_ckpts_to_keep=2`. The Run 3 checkpoints are still
  in `/mnt/cluster_storage/geo3k/ckpts_30b` and weren't deleted.
- **Status:** mitigated (recipe).

## 51. VLM SFT tokenization cache never saves for real datasets (Arrow offset overflow)
- **Symptom:** `_save_to_cache` logs `Failed to save cache ...: offset overflow while concatenating arrays, consider
  casting input from list<item: list<item: float>> to list<item: large_list<item: float>>`. Every launch then
  re-tokenizes from scratch. It failed for every dataset of 2178 rows or more (ai2d 2434 and 2178, chart2text 8191). The
  256-row eval sets save and reload fine, consistent with the 2 GiB 32-bit Arrow offset limit on `pixel_values`
  stored as nested Python float lists (#38).
- **Where:** `sft_trainer.py:_save_to_cache` (`Dataset.from_list(tokenized)`, about line 290); the tokenized rows carry pixel
  tensors as lists of floats.
- **Cost:** sequential VLM tokenization (#38) at 20-25 ms/row: 49 s for 2434 ai2d rows, 43 s for 2178, 206 s for
  8192 chart2text rows. It's paid on every launch, including restarts.
- **Proposed fix:** store `pixel_values` as bf16/fp32 bytes or `large_list` features (explicit `Features`), or cache
  only token ids plus image references and re-run the image processor lazily in the collator. At minimum, raise the
  log to an error-level metric so the silent re-tokenization is visible.
- **Status:** open.

## 52. Megatron SFT with `num_epochs` crashes in the LR scheduler
- **Symptom:** `megatron_worker.init_model` -> `get_megatron_optimizer_param_scheduler` -> megatron-core
  `OptimizerParamScheduler.__init__`: `assert self.lr_decay_steps > 0` raises `TypeError: '>' not supported between
  instances of 'NoneType' and 'int'`, before step 0.
- **Where:** `sft_trainer.py:979-994` passes `num_training_steps=self.sft_cfg.num_steps`, which is `None` when
  `num_epochs` sets the length (steps are resolved only after the dataloader is built, `:1903`). The comment there
  says the worker falls back to its default, but the explicit `None` overrides `num_training_steps=1e9` in
  `get_megatron_optimizer_param_scheduler`. No shipped recipe hits it today (`run_sft_megatron_apigen_mt.sh` uses
  `num_steps`), but any user who sets `num_epochs` on Megatron without `max_training_steps` does, and no CI covers
  that path. The 30B VLM SFT recipe (`num_epochs=2`) hit it on its first launch.
- **Fix (in this branch):** `get_megatron_optimizer_param_scheduler` maps `None` to `int(1e9)`. Megatron only supports
  `constant_with_warmup`, so this changes nothing numerically. The guard is deliberately not in `sft_trainer`: FSDP passes
  the value to transformers `get_scheduler`, where `None` makes cosine/linear fail loudly, and `1e9` would silently turn
  them constant.
- **Proposed proper fix:** resolve `num_steps = num_epochs * len(train_dataloader)` (capped by `max_training_steps`)
  before `init_model`, so both backends get the real count; add an e2e SFT test with `num_epochs` on Megatron.
- **Footnotes:** `max_training_steps <= num_warmup_steps` (e.g. 5 vs the recipe's 5) fails Megatron's
  `assert self.lr_warmup_steps < self.lr_decay_steps`. `num_steps` and `num_epochs` can't both be set
  (`sft_config.py:103`), so `max_training_steps` is the only way to cap an epoch-based run.
- **Status:** fixed (Megatron guard); proper fix open.

## 53. SFT `ckpt_interval=0` still writes a final checkpoint (406 GB for 30B-A3B)
- **Symptom:** Run 4B, with "checkpoints off" (`CKPT_INTERVAL=0`), wrote `global_step_70` (406 GB) at the end; the
  8-step TP=2 probe's final save took 6 min.
- **Where:** `sft_trainer.py:2133`: the final save runs whenever `ckpt_path` is non-empty, regardless of
  `ckpt_interval`. `CKPT_DIR=""` can't disable it because the recipe's `: "${CKPT_DIR:=...}"` treats empty as unset.
- **Fix:** recipe passes an empty `ckpt_path` when `CKPT_INTERVAL=0` (dcf8104d). Proposed: gate the final save on
  `ckpt_interval > 0` or add an explicit `save_final_ckpt` flag.
- **Status:** mitigated (recipe).

## 54. Tinker SDK drift: server implements the 0.25 API; SDK >= 0.30 requires `sample_sequence_ids` on `asample` futures
- **Symptom:** every client command in `docs/content/docs/tinker/` uses an unpinned `uv run --with tinker`, which resolves
  tinker 0.30.4 (2026-09-27). Its `SamplingClient` asserts on the `/asample` future
  (`tinker/lib/public_interfaces/sampling_client.py:57-74, 320`):
  `AssertionError: asample promises always carry sample_sequence_ids` (non-retryable). So every cookbook recipe that samples dies at
  step 0; for `vlm_classifier` that is the step-0 sampling evaluator, after `create_model` and the first sampler save succeed.
  The server pins `tinker>=0.25.0,<0.26.0` (`pyproject.toml:42`), and nothing in `skyrl/` sets `sample_sequence_ids`.
  With `--with 'tinker>=0.25,<0.26'` the same cookbook commit (5d6e1eb, which declares `tinker>=0.23.0`) runs. Not VLM-specific.
- **Proposed fix:** return `sample_sequence_ids` (one id per returned sequence, i.e. `num_samples` ids) on the `asample`
  `UntypedAPIFuture`, or pin the client in every doc command. Add a CI check that runs one cookbook `sample` against the
  server with the latest SDK.
- **Status:** open (P0 for docs; the pin is a one-line doc fix).

## 55. Megatron + LoRA with the default `merge_lora=true`: sampling a trained model 404s
- **Symptom:** on Megatron colocated with default config, `save_weights_and_get_sampling_client()` then `sample` returns
  400 `Sampling failed for sample 0: ClientResponseError: 404, message='The model `model_xxxxxxxx` does not exist.'` for
  text-only prompts (on `/inference/v1/generate`) and image prompts (on `/v1/chat/completions/render`). Cause:
  `skyrl_train_backend.py:1289-1294` sends the Tinker `model_id` as the vLLM model name whenever `_base_lora_signature` is set,
  and that is set for any LoRA model (`:573`). But under `merge_lora=true` the merged weights are pushed as a full update and
  vLLM only knows the base name (`resolve_policy_model_name`, `inference_servers/utils.py:118-136`). Not VLM-specific.
- **Related:** on the same config, base-model sampling before the first sync returns 400
  `Waking up inference engines failed: inference engines have no synced weights (slept at level 2, which discards them)`.
  FSDP samples the base model fine.
- **Workaround (verified):** `trainer.policy.megatron_config.lora_config.merge_lora=false` (colocated adapter-only sync,
  53b11553). With it, base-model and trained-LoRA sampling both work with images, mixed batches work, and weight sync works.
- **Proposed fix:** use the per-tenant name only when `_adapter_only_sync` is true, else `resolve_policy_model_name`; or
  make `merge_lora=false` the Tinker default for Megatron LoRA. Add a GPU test for sample-after-sync on the default config.
- **Status:** open (P0 for Megatron Tinker LoRA users on defaults).

## 56. `jax` extra doesn't import (flax 0.12.9 vs jax 0.11.2); Qwen3.5 VL would load text-only on JAX
- **Symptom:** `uv run --isolated --extra tinker --extra jax -m skyrl.tinker.api --backend jax ...` starts the API, then the
  engine dies importing `flax.nnx`: `AttributeError: module 'jax.experimental.hijax' has no attribute 'HiPrimitive'`
  (`flax/nnx/variablelib.py:298`). `uv.lock` resolves jax 0.11.2 / jaxlib 0.11.2 / flax 0.12.9. `pyproject.toml:73-74`
  allows `jax>=0.8,<1.0`, `flax>=0.12.2`. The JAX backend is unusable from a clean checkout (seen on an x86 CPU head node; the
  error is at import, so it's not device-specific).
- **VLM part (code reading, cross-ref #26):** Qwen3-VL has no tx model class (loud `ValueError`), but
  `Qwen3_5ForConditionalGeneration = Qwen3_5ForCausalLM` (`skyrl/tx/models/qwen3_5.py:849`). Qwen3.5 VL checkpoints
  therefore load as text-only, and `render_model_input` drops their images silently.
- **Proposed fix:** pin a compatible jax/flax pair in the lock (and a CPU smoke import test in CI); reject image chunks on
  the JAX backend (#26).
- **Status:** open.

## 57. Megatron adapter-only persistent export fails on multi-node: `lora_sync_path` defaults to worker-local `/tmp`
- **Symptom:** Megatron, `merge_lora=false`, head node without GPUs + GPU worker node. `save_weights_for_sampler(name=...)` returns
  400 `[Errno 2] No such file or directory: '/tmp/skyrl_lora_sync/model_xxxxxxxx'`: the worker writes the adapter on the GPU node
  and the engine on the head tars it. Ephemeral sync is unaffected (vLLM, on the GPU node, loads it). Setting
  `trainer.policy.model.lora.lora_sync_path` to shared storage (`/mnt/cluster_storage/...`) fixes it (157 s). Tinker-side
  instance of #2.
- **Proposed fix:** stream the adapter back through Ray (as the FSDP export does) or require/validate a shared
  `lora_sync_path` at startup when the engine and workers are on different nodes; document it in `multi_tenancy.mdx`.
- **Status:** open (workaround: shared `lora_sync_path`).

## 58. Tinker FSDP server-state UX: stale single model after a client crash; cryptic pre-create sample error
- **Symptom:** (a) FSDP supports one LoRA model per server. If a client crashes, its model stays registered until session
  cleanup (`session_timeout_sec=300`), and the next client's `create_model` fails with 400 `NotImplementedError:
  FSDPPolicyWorkerBase.register_adapter is not implemented: multi-tenant LoRA adapter management ... only supported on the
  Megatron backend`. That tells the user nothing about the stale model or the 5-minute wait. (b) A base-model `sample` sent
  before any `create_model` on colocated FSDP returns 400 `'NoneType' object has no attribute 'trainer'`. (c) The raw
  multi-KB Ray worker traceback is returned as the 400 `detail` (e.g. #4's mixed-batch crash).
- **Proposed fix:** (a) name the existing model and the expiry time in the error; (b) return "no model loaded yet; call
  create_model first" (or serve the base model); (c) return the last exception line and log the full traceback server-side.
- **Status:** open (small).

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

## Verification run (2026-09-25)
Short GPU runs on 8xH100 to confirm gaps from the code-reading recheck. W&B project `geometry3k-verify`
(entity `sky-posttraining-uc-berkeley`); logs are `/tmp/geo3k_verify_<n>.log` on the devbox.

| Gap | Verdict | Evidence |
|---|---|---|
| #15 prompt-length filter | **Real but harmless at 1024 for geometry3k.** The real loss is `max_input_length` on later turns | Images add 63-699 tokens the filter doesn't see (expanded max 844), so 0 rows > 1024. But `generator.max_input_length` (= 1024 by default) cuts the retry turn in 27/601 (step 0) to 93/601 (step 90) eval trajectories, all reward 0 |
| #16 `mm_processor_cache_gb=0` | **Safe to enable on vLLM 0.30; no speedup on geometry3k** | Override via `engine_init_kwargs` works; generate 30-32 s/step either way; no errors across 4 pause/resume cycles |
| #14 LoRA on the vision tower | **Confirmed** | 208/600 exported LoRA tensors are `visual.*` and are trained (lora_B nonzero); vLLM logs only an INFO about `enable_tower_connector_lora` and silently skips them; `exclude_modules='.*visual.*'` gives an LM-only adapter and trains fine |
| #13 critic without `pixel_values` | **Hard crash at init, before the predicted silent bug** | `model_wrapper.py:503` `config.hidden_size` raises AttributeError on `Qwen3VLConfig`; a config-validation guard is still needed |
| #12 greedy eval noise | **Measured** | Identical config twice: 88.2% per-question agreement, 0.534 vs 0.526; n=4 at T=0.6: mean 0.459, per-draw std 1.3 pts |
| #23 doubled `<|im_end|>` (new) | **Root cause found** | `RemoteInferenceClient.detokenize` keeps special tokens, so `gen_text` ends in `<|im_end|>` and the template adds a second one; also in vLLM's turn-2 prompt (consistent, off-template). Plus a rarer BPE re-tokenization mismatch (1/3 boundaries) |

Suggested priority: #15 (set `generator.max_input_length` in the recipe, a one-line recipe fix with a direct accuracy impact),
#14 (default-exclude the vision tower for LoRA), #13 (validation guard), #23 (strip the eos in the VLM generator, a 2-line fix).

## Run 3 summary: Qwen3-VL-30B-A3B Megatron 2x8 H100 (2026-09-26, completed)
- **Setup:** 2 GPU nodes x 8 H100 80GB (driver 580.178.04, CUDA 13.0) behind a CPU-only Ray head. vllm 0.30.0,
  torch 2.13.0+cu130, transformers 5.16.1, megatron-bridge 0.7.0+8e7077c6, megatron-core 0.20.0+d476c21b, ray 2.57.0.
  `run_geometry3k_30b_a3b_megatron.sh` at abcff150 (with the #32 unfreeze), with `MEGATRON_TP=4 MEGATRON_PP=2`
  (now the default, #49): EP=8, ETP=1, DP=2, 4 vLLM engines x TP=4, bs=128 x n=8, max 4096 tokens/turn,
  `max_input_length=8192`, full recompute, 4 epochs = 64 steps.
- **Launch:** `WANDB_ENTITY=sky-posttraining-uc-berkeley LOGGER=wandb NUM_NODES=2 MEGATRON_TP=4 MEGATRON_PP=2
  DATA_DIR=/mnt/cluster_storage/geo3k/data EXPORT_PATH=/mnt/cluster_storage/geo3k/exports_30b
  CKPT_PATH=/mnt/cluster_storage/geo3k/ckpts_30b bash examples/train/geometry3k/run_geometry3k_30b_a3b_megatron.sh
  trainer.log_path=/mnt/cluster_storage/geo3k/logs_30b`
- **wandb:** https://wandb.ai/sky-posttraining-uc-berkeley/geometry3k/runs/gleghn3l (the earlier TP=2 OOM attempt is
  `97wwyvrm`).
- **Result:** the Megatron VLM MoE path works end to end: `Qwen3VLMoEBridge` dispatch, the vision tower under EP=8,
  multi-node colocated vLLM with images, NCCL weight sync, and checkpointing. The only code change was the startup
  param-count log. 64 steps completed with exit 0 in 8 h 11 min (11:22 init to 19:33).
- **Trainable params (rank-0 shard, TP=4/PP=2):** language_model=2009.4M/2009.4M, vision_model=138.1M/138.1M
  (the projector sits under vision_model). The #32 unfreeze took effect. No frozen-LM steps were run.
- **Eval pass@1 (601 test, greedy):** 0 **0.577**, 5 0.621, 10 0.641, 15 0.672, 20 0.699, 25 0.697, 30 0.719,
  35 0.702, 40 0.720, 45 0.734, 50 0.772, 55 0.765, 60 **0.807**, 64 0.797 (+22 pts; the 8B Megatron run went from
  0.539 to a 0.729 peak over 96 steps). The TP=2 attempt's step 0 was 0.594; the gap is within #12 greedy noise.
- **Train (16-step epoch means):**

  | epoch | reward | resp len | entropy | logprob abs diff |
  |---|---|---|---|---|
  | 1 | 0.622 | 1970 | 0.330 | 0.0215 |
  | 2 | 0.700 | 1953 | 0.236 | 0.0196 |
  | 3 | 0.726 | 1791 | 0.144 | 0.0173 |
  | 4 | 0.757 | 1693 | 0.087 | 0.0165 |

  Entropy collapses steadily (0.366 at step 1, 0.075 at step 61) without hurting eval over 64 steps. That's worth
  watching on longer runs.
- **Trainer-vs-vLLM logprob abs diff:** 0.0221 at step 1, falling to about 0.0165. That's about 2x the 8B Megatron value (0.0116).
  Candidate causes: MoE routing mismatches between Megatron grouped-GEMM experts and vLLM fused MoE at TP=4, or PP=2.
  It is only logged as a batch aggregate, so no split by turn or length. It falls as entropy falls, consistent with
  fewer near-tie routing/token decisions. Not investigated further.
- **Step time (non-checkpoint steps, about 375 s; 8B: about 95 s):** generate 84-88 s (23%), fwd logprobs about 44 s (12%),
  policy_train about 219 s (58%), sync_weights about 25 s (7%). Eval 162-185 s. Checkpoint save 253-337 s at 406 GB each
  (#6, #50), so checkpoint steps take 624-708 s.
- **Peak GPU memory (nvidia-smi every 5 s, node0 / node1):** overall 79.2 / 79.1 GiB, set during generate
  (colocated vLLM at `gpu_memory_utilization=0.7` plus trainer residue). policy_train 59.9 / 68.4 GiB,
  fwd logprobs 30.7 / 48.1 GiB, sync_weights 76.3 / 76.6 GiB, checkpoint save 57.4 / 56.0 GiB. The trainer has about 11 GiB of
  headroom on the heavier PP stage; the binding constraint is colocated vLLM, not the trainer.
- **#15 check:** 0/601 eval trajectories are cut by `max_input_length` at every eval (steps 0-64), so the 8192 budget
  fixes it. The remaining `stop_reason=length` count is real 4096-token turns: 154, 153, 152, 159, 142, 135, 120, 136, 132,
  118, 88, 86, 69, 66. These are mostly greedy repetition loops in turn 1 (#48).
- **Operational notes:** worker infra logs default to per-node `/tmp/skyrl-logs`, so the new recipe `LOG_PATH`
  env points them at shared storage. Editing or `git reset`-ing the recipe while it runs makes bash
  fail at exit ("unexpected EOF", since bash reads scripts lazily); the training itself is unaffected.

## Run 4 summary: VLM SFT on Megatron (2026-09-26/27; B2 interrupted)
Same cluster and versions as Run 3 (2x8 H100 80GB; vllm 0.30.0, torch 2.13.0+cu130, transformers 5.16.1,
megatron-bridge 0.7.0+8e7077c6). Project `skyrl_sft`, entity `sky-posttraining-uc-berkeley`. The trainable-param log
(`MegatronPolicyWorkerBase.init_model`) fires on the SFT path too. It lands in each node's `/tmp/skyrl-logs` infra log,
not in the driver log.
- **Run A: Qwen3-VL-2B, `run_sft_megatron_vlm.sh`, 1x8 H100, DP=8** (W&B `g9g6nnaz`). The recipe as-is failed
  (`batch_size (4) must be divisible by data-parallel size (8)`), so it was run with `batch_size=8`. The recipe also generated full
  ai2d (2434 rows) instead of a 512-row subset. Both are fixed in 3e8a5f48. Trainable (rank 0): language_model=1720.6M/1720.6M,
  vision_model=407.0M/407.0M. train/loss: step 1 1.099, 10 0.160, 20 0.033, 30 0.043 (about 0 from step 22; ai2d answers are
  1-2 tokens). Step 0.6-1.1 s (step 1 8.7 s). Checkpoints are 28 GB: 35 s, 33 s, then 260 s at step 30 (shared-storage
  contention); HF export 17 s. Peak 26.2 GiB/GPU. Tokenization 49 s for 2434 rows; cache save failed (#51).
- **Run B: Qwen3-VL-30B-A3B, `run_sft_megatron_vlm_30b_a3b.sh`, 2x8, TP=4/PP=2/EP=8/ETP=1 (DP=2), ai2d** (W&B
  `47nu8yqe`). The first launch crashed in the scheduler (#52), fixed. It was then restarted once with `eval_before_train=true`
  (now in the recipe). The `train[:-256]` / `train[-256:]` slices on a local parquet dir work (2178/256). Trainable
  (rank 0): language_model=2009.4M/2009.4M, vision_model=138.1M/138.1M. 70 steps (2 epochs x 35, batch 64).
  **Eval loss:** 0 1.584, 10 0.0122, 20 0.0256, 30 0.0215, 40 0.0273, 50 0.0279, 60 0.0171, 70 0.0322. Train loss is about
  0.0004-0.01 at the end. This is a formatting check only (short answers). Step median 24.1 s (22.8-34.6); eval about 143 s per
  pass. Peak 51.2 / 52.2 GiB. Tokenization 43 s + 5 s. A 406 GB final checkpoint was written despite checkpoints being off (#53).
- **TP=2/PP=1/EP=8 probe (ai2d, 8 steps, `max_training_steps=8`)** (W&B `lsitr9yv`): 11.0-12.5 s/step (step 1 47.9 s),
  about 2.1x faster than TP=4/PP=2, with a peak of 69.3 / 69.7 GiB. SFT doesn't need PP: the RL OOM (#49) was vLLM-bound.
  (`max_training_steps=5` failed the warmup assert, #52 footnote.)
- **B2: 30B-A3B on chart2text (8448 rows = 8192 train + 256 eval), TP=4/PP=2 recipe defaults** (W&B `5wf97d37`),
  run with `ckpt_path=` (#53). **Interrupted at step 80 of 256** (session shutdown, 2026-09-27 00:22). Tokenization:
  206 s for 8192 rows (1 filtered for length); cache save failed (#51). **Eval loss:** 0 0.997, 10 0.765, 20 0.743,
  30 0.746, 40 0.736, 50 0.734, 60 0.736, 70 0.742. Train loss at steps 65/70/75 was 0.725/0.750/0.766, so the train/eval gap
  is about 0 at step 70 (no overfitting within the first epoch; epoch 1 ends at step 128). Most of the gain comes in the first 10 steps; eval
  plateaus at about 0.735 from step 40. Step median 24.7 s.
- **Takeaways:** the Megatron VLM SFT path works for dense 2B and MoE 30B-A3B, with the LM and vision tower fully trained.
  The blockers were a scheduler crash with `num_epochs` (#52) and batch/DP divisibility; the costs are the re-tokenize-every-launch
  cache failure (#51) and unwanted final checkpoints (#53). Open: B2 to completion (eval at 100-256), and B2 at TP=2/PP=1
  to confirm it fits with chart2text lengths.

## Run 5 summary: Tinker VLM (2026-09-27)
Cluster: CPU-only head (API server + engine) + 1x8 H100 80GB worker. Branch `vlm-support-gaps` @ d9bb9008, vllm 0.30.0,
torch 2.13.0+cu130, transformers 5.16.1, megatron-bridge 0.7.0+8e7077c6, server-side tinker 0.25.0. Client: tinker 0.25.0
(0.30.4 fails, #54), tinker-cookbook @ 5d6e1eb. Model: Qwen/Qwen3-VL-8B-Instruct (2B only for the #29 check). Test images: COCO
val2017 000000039769 (two cats on a pink couch, 640x480, 300 image tokens) and 000000000632 (bedroom, 640x483, 300 tokens).
Probe scripts (not in repo): the tinker SDK builds `[text, ImageChunk(jpeg, expected_tokens from the cookbook image processor),
text]` Qwen3-VL chat prompts, with pre-shifted datums weighted on the answer tokens.

Server commands (all with `--checkpoints-base` on `/mnt/cluster_storage`; parallel servers used `--port 800x
--database-url sqlite:////tmp/...`):
- FSDP colocated (doc): `uv run --isolated --extra tinker --extra fsdp -m skyrl.tinker.api --base-model Qwen/Qwen3-VL-8B-Instruct
  --backend fsdp --backend-config '{"trainer.remove_microbatch_padding": false}'`
- FSDP non-colocated: same + `"trainer.placement.colocate_all": false`
- FSDP + processor cache: same as doc + `"generator.inference_engine.engine_init_kwargs.mm_processor_cache_gb": 4`
- Megatron: `--extra megatron --backend megatron`, with `"trainer.remove_microbatch_padding": false`, then also
  `"trainer.policy.megatron_config.lora_config.merge_lora": false` and, for persistent export,
  `"trainer.policy.model.lora.lora_sync_path": "/mnt/cluster_storage/tvlm_lora_sync"`
- Cookbook (working form): `TINKER_API_KEY=tml-dummy uv run --with 'tinker>=0.25,<0.26' --with datasets --with torch python -m
  tinker_cookbook.recipes.vlm_classifier.train base_url=http://localhost:8000 model_name=Qwen/Qwen3-VL-8B-Instruct
  experiment_dir=/tmp/tvlm/vlmcls renderer_name=qwen3_vl_instruct max_steps=40 run_nll_evaluator=false`

| Test | What | Verdict |
|---|---|---|
| 1 | Cookbook `vlm_classifier`, 40 steps (documented path) | **Works after 4 client-command fixes** (#31, #54). Loss moves: see below |
| 2a/2b | Colocated FSDP sample, 1 and 2 images | **Pass**: "Two tabby cats are sleeping side by side on a bright pink couch, with remote controls nearby."; both images described |
| 2c | FSDP forward + forward_backward, 4 image datums | **Pass**: NLL 0.2685 (fwd = fb), wrong answers 9.53, finite, NLL goes to 0 after 3 steps |
| 2d | FSDP mixed 2 image + 2 text in one fb | **Fail**: vision-tower `repeat_interleave` crash (#4/#18 confirmed for Tinker) |
| 3a | Wrong `expected_tokens` on forward_backward | **Clear 400**: `Image 0: expected_tokens=290 but render returned 300 placeholder tokens` |
| 3b | Wrong `expected_tokens` on sample | **Silently succeeds** (#27 confirmed) |
| 4 | Non-colocated FSDP sample with image | **Image silently dropped**: 321 -> 21 prompt tokens, hallucinated answer (#25 confirmed) |
| 5 | Megatron colocated, default `merge_lora=true` | **Trained-LoRA sampling 404s**, text and image (#55); base sample 400 until first sync |
| 5' | Megatron colocated, `merge_lora=false` | **Pass** 2a/2b/2c and **2d** (mixed batch OK, per-row NLL = alone). NLL 0.2841 vs FSDP 0.2685 on the same 4 datums |
| 5 log | Trainable params (rank 0 shard) | Present in the worker infra log: `language_model=73.1M/8263.9M, vision_model=17.7M/594.1M` (LoRA on ViT) |
| 6 | Train, sync, sample with image | **Pass** FSDP (ephemeral + persistent), Megatron `merge_lora=false` (ephemeral; persistent after #57 workaround). Adapter has 232/736 vision tensors (#14); Megatron export lacks processor (#30) |
| 7 | Media transfer | JPEG 173 KB -> `/render` request 231 KB -> response 4.92 MB; the cache setting doesn't produce `kwargs_data=None` (#47) |
| 8 | JAX backend | **Not run**: jax extra import failure (#56) |
| #29 | Server default `remove_microbatch_padding=True` | **Confirmed**: asserts on the first fb, even for text-only datums |

Test 1 numbers (LoRA r32, lr 5e-4 cosine, batch 32, 1 GPU, `max_steps=40`): train NLL 3.408 (step 0), 1.448 (1), 0.953 (2),
0.137 (5), 0.114 (10), 0.036 (15), 0.020 (19), 0.012-0.058 (25-39). Sampling-eval accuracy (128 test images): **0.8% at step 0
-> 81.3% at step 20**. Median step 29.8 s (48.8 s step 0). Server: `create_model` 108 s, first `save_weights_for_sampler` 174 s
(later 9-52 s), 128 image samples about 4 s to submit. Server-side render + batch conversion is about 1% of engine wall time (py-spy,
about 0.3 s per 32-image step); 98% is the FSDP worker's `forward_backward`. The skipped NLL evaluator takes 952 s for the full
caltech101 test split (about 6.1k images, 6.4 images/s) on this 1-GPU server.

Other timings (8B, 1 GPU per role): FSDP forward on 4 image datums 14.9 s cold, fb 3.7-3.9 s warm; Megatron forward 31.0 s cold, fb
3.3-5.4 s warm; first sampler sync FSDP 174-187 s, Megatron 154 s; Megatron persistent adapter export 157 s; `create_model`
FSDP 108 s, Megatron 137 s.

Takeaways: the colocated FSDP path works end to end for single-type (all-image) batches, and the cookbook VLM recipe trains
once its client command is fixed. The silent failures are non-colocated sampling (#25), `expected_tokens` on sample (#27), and
Qwen3.5 on JAX (#26/#56). The hard failures are mixed batches on FSDP (#4), Megatron LoRA sampling on defaults (#55), the
unpinned SDK (#54), the `remove_microbatch_padding` default (#29), and multi-node adapter export (#57). Megatron with
`merge_lora=false` is currently the most complete Tinker VLM configuration, apart from the missing processor in its export (#30).
