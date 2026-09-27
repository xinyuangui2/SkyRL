set -x

# VLM SFT for Qwen3-VL-30B-A3B (MoE) on the Megatron backend, 2 nodes x 8 H100 80GB.
#
# Larger-model counterpart of run_sft_megatron_vlm.sh (Qwen3-VL-2B, 4 GPUs). Parallelism mirrors the
# validated RL recipe for the same model (examples/train/geometry3k/run_geometry3k_30b_a3b_megatron.sh):
# TP=4, PP=2, CP=1, EP=8, ETP=1 (DP=2), full activation recompute. TP=2/PP=1 OOMs on 80GB H100s for RL;
# SFT has no colocated vLLM so it may fit, but start from the layout that is known to run.
#
# Megatron-Bridge's Qwen3VLMoEModelProvider defaults freeze_language_model=True and
# freeze_vision_model=True (VLM_GAPS.md #32), so the freeze_* overrides below are required; without
# them only the vision projector trains. See
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/8e7077c6826d17eb4d4d54e6eb15c5a581eda4c0/src/megatron/bridge/models/qwen_vl/qwen3_vl_provider.py#L272-L274
#
# Data: the_cauldron `chart2text` subset (27k rows, ~50-word free-form chart descriptions), capped at NUM_ROWS,
# converted to chat `messages` with {"type": "image", "image": <data-uri>} parts; the last EVAL_ROWS rows are
# held out for eval loss. chart2text is used instead of `ai2d` because ai2d answers are one or two tokens, so
# last-assistant-turn loss collapses to ~0 within 20 steps and measures formatting, not learning. Other
# long-answer subsets: vistext (10k), localized_narratives (200k), mimic_cgd (71k, 2 images per row).
# VLM SFT constraints (enforced by the trainer): no sequence packing / microbatch padding removal,
# no context or sequence parallelism, last-assistant-message loss only, every sample must carry images.
#
# uv run examples/train/sft/prepare_cauldron_vlm.py --output-dir $HOME/data/cauldron-vlm-chart2text --config chart2text --num-rows 8448
# NUM_NODES=2 bash examples/train/sft/run_sft_megatron_vlm_30b_a3b.sh

: "${CAULDRON_CONFIG:="chart2text"}"
: "${DATA_DIR:="$HOME/data/cauldron-vlm-$CAULDRON_CONFIG"}"
# 8192 train + 256 eval rows: 2 epochs = 256 steps at batch 64.
: "${NUM_ROWS:=8448}"
: "${EVAL_ROWS:=256}"
: "${MODEL_NAME:="Qwen/Qwen3-VL-30B-A3B-Instruct"}"
: "${NUM_NODES:=2}"
: "${NUM_GPUS:=8}"
: "${MEGATRON_TP:=4}"
: "${MEGATRON_PP:=2}"
: "${MEGATRON_CP:=1}"
: "${MEGATRON_EP:=8}"
: "${MEGATRON_ETP:=1}"
: "${RECOMPUTE_GRANULARITY:="full"}"
: "${RECOMPUTE_METHOD:="uniform"}"
: "${RECOMPUTE_NUM_LAYERS:=1}"
: "${BATCH_SIZE:=64}"
: "${MICRO_BATCH_SIZE:=1}"
: "${MAX_LENGTH:=4096}"
: "${NUM_EPOCHS:=2}"
: "${LR:=5e-6}"
: "${EVAL_INTERVAL:=10}"
: "${LOGGER:=wandb}"
: "${CKPT_DIR:="$HOME/ckpts/skyrl_sft_megatron_vlm_30b_a3b"}"
# Megatron checkpoints for this model are about 406 GB each; off by default.
: "${CKPT_INTERVAL:=0}"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== Generating the_cauldron ($CAULDRON_CONFIG) VLM SFT dataset ==="
  uv run examples/train/sft/prepare_cauldron_vlm.py --output-dir "$DATA_DIR" --config "$CAULDRON_CONFIG" ${NUM_ROWS:+--num-rows "$NUM_ROWS"}
fi

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path="$MODEL_NAME" \
    train_datasets="['$DATA_DIR']" \
    train_dataset_splits="['train[:-$EVAL_ROWS]']" \
    eval_datasets="['$DATA_DIR']" \
    eval_dataset_splits="['train[-$EVAL_ROWS:]']" \
    eval_interval=$EVAL_INTERVAL \
    messages_key=messages \
    max_length=$MAX_LENGTH \
    num_epochs=$NUM_EPOCHS \
    batch_size=$BATCH_SIZE \
    micro_train_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    remove_microbatch_padding=false \
    train_on_what=last_assistant_message \
    seed=42 \
    optimizer_config.lr=$LR \
    optimizer_config.weight_decay=1e-2 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=5 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=$NUM_NODES \
    placement.num_gpus_per_node=$NUM_GPUS \
    megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
    megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
    megatron_config.context_parallel_size=$MEGATRON_CP \
    megatron_config.expert_model_parallel_size=$MEGATRON_EP \
    megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
    megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
    megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
    megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
    megatron_config.transformer_config_kwargs.freeze_language_model=false \
    megatron_config.transformer_config_kwargs.freeze_vision_model=false \
    logger=$LOGGER \
    project_name=skyrl_sft \
    run_name="skyrl_sft_megatron_vlm_30b_a3b_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}" \
    ckpt_path="$CKPT_DIR" \
    ckpt_interval=$CKPT_INTERVAL \
    max_ckpts_to_keep=1 \
    hf_save_interval=0 \
    resume_from="" \
    "$@"
