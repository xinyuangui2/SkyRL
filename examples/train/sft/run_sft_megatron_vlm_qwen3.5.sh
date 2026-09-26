set -x

# VLM SFT for Qwen3.5 on the Megatron backend, 1 or 2 nodes.
#
# Qwen3.5 checkpoints are unified vision-language models (Qwen3_5ForConditionalGeneration with a
# vision_config), so the SFT trainer takes the VLM path automatically. Unlike the Qwen3.5 RL
# recipes, this does NOT set language_model_only: the vision tower is loaded and trained, via
# Megatron-Bridge's Qwen35VLModelProvider.
#
# Defaults: Qwen3.5-9B (dense) on 1 node x 8 GPUs, DP=8.
# 2-node MoE example:
#   NUM_NODES=2 MODEL_NAME=Qwen/Qwen3.5-35B-A3B MEGATRON_TP=2 MEGATRON_EP=8 \
#     bash examples/train/sft/run_sft_megatron_vlm_qwen3.5.sh
# 2-node dense example:
#   NUM_NODES=2 MODEL_NAME=Qwen/Qwen3.5-27B MEGATRON_TP=4 MEGATRON_PP=2 \
#     bash examples/train/sft/run_sft_megatron_vlm_qwen3.5.sh
#
# The freeze_* overrides below are explicit on purpose: Megatron-Bridge's Qwen3-VL MoE provider
# freezes the language model and vision tower by default, see
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/8e7077c6826d17eb4d4d54e6eb15c5a581eda4c0/src/megatron/bridge/models/qwen_vl/qwen3_vl_provider.py#L272-L274
# The Qwen3.5 providers default to False, but setting them keeps the recipe safe across model swaps.
#
# VLM SFT constraints (enforced by the trainer): no sequence packing / microbatch padding removal,
# no context or sequence parallelism, last-assistant-message loss only, every sample must carry images.
# Data: chat `messages` with {"type": "image", "image": <data-uri>} parts; the_cauldron prep below.
#
# uv run examples/train/sft/prepare_cauldron_vlm.py --output-dir $HOME/data/cauldron-vlm
# bash examples/train/sft/run_sft_megatron_vlm_qwen3.5.sh

: "${DATA_DIR:="$HOME/data/cauldron-vlm"}"
: "${MODEL_NAME:="Qwen/Qwen3.5-9B"}"
: "${NUM_NODES:=1}"
: "${NUM_GPUS:=8}"
: "${MEGATRON_TP:=1}"
: "${MEGATRON_PP:=1}"
: "${MEGATRON_EP:=1}"
: "${MEGATRON_ETP:=1}"
: "${RECOMPUTE_GRANULARITY:="full"}"
: "${RECOMPUTE_METHOD:="uniform"}"
: "${RECOMPUTE_NUM_LAYERS:=1}"
: "${BATCH_SIZE:=32}"
: "${MICRO_BATCH_SIZE:=1}"
: "${MAX_LENGTH:=4096}"
: "${NUM_STEPS:=200}"
: "${LR:=2e-6}"
: "${LOGGER:=wandb}"
: "${CKPT_DIR:="$HOME/ckpts/skyrl_sft_megatron_vlm_qwen3.5"}"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== Generating the_cauldron VLM SFT dataset ==="
  uv run examples/train/sft/prepare_cauldron_vlm.py --output-dir "$DATA_DIR"
fi

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path="$MODEL_NAME" \
    train_datasets="['$DATA_DIR']" \
    train_dataset_splits="['train']" \
    messages_key=messages \
    max_length=$MAX_LENGTH \
    num_steps=$NUM_STEPS \
    batch_size=$BATCH_SIZE \
    micro_train_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    remove_microbatch_padding=false \
    train_on_what=last_assistant_message \
    seed=42 \
    optimizer_config.lr=$LR \
    optimizer_config.weight_decay=1e-2 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=10 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=$NUM_NODES \
    placement.num_gpus_per_node=$NUM_GPUS \
    megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
    megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
    megatron_config.context_parallel_size=1 \
    megatron_config.expert_model_parallel_size=$MEGATRON_EP \
    megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
    megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
    megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
    megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
    megatron_config.transformer_config_kwargs.freeze_language_model=false \
    megatron_config.transformer_config_kwargs.freeze_vision_model=false \
    logger=$LOGGER \
    project_name=skyrl_sft \
    run_name="skyrl_sft_megatron_vlm_$(basename $MODEL_NAME)_n${NUM_NODES}" \
    ckpt_path="$CKPT_DIR" \
    ckpt_interval=50 \
    hf_save_interval=100 \
    resume_from="" \
    "$@"
