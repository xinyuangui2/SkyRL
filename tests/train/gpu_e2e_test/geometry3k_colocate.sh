set -euo pipefail

# End-to-end VLM RL smoke test: multi-turn GRPO on Geometry-3K with Qwen3-VL-2B-Instruct, FSDP,
# colocated vLLM, on the 4-GPU l4_ci compute config.
#
# Runs the stock geometry3k recipe on a 512-example train subset (8 steps at batch 64) with
# eval_before_train, then asserts wandb summary metrics via get_summary.py.
#
# Thresholds: initial values, deliberately loose. Recalibrate to 5% allowance from the min/max of
# the last ~10 nightly runs once they exist (same procedure as gsm8k_colocate.sh).

RUN_NAME="run_$(date +%Y%m%d%H%M%S)_$$"
PROJECT_NAME="geometry3k_ci"
SCRIPT_DIR=$(dirname $(realpath $0))
DATA_DIR="${DATA_DIR:-$HOME/data/geometry_3k_ci}"

# Qwen3-VL-2B-Instruct greedy pass@1 on the geometry3k test split before training; the 8B model
# scores 0.53. A 2B model that has lost its vision path scores near 0.
EVAL_ACC_MIN_VALUE=0.20
TRAIN_ACC_MIN_VALUE=0.15
# Mean generated tokens per trajectory (3 turns x up to 1024). Guards against degenerate
# max-length generations.
NUM_TOKENS_MAX_VALUE=2500
LOGPROBS_DIFF_MAX_VALUE=0.05

uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir "$DATA_DIR" --max_train_samples 512

NUM_GPUS=4 DATA_DIR="$DATA_DIR" bash examples/train/geometry3k/run_geometry3k.sh \
  trainer.policy.model.path="Qwen/Qwen3-VL-2B-Instruct" \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.eval_interval=8 \
  trainer.eval_batch_size=128 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=100 \
  generator.sampling_params.max_generate_length=1024 \
  generator.max_input_length=4096 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  trainer.logger=wandb \
  trainer.project_name=\"$PROJECT_NAME\" \
  trainer.run_name=\"$RUN_NAME\" \
  trainer.export_path="$HOME/exports/geometry3k_ci" \
  trainer.ckpt_path="$HOME/ckpts/geometry3k_ci"

uv run --isolated --extra fsdp $SCRIPT_DIR/get_summary.py --run_name $RUN_NAME --project_name $PROJECT_NAME --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" "generate/avg_num_tokens <= $NUM_TOKENS_MAX_VALUE" "policy/rollout_train_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"
