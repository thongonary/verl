#!/usr/bin/env bash
set -euo pipefail
set -x

# Optional first positional arg selects the rollout engine (default: vllm).
# We must shift it out so it is not forwarded to Hydra as an override.
if [[ $# -gt 0 ]]; then
  ENGINE="$1"
  shift
else
  ENGINE=vllm
fi

# Optional: make first-iteration debugging much faster.
# - disables torch.compile in actor/ref FSDP
# - disables vLLM CUDA graph capture (enforce eager)
# Usage:
#   FAST_DEBUG=1 bash ./examples/grpo_trainer/run_qwen2_5_vl-3b_computer_use.sh
FAST_DEBUG=${FAST_DEBUG:-1}

# Default to dumping trajectories into $HOME/trl_dumps/<run_id>/... so reruns are easy to inspect.
# You can always override these by setting env vars before running the script.
VERL_TRAJECTORY_DUMP_DIR=${VERL_TRAJECTORY_DUMP_DIR:-$HOME/trl_dumps}
VERL_RUN_ID=${VERL_RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}
export VERL_TRAJECTORY_DUMP_DIR VERL_RUN_ID

# Optional: if VERL_TRAJECTORY_DUMP_DIR is set, create a per-run subdirectory so runs don't mix.
# The agent loop will still create per-rollout subdirs underneath.
if [[ -n "${VERL_TRAJECTORY_DUMP_DIR:-}" ]]; then
  RUN_ID=${VERL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
  export VERL_TRAJECTORY_DUMP_DIR="${VERL_TRAJECTORY_DUMP_DIR%/}/${RUN_ID}"
  echo "[computer_use] VERL_TRAJECTORY_DUMP_DIR=${VERL_TRAJECTORY_DUMP_DIR}"
fi


# Optional: dump per-rollout trajectories (screenshots + actions) for debugging.
# Example:
#   VERL_TRAJECTORY_DUMP_DIR=$HOME/trl_dumps bash ./examples/grpo_trainer/run_qwen2_5_vl-3b_computer_use.sh

# 1) Create a tiny dataset from the current VNC screen (4 train rows / 1 val row)
python -m examples.computer_use_rl.create_vnc_dataset \
  --vnc-host 127.0.0.1 \
  --vnc-port 5901 \
  --out-dir "$HOME/data/computer_use_dummy" \
  --num-train 4 \
  --num-val 1 \
  --task "(dummy) Move cursor / click / press enter"

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

EXTRA_ARGS=()
if [[ "$FAST_DEBUG" == "1" ]]; then
  # A belt-and-suspenders global kill switch: even if some component still tries
  # to enable torch.compile, TorchDynamo will be disabled.
  export TORCHDYNAMO_DISABLE=1

  EXTRA_ARGS+=(
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
  )
  echo "[computer_use] FAST_DEBUG=1 (enforce_eager + disable torch.compile)"
fi

# 2) Run GRPO with Qwen2.5-VL-3B on 4 GPUs
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/computer_use_dummy/train.parquet \
  data.val_files=$HOME/data/computer_use_dummy/test.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.image_key=images \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=512 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=vnc_single_action_agent \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  custom_reward_function.path=$PROJECT_DIR/examples/computer_use_rl/computer_use_reward.py \
  custom_reward_function.name=compute_score \
  +custom_reward_function.reward_kwargs.vnc_host=127.0.0.1 \
  +custom_reward_function.reward_kwargs.vnc_port=5901 \
  +custom_reward_function.reward_kwargs.max_actions=8 \
  +custom_reward_function.reward_kwargs.dummy_reward=1.0 \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name='computer_use_grpo' \
  trainer.experiment_name='qwen2_5_vl_3b_vnc_dummy' \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=999999 \
  trainer.test_freq=1 \
  "${EXTRA_ARGS[@]}" \
  $@
