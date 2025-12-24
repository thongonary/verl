#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}

# 1) Create a tiny dataset from the current VNC screen (4 train rows / 1 val row)
python -m examples.computer_use_rl.create_vnc_dataset \
  --vnc-host 127.0.0.1 \
  --vnc-port 5901 \
  --out-dir "$HOME/data/computer_use_dummy" \
  --num-train 4 \
  --num-val 1 \
  --task "(dummy) Move cursor / click / press enter"

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

# 2) Run GRPO with Qwen2.5-VL-3B on 4 GPUs
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/computer_use_dummy/train.parquet \
  data.val_files=$HOME/data/computer_use_dummy/test.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=1024 \
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
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.n=2 \
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
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=999999 \
  trainer.test_freq=1 \
  $@
