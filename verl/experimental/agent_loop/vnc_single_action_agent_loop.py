import ast
import copy
import json
import logging
import os
from typing import Any
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_action_dict(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a single JSON-ish object from model text."""
    if not text:
        return None

    s = text.strip()

    # Remove common wrappers
    if s.startswith("```"):
        # strip fenced blocks
        s = s.strip("`")
        s = s.replace("json\n", "", 1).strip()

    # Fast path: whole string is JSON
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Extract the first {...} span
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = s[start : end + 1]
        try:
            parsed = json.loads(blob)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            try:
                parsed = ast.literal_eval(blob)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

    # Last resort: python literal eval
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


@register("vnc_single_action_agent")
class VncSingleActionAgentLoop(AgentLoopBase):
    """Closed-loop VNC agent: generate exactly one action, execute it, add new screenshot, repeat."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        reward_kwargs = (getattr(self.config, "custom_reward_function", None) or {}).get("reward_kwargs", {})
        self.vnc_host = reward_kwargs.get("vnc_host", "127.0.0.1")
        self.vnc_port = int(reward_kwargs.get("vnc_port", 5901))
        self.max_actions = int(reward_kwargs.get("max_actions", 8))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Import lazily so training without VNC deps still works.
        from examples.computer_use_rl.vnc_env import KvmVncEnv, VncAction
        from examples.computer_use_rl.vnc_lock import vnc_global_lock

        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex

        env = KvmVncEnv(
            host=self.vnc_host,
            port=self.vnc_port,
            resize_hw=(512, 512),
            shutdown_reactor_on_close=True,
        )

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        prompt_ids: list[int] = []

        try:
            # Serialize all interaction with the single shared VM.
            with vnc_global_lock(host=self.vnc_host, port=self.vnc_port):
                # Sync the first screenshot with the live env.
                obs0 = env.reset()
                img0 = Image.fromarray(obs0, mode="RGB")

                if image_data is None:
                    image_data = [img0]
                elif isinstance(image_data, list) and len(image_data) > 0:
                    # Replace the last image from dataset with the latest env view.
                    image_data = list(image_data)
                    image_data[-1] = img0
                else:
                    image_data = [img0]

                # Build initial prompt token ids from the full message history.
                if self.processor is not None:
                    raw_prompt = await self.loop.run_in_executor(
                        None,
                        lambda: self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False,
                            **self.apply_chat_template_kwargs,
                        ),
                    )
                    model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
                    prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                else:
                    prompt_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            **self.apply_chat_template_kwargs,
                        ),
                    )

                # Rollout loop
                num_actions = 0
                while True:
                    if num_actions >= self.max_actions:
                        break
                    if len(response_mask) >= self.response_length:
                        break

                    # Rebuild prompt_ids from the full conversation each turn.
                    # This avoids concatenating multiple independently-templated segments (which repeats
                    # system/BOS tokens and can break Qwen2-VL RoPE indexing).
                    if self.processor is not None:
                        raw_prompt = await self.loop.run_in_executor(
                            None,
                            lambda: self.processor.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=False,
                                **self.apply_chat_template_kwargs,
                            ),
                        )
                        model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
                        prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                    else:
                        prompt_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=True,
                                **self.apply_chat_template_kwargs,
                            ),
                        )

                    with simple_timer("generate_sequences", metrics):
                        output = await self.server_manager.generate(
                            request_id=request_id,
                            prompt_ids=prompt_ids,
                            sampling_params=sampling_params,
                            image_data=image_data,
                        )

                    generated_ids = output.token_ids
                    response_ids += generated_ids
                    response_mask += [1] * len(generated_ids)
                    if output.log_probs:
                        response_logprobs += output.log_probs

                    # Parse one action from the model output
                    decoded = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    )
                    action_dict = _extract_action_dict(decoded)
                    if not action_dict:
                        # Can't parse -> terminate to avoid spinning
                        break

                    action_type = action_dict.get("type") or action_dict.get("kind")
                    if isinstance(action_type, str):
                        action_type = action_type.strip().lower()

                    if action_type in ("done", "finish", "stop"):
                        break

                    # Execute env step
                    if action_type == "move":
                        dx = int(action_dict.get("dx", 0))
                        dy = int(action_dict.get("dy", 0))
                        vnc_action = VncAction(kind="move", dx=dx, dy=dy)
                    elif action_type == "click_left":
                        vnc_action = VncAction(kind="click_left")
                    elif action_type == "key":
                        key = action_dict.get("key")
                        if not isinstance(key, str) or not key:
                            break
                        vnc_action = VncAction(kind="key", key=key)
                    else:
                        # Unknown action -> terminate
                        break

                    obs = env.step(vnc_action)
                    new_img = Image.fromarray(obs, mode="RGB")

                    # Append user observation turn with a new image placeholder
                    obs_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Observation screenshot: "},
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": "\nReturn exactly ONE action as a JSON object. If the task is complete, return {\"type\": \"done\"}.",
                            },
                        ],
                    }
                    messages.append(obs_message)
                    if image_data is None:
                        image_data = [new_img]
                    elif not isinstance(image_data, list):
                        image_data = [image_data, new_img]
                    else:
                        image_data.append(new_img)

                    num_actions += 1

        finally:
            env.close()

        multi_modal_out = {"image": image_data} if image_data is not None else {}

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_out,
            num_turns=2 + len(messages),
            metrics=metrics,
        )
