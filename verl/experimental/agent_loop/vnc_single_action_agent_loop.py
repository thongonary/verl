import ast
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _safe_filename(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)[:180]


def _maybe_get_cfg_str(config: Any, key_path: str) -> str | None:
    """Best-effort OmegaConf-style path lookup without hard dependency."""
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.select(config, key_path)
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
    except Exception:
        return None


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


def _extract_text_from_message(message: Any) -> str | None:
    """Extract concatenated text from a chat message.

    Supports HF-style messages where `content` can be a string or a list of
    multimodal items like {"type": "text", "text": "..."}.
    """
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        s = content.strip()
        return s if s else None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
        s = "".join(parts).strip()
        return s if s else None
    return None


def _extract_initial_instruction(messages: list[Any]) -> str | None:
    """Best-effort extraction of the original user instruction/task."""
    if not messages:
        return None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _extract_text_from_message(msg)
            if text:
                return text
    return None


def _extract_thinking(decoded_text: str) -> dict[str, Any] | None:
    """Extract explicit model 'thinking' blocks if present.

    This does not attempt to infer hidden chain-of-thought; it only captures
    text that the model explicitly emitted (e.g., <think>...</think> or a
    ```think fenced block).
    """
    if not decoded_text or not isinstance(decoded_text, str):
        return None

    s = decoded_text
    # Common explicit tags
    if "<think>" in s and "</think>" in s:
        start = s.find("<think>") + len("<think>")
        end = s.find("</think>", start)
        if end > start:
            think_text = s[start:end].strip()
            # Remove the thinking block from the visible output
            remainder = (s[: s.find("<think>")] + s[end + len("</think>") :]).strip()
            return {
                "present": True,
                "format": "tag",
                "thinking_text": think_text,
                "output_without_thinking": remainder,
            }

    # Fenced block ```think ... ```
    fence = "```"
    idx = s.find(fence)
    while idx != -1:
        end_fence = s.find(fence, idx + len(fence))
        if end_fence == -1:
            break
        header = s[idx + len(fence) :].lstrip()
        if header.startswith("think"):
            body_start = idx + len(fence) + header.find("think") + len("think")
            # Skip optional newline
            if body_start < len(s) and s[body_start] == "\n":
                body_start += 1
            think_text = s[body_start:end_fence].strip()
            remainder = (s[:idx] + s[end_fence + len(fence) :]).strip()
            return {
                "present": True,
                "format": "fence",
                "thinking_text": think_text,
                "output_without_thinking": remainder,
            }
        idx = s.find(fence, end_fence + len(fence))

    # No explicit thinking
    return {"present": False}


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

        # Optional trajectory dumping for debugging.
        # Enable by setting VERL_TRAJECTORY_DUMP_DIR or hydra override:
        #   +actor_rollout_ref.rollout.agent.trajectory_dump_dir=/path
        self.trajectory_dump_dir = (
            os.getenv("VERL_TRAJECTORY_DUMP_DIR")
            or _maybe_get_cfg_str(self.config, "actor_rollout_ref.rollout.agent.trajectory_dump_dir")
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Import lazily so training without VNC deps still works.
        from examples.computer_use_rl.vnc_env import KvmVncEnv, VncAction
        from examples.computer_use_rl.vnc_lock import vnc_global_lock

        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))

        original_instruction = _extract_initial_instruction(messages)

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex

        trajectory = kwargs.get("trajectory") or {}
        global_step = trajectory.get("step", -1)
        sample_index = trajectory.get("sample_index", -1)
        rollout_n = trajectory.get("rollout_n", -1)
        validate = bool(trajectory.get("validate", False))

        dump_root: Path | None = None
        if self.trajectory_dump_dir:
            # Each rollout gets its own directory to avoid collisions across workers.
            exp_name = _maybe_get_cfg_str(self.config, "trainer.experiment_name") or "experiment"
            run_name = f"step_{global_step}_sample_{sample_index}_rollout_{rollout_n}_req_{request_id}"
            run_name = _safe_filename(run_name)
            dump_root = Path(self.trajectory_dump_dir).expanduser().resolve() / _safe_filename(exp_name) / run_name
            dump_root.mkdir(parents=True, exist_ok=True)

        trace_events: list[dict[str, Any]] = []

        env = KvmVncEnv(
            host=self.vnc_host,
            port=self.vnc_port,
            resize_hw=(512, 512),
            shutdown_reactor_on_close=False,
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

                if dump_root is not None:
                    img0.save(dump_root / "screen_000_reset.png")

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

                    if dump_root is not None:
                        (dump_root / f"model_output_{num_actions:03d}.txt").write_text(decoded, encoding="utf-8")

                    thinking = _extract_thinking(decoded)

                    # Prefer parsing actions from output without explicit thinking blocks (if present)
                    action_parse_text = (
                        thinking.get("output_without_thinking")
                        if isinstance(thinking, dict) and thinking.get("present") and thinking.get("output_without_thinking")
                        else decoded
                    )
                    action_dict = _extract_action_dict(action_parse_text)
                    if not action_dict:
                        trace_events.append(
                            {
                                "turn": num_actions,
                                "parsed": False,
                                "decoded": decoded,
                                "thinking": thinking,
                                "reason": "parse_failed",
                            }
                        )
                        # Can't parse -> terminate to avoid spinning
                        break

                    action_type = action_dict.get("type") or action_dict.get("kind")
                    if isinstance(action_type, str):
                        action_type = action_type.strip().lower()

                    if action_type in ("done", "finish", "stop"):
                        trace_events.append(
                            {
                                "turn": num_actions,
                                "parsed": True,
                                "action": action_dict,
                                "action_type": action_type,
                                "thinking": thinking,
                                "terminated": True,
                            }
                        )
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
                        trace_events.append(
                            {
                                "turn": num_actions,
                                "parsed": True,
                                "action": action_dict,
                                "action_type": action_type,
                                "thinking": thinking,
                                "reason": "unknown_action",
                            }
                        )
                        break

                    trace_events.append(
                        {
                            "turn": num_actions,
                            "parsed": True,
                            "action": action_dict,
                            "action_type": action_type,
                            "thinking": thinking,
                        }
                    )

                    obs = env.step(vnc_action)
                    new_img = Image.fromarray(obs, mode="RGB")

                    if dump_root is not None:
                        new_img.save(dump_root / f"screen_{num_actions + 1:03d}_after.png")

                    # IMPORTANT: do NOT keep appending image placeholders each turn.
                    # For Qwen2-VL, each image adds a large number of tokens; appending one image
                    # per step quickly exceeds vLLM max_model_len and crashes with negative max_tokens.
                    # Instead, replace the latest screenshot image in-place.
                    if image_data is None:
                        image_data = [new_img]
                    elif not isinstance(image_data, list):
                        image_data = [new_img]
                    elif len(image_data) == 0:
                        image_data = [new_img]
                    else:
                        image_data[-1] = new_img

                    num_actions += 1

        finally:
            env.close()

        multi_modal_out = {"image": image_data} if image_data is not None else {}

        if dump_root is not None:
            summary = {
                "trajectory": {
                    "global_step": global_step,
                    "sample_index": sample_index,
                    "rollout_n": rollout_n,
                    "validate": validate,
                    "request_id": request_id,
                },
                "original_instruction": original_instruction,
                "vnc": {"host": self.vnc_host, "port": self.vnc_port, "max_actions": self.max_actions},
                "events": trace_events,
                "metrics": metrics,
                "num_messages": len(messages),
                "final_num_images": len(image_data) if isinstance(image_data, list) else (1 if image_data else 0),
                "messages": messages,
            }
            (dump_root / "trajectory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_out,
            # Keep this roughly consistent as "steps" for metrics. Messages no longer grow per step.
            num_turns=2 + num_actions,
            metrics=metrics,
        )
