from __future__ import annotations

import argparse
import os
from typing import Any

import datasets
from PIL import Image

from examples.computer_use_rl.vnc_env import KvmVncEnv
from examples.computer_use_rl.vnc_lock import vnc_global_lock


def _make_prompt(task: str) -> list[dict[str, Any]]:
    # RLHFDataset in verl expects <image> placeholders when images are present.
    # It will split message content by "<image>" and provide the corresponding image(s).
    return [
        {
            "role": "user",
            "content": (
                "You are a computer-use agent. You will be given a screenshot.\n"
                "Screenshot: <image>\n\n"
                f"Task: {task}\n\n"
                "Output EXACTLY ONE action as a single JSON object (and nothing else).\n"
                "Allowed actions:\n"
                "- {\"type\": \"move\", \"dx\": int, \"dy\": int}\n"
                "- {\"type\": \"click_left\"}\n"
                "- {\"type\": \"key\", \"key\": string}\n"
                "- {\"type\": \"done\"}\n"
                "Do not output any other text."
            ),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny VNC screenshot parquet dataset for GRPO")
    parser.add_argument("--vnc-host", default="127.0.0.1")
    parser.add_argument("--vnc-port", type=int, default=5901, help="display :1 -> 5901")
    parser.add_argument("--out-dir", default=os.path.expanduser("~/data/computer_use_dummy"))
    parser.add_argument("--num-train", type=int, default=4)
    parser.add_argument("--num-val", type=int, default=1)
    parser.add_argument("--task", default="Do nothing.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with vnc_global_lock(host=args.vnc_host, port=args.vnc_port):
        env = KvmVncEnv(
            host=args.vnc_host,
            port=args.vnc_port,
            resize_hw=(512, 512),
            shutdown_reactor_on_close=True,
        )
        try:
            obs = env.reset()  # uint8 HxWx3
        finally:
            env.close()

    screenshot = Image.fromarray(obs, mode="RGB")

    def make_row(split: str, idx: int) -> dict[str, Any]:
        return {
            "data_source": "computer_use_dummy",
            "prompt": _make_prompt(args.task),
            "images": [screenshot],
            "ability": "computer_use",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"split": split, "index": idx, "task": args.task},
        }

    train_rows = [make_row("train", i) for i in range(args.num_train)]
    val_rows = [make_row("val", i) for i in range(args.num_val)]

    features = datasets.Features(
        {
            "data_source": datasets.Value("string"),
            "prompt": datasets.Sequence(
                {
                    "role": datasets.Value("string"),
                    "content": datasets.Value("string"),
                }
            ),
            "images": datasets.Sequence(datasets.Image()),
            "ability": datasets.Value("string"),
            "reward_model": {"style": datasets.Value("string"), "ground_truth": datasets.Value("string")},
            "extra_info": datasets.Value("string"),
        }
    )

    # HuggingFace datasets can handle nested dicts but parquet+features can be finicky.
    # Keep extra_info as a string to stay robust.
    for row in train_rows:
        row["extra_info"] = str(row["extra_info"])
    for row in val_rows:
        row["extra_info"] = str(row["extra_info"])

    train_ds = datasets.Dataset.from_list(train_rows)
    val_ds = datasets.Dataset.from_list(val_rows)

    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print("wrote", train_path)
    print("wrote", val_path)


if __name__ == "__main__":
    main()
