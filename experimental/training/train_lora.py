from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: str | Path, adapter_id: str | None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if adapter_id is None or row.get("adapter_id") == adapter_id:
                rows.append(row)
    if not rows:
        raise ValueError(f"No training rows found in {path} for adapter_id={adapter_id!r}")
    return rows


def train_lora(
    base_model: str,
    train_file: str | Path,
    adapter_id: str | None,
    output_dir: str | Path,
    max_steps: int = 80,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_length: int = 1024,
    gradient_accumulation_steps: int = 4,
) -> Path:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional training path
        raise RuntimeError(
            "LoRA training requires optional dependencies. "
            "Install with `uv sync --extra real` or "
            "`pip install torch transformers peft accelerate`."
        ) from exc

    rows = _load_rows(train_file, adapter_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class SFTDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            row = rows[index % len(rows)]
            prompt = str(row["prompt"]).rstrip() + "\n"
            completion = str(row["completion"]).strip() + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + completion_ids)[-max_length:]
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            padding = max_length - len(input_ids)
            return {
                "input_ids": torch.tensor(input_ids + [tokenizer.pad_token_id] * padding),
                "attention_mask": torch.tensor([1] * len(input_ids) + [0] * padding),
                "labels": torch.tensor(labels + [-100] * padding),
            }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.train()
    loader = DataLoader(SFTDataset(), batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / gradient_accumulation_steps
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if step >= max_steps:
                break
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    metadata = {
        "base_model": base_model,
        "adapter_id": adapter_id or "multitask",
        "train_file": str(train_file),
        "rows": len(rows),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--adapter-id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    print(
        train_lora(
            args.base_model,
            args.train_file,
            args.adapter_id,
            args.output_dir,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    )


if __name__ == "__main__":
    main()
