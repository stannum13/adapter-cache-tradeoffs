from __future__ import annotations

import time
from typing import Any

from adapter_cache_bench.backends.base import Backend
from adapter_cache_bench.bench.quality import evaluate_prediction
from adapter_cache_bench.cache.cache_models import CacheModel
from adapter_cache_bench.config import BackendConfig
from adapter_cache_bench.types import (
    BackendResponse,
    RequestMetrics,
    RequestRecord,
    RoutingDecision,
)


class TransformersBackend(Backend):
    """CPU-friendly Hugging Face causal LM backend for real model-output smoke runs."""

    def __init__(self, config: BackendConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional deps
            raise RuntimeError(
                "TransformersBackend requires optional dependencies. "
                "Run with `uv sync --extra real` or `uv run --extra real ...`."
            ) from exc

        self.config = config
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            local_files_only=bool(config.extra_body.get("local_files_only", False)),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_name = str(config.extra_body.get("torch_dtype", "auto"))
        dtype = getattr(torch, dtype_name, None) if dtype_name != "auto" else "auto"
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            local_files_only=bool(config.extra_body.get("local_files_only", False)),
            **kwargs,
        )
        self.device = str(config.extra_body.get("device", "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self, request: RequestRecord, decision: RoutingDecision, cache_model: CacheModel
    ) -> BackendResponse:
        cached = cache_model.estimate_cached_prefix_tokens(
            decision.adapter_id,
            request.prompt,
            request.tenant_id,
            request.trust_group_id,
        )
        max_input_tokens = int(self.config.extra_body.get("max_input_tokens", 1024))
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        max_new_tokens = int(self.config.extra_body.get("max_new_tokens", request.max_tokens))
        max_new_tokens = max(1, min(max_new_tokens, request.max_tokens))
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.temperature > 0:
            generate_kwargs["temperature"] = self.config.temperature
        started = time.perf_counter()
        with self.torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        generated_ids = output_ids[0, prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        output_tokens = max(1, int(generated_ids.shape[-1]))
        cached = min(cached, prompt_tokens)
        metrics = RequestMetrics(
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached,
            uncached_prompt_tokens=max(0, prompt_tokens - cached),
            prefill_ms=0.0,
            decode_ms=elapsed_ms,
            queue_ms=0.0,
            ttft_ms=elapsed_ms,
            itl_ms=elapsed_ms / output_tokens,
            tpot_ms=elapsed_ms / output_tokens,
            e2e_ms=elapsed_ms,
            output_tokens=output_tokens,
        )
        quality = evaluate_prediction(
            task_type=request.task_type,
            adapter_id=decision.adapter_id,
            prediction=text,
            ground_truth=request.ground_truth,
        )
        cache_model.observe_request(
            decision.adapter_id,
            request.prompt,
            request.tenant_id,
            request.trust_group_id,
        )
        return BackendResponse(
            request_id=request.request_id,
            adapter_id=decision.adapter_id,
            text=text,
            metrics=metrics,
            quality=quality,
        )
