from __future__ import annotations

from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from models.model_loader import ModelRegistry
from prompts.prompt_history import PromptHistoryStore


class PromptExecutionRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    do_sample: bool = True


class PromptExecutionResult(BaseModel):
    model_id: str
    prompt: str
    response: str
    latency_ms: float


class PromptRunner:
    def __init__(
        self,
        model_registry: ModelRegistry,
        history_store: PromptHistoryStore | None = None,
    ) -> None:
        self._model_registry = model_registry
        self._history_store = history_store

    def execute(self, request: PromptExecutionRequest) -> PromptExecutionResult:
        loaded = self._model_registry.get_loaded_model(request.model_id)
        if not loaded:
            raise ValueError(
                f"Model '{request.model_id}' is not loaded. Load the model before executing prompts."
            )

        model = loaded.model
        tokenizer = loaded.tokenizer

        if not hasattr(model, "generate"):
            raise ValueError(
                "Loaded model does not support text generation. Register/load a causal language model."
            )

        encoded = tokenizer(request.prompt, return_tensors="pt")
        encoded = self._move_to_model_device(encoded, model)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "pad_token_id": tokenizer.eos_token_id,
        }

        start_time = perf_counter()

        try:
            import torch

            with torch.no_grad():
                generated_ids = model.generate(**encoded, **generation_kwargs)
        except ImportError as error:
            raise RuntimeError(
                "torch is not installed. Install dependencies from backend/requirements.txt first."
            ) from error

        latency_ms = (perf_counter() - start_time) * 1000

        prompt_length = encoded["input_ids"].shape[1]
        response_ids = generated_ids[0][prompt_length:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        if self._history_store is not None:
            self._history_store.add_entry(
                model=request.model_id,
                prompt=request.prompt,
                response=response_text,
            )

        return PromptExecutionResult(
            model_id=request.model_id,
            prompt=request.prompt,
            response=response_text,
            latency_ms=round(latency_ms, 2),
        )

    @staticmethod
    def _move_to_model_device(encoded_inputs: dict[str, Any], model: Any) -> dict[str, Any]:
        try:
            model_device = model.device
        except AttributeError:
            model_device = next(model.parameters()).device

        return {key: value.to(model_device) for key, value in encoded_inputs.items()}
