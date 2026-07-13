from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ModelRegistrationRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=100)
    source: ModelSource
    model_name_or_path: str = Field(min_length=1)
    revision: str | None = None
    trust_remote_code: bool = False
    device_map: str | None = "auto"
    local_files_only: bool = False


class ModelRecord(BaseModel):
    model_id: str
    source: ModelSource
    model_name_or_path: str
    revision: str | None = None
    trust_remote_code: bool = False
    device_map: str | None = "auto"
    local_files_only: bool = False
    status: str = "registered"
    last_error: str | None = None


@dataclass
class LoadedModelArtifacts:
    model: Any
    tokenizer: Any


class ModelRegistry:
    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}
        self._loaded_models: dict[str, LoadedModelArtifacts] = {}
        self._lock = Lock()

    def register(self, request: ModelRegistrationRequest) -> ModelRecord:
        record = ModelRecord(**request.model_dump(), status="registered", last_error=None)
        with self._lock:
            self._records[record.model_id] = record
        return record

    def list(self) -> list[ModelRecord]:
        with self._lock:
            return list(self._records.values())

    def get(self, model_id: str) -> ModelRecord | None:
        with self._lock:
            return self._records.get(model_id)

    def load(self, model_id: str) -> ModelRecord:
        with self._lock:
            record = self._records.get(model_id)

        if not record:
            raise KeyError(f"Model '{model_id}' is not registered.")

        loaded = self._load_model(record)

        with self._lock:
            self._loaded_models[model_id] = loaded
            updated_record = record.model_copy(update={"status": "loaded", "last_error": None})
            self._records[model_id] = updated_record
            return updated_record

    def get_loaded_model(self, model_id: str) -> LoadedModelArtifacts | None:
        with self._lock:
            return self._loaded_models.get(model_id)

    def _load_model(self, record: ModelRecord) -> LoadedModelArtifacts:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "transformers is not installed. Install dependencies from backend/requirements.txt first."
            ) from error

        model_path = record.model_name_or_path

        if record.source == ModelSource.LOCAL:
            resolved_path = Path(model_path).expanduser().resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"Local model path does not exist: '{resolved_path}'"
                )
            model_path = str(resolved_path)

        local_only = record.local_files_only or record.source == ModelSource.LOCAL

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=record.revision,
            trust_remote_code=record.trust_remote_code,
            local_files_only=local_only,
        )

        model_kwargs: dict[str, Any] = {
            "revision": record.revision,
            "trust_remote_code": record.trust_remote_code,
            "local_files_only": local_only,
        }
        if record.device_map:
            model_kwargs["device_map"] = record.device_map

        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        return LoadedModelArtifacts(model=model, tokenizer=tokenizer)
