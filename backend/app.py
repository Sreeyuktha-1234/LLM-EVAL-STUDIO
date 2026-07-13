from fastapi import FastAPI, HTTPException

from config import settings
from models.model_loader import ModelRegistrationRequest, ModelRegistry

app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)
model_registry = ModelRegistry()


@app.get("/")
def read_root() -> dict:
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get(f"{settings.api_v1_prefix}/info")
def get_app_info() -> dict:
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug": settings.debug,
    }


@app.post(f"{settings.api_v1_prefix}/models/register")
def register_model(request: ModelRegistrationRequest) -> dict:
    record = model_registry.register(request)
    return {
        "message": "Model registered successfully.",
        "model": record.model_dump(),
    }


@app.get(f"{settings.api_v1_prefix}/models")
def list_models() -> dict:
    records = model_registry.list()
    return {"models": [record.model_dump() for record in records]}


@app.get(f"{settings.api_v1_prefix}/models/{{model_id}}")
def get_model(model_id: str) -> dict:
    record = model_registry.get(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found.")
    return {"model": record.model_dump()}


@app.post(f"{settings.api_v1_prefix}/models/{{model_id}}/load")
def load_model(model_id: str) -> dict:
    try:
        record = model_registry.load(model_id)
        return {
            "message": "Model loaded successfully.",
            "model": record.model_dump(),
        }
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected model loading error: {error}",
        ) from error
