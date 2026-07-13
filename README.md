# LLM-EVAL-STUDIO

Minimal starter repository for an LLM evaluation studio project.

## Project Structure

```text
LLM-EVAL-STUDIO/
├── backend/
│   ├── app.py
│   ├── config.py
│   └── requirements.txt
├── frontend/
├── README.md
└── .env.example
```

## Backend (FastAPI)

The backend is built with FastAPI and reads configuration from environment variables.
It also includes a model registry that can register and load:

- HuggingFace hub models
- Local fine-tuned models

### 1. Create and activate virtual environment

Windows (PowerShell):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

From the project root, create your `.env` file from `.env.example`:

```powershell
cd ..
Copy-Item .env.example .env
```

You can edit `.env` values as needed.

### 4. Run the API

From `backend/`:

```powershell
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 5. Test endpoints

- API root: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- Info endpoint: `http://127.0.0.1:8000/api/v1/info`
- Swagger docs: `http://127.0.0.1:8000/docs`

### 6. Model registry endpoints

Register a HuggingFace model:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/models/register" -ContentType "application/json" -Body '{
	"model_id": "bert-base",
	"source": "huggingface",
	"model_name_or_path": "bert-base-uncased",
	"revision": "main",
	"trust_remote_code": false,
	"device_map": "auto"
}'
```

Register a local fine-tuned model:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/models/register" -ContentType "application/json" -Body '{
	"model_id": "my-local-model",
	"source": "local",
	"model_name_or_path": "C:/models/my-finetuned-model",
	"local_files_only": true,
	"trust_remote_code": false,
	"device_map": "auto"
}'
```

List registered models:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/models"
```

Load a registered model:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/models/bert-base/load"
```

Get a single model record:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/models/bert-base"
```

## Environment Variables

Defined in `.env.example`:

- `APP_NAME`: Application name shown in API metadata
- `APP_VERSION`: API version string
- `DEBUG`: FastAPI debug flag (`true`/`false`)
- `API_V1_PREFIX`: Prefix for versioned API routes
- `HOST`: Host value for local run configuration
- `PORT`: Port value for local run configuration

## Frontend

`frontend/` is currently a placeholder for your UI implementation.

## Next Steps

1. Add frontend framework setup (React/Vue/Next.js, etc.).
2. Add API routers under `backend` as the project grows.
3. Add tests and CI workflow.