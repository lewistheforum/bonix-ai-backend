# Medicare AI Backend

FastAPI backend application for Medicare AI services including package setup, clinic recommendations, and chatbot functionality.

## Project Structure

```
medicare-ai-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application setup
│   ├── config.py               # Configuration settings
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── package_setup.py        # Package setup API routes
│   │       ├── recommendation_clinic.py # Clinic recommendation API routes
│   │       └── chatbot.py              # Chatbot API routes
│   ├── dto/
│   │   ├── __init__.py
│   │   ├── package_setup_dto.py        # Package setup DTOs
│   │   ├── recommendation_clinic_dto.py # Clinic recommendation DTOs
│   │   └── chatbot_dto.py              # Chatbot DTOs
│   ├── services/
│   │   ├── __init__.py
│   │   ├── package_setup_service.py    # Package setup business logic
│   │   ├── recommendation_clinic_service.py # Clinic recommendation logic
│   │   └── chatbot_service.py          # Chatbot business logic
│   ├── models/
│   │   └── __init__.py                 # Database models (if needed)
│   └── utils/
│       ├── __init__.py
│       └── logger.py                   # Logging utility
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables example
└── README.md                 # This file
```

## Architecture

The application follows a layered architecture:

- **Controllers (Routers)**: Handle HTTP requests and responses (`app/api/v1/`)
- **Services**: Contain business logic (`app/services/`)
- **DTOs**: Data Transfer Objects for request/response validation (`app/dto/`)
- **Models**: Database models (if using ORM) (`app/models/`)
- **Config**: Application configuration (`app/config.py`)
- **Utils**: Utility functions (`app/utils/`)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy environment file:
```bash
cp .env.example .env
```

4. Update `.env` with your configuration

## Running the Application

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --reload
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### Package Setup API (`/api/v1/package-setup`)
- `POST /` - Create a new package
- `GET /{package_id}` - Get package by ID
- `GET /` - List all packages (with pagination)
- `PUT /{package_id}` - Update a package
- `DELETE /{package_id}` - Delete a package

### Recommendation Clinic API (`/api/v1/recommendation-clinic`)
- `POST /recommend` - Get clinic recommendations based on symptoms
- `GET /health` - Health check

### Chatbot API (`/api/v1/chatbot`)
- `POST /chat` - Send message to chatbot
- `GET /conversation/{conversation_id}` - Get conversation history
- `DELETE /conversation/{conversation_id}` - Delete conversation
- `GET /health` - Health check

## Development

### Adding New APIs

1. Create DTOs in `app/dto/`
2. Create service in `app/services/`
3. Create router in `app/api/v1/`
4. Register router in `app/main.py`

### Example: Adding a new API

1. **Create DTO** (`app/dto/new_feature_dto.py`):
```python
from pydantic import BaseModel

class NewFeatureRequest(BaseModel):
    field: str
```

2. **Create Service** (`app/services/new_feature_service.py`):
```python
from app.dto.new_feature_dto import NewFeatureRequest

class NewFeatureService:
    async def process(self, request: NewFeatureRequest):
        # Business logic here
        pass
```

3. **Create Router** (`app/api/v1/new_feature.py`):
```python
from fastapi import APIRouter
from app.dto.new_feature_dto import NewFeatureRequest
from app.services.new_feature_service import NewFeatureService

router = APIRouter()
service = NewFeatureService()

@router.post("")
async def endpoint(request: NewFeatureRequest):
    return await service.process(request)
```

4. **Register in main.py**:
```python
from app.api.v1 import new_feature

app.include_router(
    new_feature.router,
    prefix=f"{settings.API_V1_PREFIX}/new-feature",
    tags=["New Feature"]
)
```

## Testing

You can test the APIs using:
- Swagger UI: http://localhost:8000/docs
- Postman or any HTTP client
- curl commands

Example:
```bash
curl -X POST "http://localhost:8000/api/v1/chatbot/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a headache"}'
```

## License

MIT

# medicare-ai-backend
# bonix-ai-backend
