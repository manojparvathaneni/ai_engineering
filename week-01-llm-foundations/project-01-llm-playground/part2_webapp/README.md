# LLM Playground v2

A full-stack LLM playground demonstrating production architecture patterns. Built as the Week 1 project for the LLM Foundations course.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                            │
│   • Light, modern UI          • Session management sidebar          │
│   • Real-time chat            • Settings configuration              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST API
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ /api/generate│  │ /api/sessions│  │ /api/config  │              │
│  │              │  │              │  │              │              │
│  │ • Text gen   │  │ • Create     │  │ • Get/Set    │              │
│  │ • History    │  │ • List       │  │ • Templates  │              │
│  │ • Params     │  │ • Delete     │  │ • Defaults   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                              │                                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              Service Layer                          │           │
│  │  • API Key Auth    • Rate Limiting    • Logging     │           │
│  └─────────────────────────────────────────────────────┘           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │  Claude API     │
                      └─────────────────┘
```

## Features

### Backend (FastAPI)
- **Text Generation** - Full parameter control (temperature, top_p, top_k, max_tokens)
- **Session Management** - Create, list, retrieve, delete conversations
- **Configuration** - User-specific defaults, system prompts
- **Authentication** - Simple API key auth (extensible to OAuth/JWT)
- **Rate Limiting** - Per-user request throttling
- **Templates** - Pre-built system prompt templates

### Frontend (React)
- **Modern Light UI** - Clean, professional design
- **Session Sidebar** - Easy conversation management
- **Settings Panel** - Configure defaults and system prompts
- **Real-time Stats** - Token counts, generation time
- **Keyboard Shortcuts** - Enter to send, Shift+Enter for newline

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# Run the server
uvicorn main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

**API Documentation:** Visit `http://localhost:8000/docs` for interactive Swagger UI

### 2. Frontend Setup

The frontend is a React component. You can use it in several ways:

#### Option A: Copy to Existing React Project
Copy `frontend/LLMPlayground.jsx` to your React project and import it.

#### Option B: Use with Vite (Quick Setup)
```bash
# Create new Vite project
npm create vite@latest llm-playground-frontend -- --template react
cd llm-playground-frontend

# Copy the component
cp ../frontend/LLMPlayground.jsx src/

# Update App.jsx to use the component
# Then run:
npm install
npm run dev
```

#### Option C: View as Artifact
The component works directly in Claude's artifact viewer!

### 3. Configuration

The frontend's Setup screen lets you configure:
- **Backend URL** - Where your FastAPI server runs
- **API Key** - Use `demo-key-12345` for testing

## API Reference

### Generate Text
```bash
POST /api/generate
{
  "prompt": "Hello, how are you?",
  "session_id": "optional-session-id",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "max_tokens": 150
}
```

### Sessions
```bash
GET  /api/sessions              # List all sessions
POST /api/sessions              # Create new session
GET  /api/sessions/{id}         # Get session with history
PUT  /api/sessions/{id}         # Update session name/prompt
DELETE /api/sessions/{id}       # Delete session
```

### Configuration
```bash
GET  /api/config                # Get user config
PUT  /api/config                # Update config
POST /api/config/reset          # Reset to defaults
GET  /api/templates             # Get prompt templates
```

## Teaching Applications

This project demonstrates several key concepts:

### System Design Patterns
| Pattern | Implementation |
|---------|---------------|
| Separation of Concerns | Frontend ↔ Backend ↔ LLM API |
| Authentication | API key validation middleware |
| Rate Limiting | Per-user request throttling |
| Session Management | Conversation history storage |
| Configuration | User-specific defaults |

### LLM Concepts
| Concept | Where to See It |
|---------|-----------------|
| Temperature | Settings panel slider |
| Top-p Sampling | Generation parameters |
| Top-k Sampling | Generation parameters |
| System Prompts | Configuration & templates |
| Context Window | Session history truncation |

## Extending the Project

### Add Database Persistence
Replace in-memory stores with SQLite/PostgreSQL:
```python
# Use SQLAlchemy or similar
from sqlalchemy import create_engine
# ... implement database models
```

### Add OAuth Authentication
```python
# Use authlib or similar
from authlib.integrations.starlette_client import OAuth
oauth = OAuth()
oauth.register('google', ...)
```

### Add RAG (Week 2 Preview)
```python
# Add document retrieval before generation
@app.post("/api/generate")
async def generate_with_rag(request):
    # 1. Retrieve relevant documents
    docs = await retrieve_documents(request.prompt)
    # 2. Enhance prompt with context
    enhanced_prompt = f"Context: {docs}\n\nQuestion: {request.prompt}"
    # 3. Generate with context
    ...
```

## Files

```
llm-playground-v2/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── LLMPlayground.jsx    # React component
└── README.md                # This file
```

## Troubleshooting

### CORS Errors
The backend allows all origins by default. For production, update:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    ...
)
```

### Rate Limit Exceeded
Default is 10 requests/minute. Adjust in `api_keys_store`:
```python
api_keys_store = {
    "your-key": {"user_id": "user", "tier": "pro", "rate_limit": 100}
}
```

### Connection Refused
Make sure:
1. Backend is running (`uvicorn main:app --reload`)
2. Correct port in frontend config (default: 8000)
3. `ANTHROPIC_API_KEY` environment variable is set

---

**Week 1 Complete!** 🎉

This project integrates everything from the course:
- Pre-training concepts (tokenization in stats display)
- Post-training concepts (system prompts, generation parameters)
- System design (safety, context management, architecture)
- Evaluation awareness (token counts, timing metrics)
