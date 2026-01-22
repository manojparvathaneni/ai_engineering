"""
LLM Playground Backend
======================

A production-style FastAPI backend demonstrating:
- Session management (conversation history)
- Configuration management (system prompts, defaults)
- Simple API key authentication
- Rate limiting
- Proper error handling

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import time
import httpx
import os
from contextlib import asynccontextmanager

# =============================================================================
# Configuration & Storage (In production: use Redis/PostgreSQL)
# =============================================================================

# In-memory storage for demo purposes
sessions_store: Dict[str, dict] = {}
configs_store: Dict[str, dict] = {}
api_keys_store: Dict[str, dict] = {
    "demo-key-12345": {"user_id": "demo_user", "tier": "free", "rate_limit": 10}
}

# Default configuration
DEFAULT_CONFIG = {
    "system_prompt": "You are a helpful AI assistant. Be concise and clear.",
    "default_temperature": 0.7,
    "default_top_p": 0.9,
    "default_top_k": 50,
    "default_max_tokens": 150,
    "model": "claude-sonnet-4-20250514"
}

# Rate limiting tracker
rate_limit_tracker: Dict[str, List[float]] = {}


# =============================================================================
# Pydantic Models (Request/Response schemas)
# =============================================================================

class GenerateRequest(BaseModel):
    """Request body for text generation."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    system_prompt: Optional[str] = None  # Override for this request


class GenerateResponse(BaseModel):
    """Response from text generation."""
    text: str
    session_id: str
    usage: dict
    generation_time_ms: int
    parameters_used: dict


class SessionCreate(BaseModel):
    """Request to create a new session."""
    name: Optional[str] = "New Chat"
    system_prompt: Optional[str] = None


class SessionResponse(BaseModel):
    """Session information."""
    session_id: str
    name: str
    created_at: str
    message_count: int
    system_prompt: Optional[str]


class Message(BaseModel):
    """A single message in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str


class ConfigUpdate(BaseModel):
    """Configuration update request."""
    system_prompt: Optional[str] = None
    default_temperature: Optional[float] = Field(None, ge=0, le=2)
    default_top_p: Optional[float] = Field(None, ge=0, le=1)
    default_top_k: Optional[int] = Field(None, ge=1, le=100)
    default_max_tokens: Optional[int] = Field(None, ge=1, le=4096)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


# =============================================================================
# Authentication & Rate Limiting
# =============================================================================

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> dict:
    """
    Verify API key and return user info.
    
    In production: validate against database, check expiration, etc.
    """
    # Allow requests without key for demo (uses demo key)
    if not x_api_key:
        x_api_key = "demo-key-12345"
    
    if x_api_key not in api_keys_store:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user_info = api_keys_store[x_api_key]
    
    # Check rate limit
    user_id = user_info["user_id"]
    current_time = time.time()
    window_start = current_time - 60  # 1-minute window
    
    if user_id not in rate_limit_tracker:
        rate_limit_tracker[user_id] = []
    
    # Clean old entries
    rate_limit_tracker[user_id] = [
        t for t in rate_limit_tracker[user_id] if t > window_start
    ]
    
    if len(rate_limit_tracker[user_id]) >= user_info["rate_limit"]:
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Max {user_info['rate_limit']} requests/minute."
        )
    
    rate_limit_tracker[user_id].append(current_time)
    
    return user_info


# =============================================================================
# FastAPI App Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 LLM Playground Backend starting...")
    print("📝 Demo API key: demo-key-12345")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="LLM Playground API",
    description="Backend API for the LLM Playground teaching tool",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/info")
async def get_info():
    """Get API information and available endpoints."""
    return {
        "name": "LLM Playground API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /api/generate - Generate text",
            "sessions": {
                "create": "POST /api/sessions - Create new session",
                "list": "GET /api/sessions - List all sessions",
                "get": "GET /api/sessions/{id} - Get session with history",
                "delete": "DELETE /api/sessions/{id} - Delete session"
            },
            "config": {
                "get": "GET /api/config - Get current configuration",
                "update": "PUT /api/config - Update configuration"
            }
        },
        "authentication": "Pass API key in X-API-Key header (optional for demo)"
    }


# =============================================================================
# Generation Endpoint
# =============================================================================

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    user: dict = Depends(verify_api_key)
):
    """
    Generate text using Claude API.
    
    Supports:
    - Session-based conversation (maintains history)
    - Parameter overrides (temperature, top_p, etc.)
    - Custom system prompts
    """
    start_time = time.time()
    
    # Get or create session
    if request.session_id and request.session_id in sessions_store:
        session = sessions_store[request.session_id]
    else:
        # Create new session
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "name": "Quick Chat",
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
            "system_prompt": request.system_prompt or DEFAULT_CONFIG["system_prompt"],
            "user_id": user["user_id"]
        }
        sessions_store[session_id] = session
        request.session_id = session_id
    
    # Get user config or defaults
    user_config = configs_store.get(user["user_id"], DEFAULT_CONFIG)
    
    # Build parameters (request overrides > user config > defaults)
    params = {
        "temperature": request.temperature or user_config.get("default_temperature", 0.7),
        "top_p": request.top_p or user_config.get("default_top_p", 0.9),
        "top_k": request.top_k or user_config.get("default_top_k", 50),
        "max_tokens": request.max_tokens or user_config.get("default_max_tokens", 150)
    }
    
    # Build messages with history
    messages = []
    for msg in session["messages"][-10:]:  # Last 10 messages for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": request.prompt})
    
    # Get system prompt
    system_prompt = request.system_prompt or session.get("system_prompt") or DEFAULT_CONFIG["system_prompt"]
    
    # Call Claude API
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": user_config.get("model", DEFAULT_CONFIG["model"]),
                    "max_tokens": params["max_tokens"],
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                    "top_k": params["top_k"],
                    "system": system_prompt,
                    "messages": messages
                }
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("error", {}).get("message", "Unknown error")
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            data = response.json()
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Claude API timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to connect to Claude API: {str(e)}")
    
    # Extract response
    generated_text = data["content"][0]["text"]
    usage = data.get("usage", {})
    
    # Save to session history
    timestamp = datetime.utcnow().isoformat()
    session["messages"].append({
        "role": "user",
        "content": request.prompt,
        "timestamp": timestamp
    })
    session["messages"].append({
        "role": "assistant", 
        "content": generated_text,
        "timestamp": timestamp
    })
    
    generation_time = int((time.time() - start_time) * 1000)
    
    return GenerateResponse(
        text=generated_text,
        session_id=session["id"],
        usage=usage,
        generation_time_ms=generation_time,
        parameters_used=params
    )


# =============================================================================
# Session Management Endpoints
# =============================================================================

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreate,
    user: dict = Depends(verify_api_key)
):
    """Create a new conversation session."""
    session_id = str(uuid.uuid4())
    
    session = {
        "id": session_id,
        "name": request.name or "New Chat",
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
        "system_prompt": request.system_prompt or DEFAULT_CONFIG["system_prompt"],
        "user_id": user["user_id"]
    }
    
    sessions_store[session_id] = session
    
    return SessionResponse(
        session_id=session_id,
        name=session["name"],
        created_at=session["created_at"],
        message_count=0,
        system_prompt=session["system_prompt"]
    )


@app.get("/api/sessions", response_model=List[SessionResponse])
async def list_sessions(user: dict = Depends(verify_api_key)):
    """List all sessions for the authenticated user."""
    user_sessions = [
        SessionResponse(
            session_id=s["id"],
            name=s["name"],
            created_at=s["created_at"],
            message_count=len(s["messages"]),
            system_prompt=s.get("system_prompt")
        )
        for s in sessions_store.values()
        if s.get("user_id") == user["user_id"]
    ]
    
    # Sort by created_at descending
    user_sessions.sort(key=lambda x: x.created_at, reverse=True)
    
    return user_sessions


@app.get("/api/sessions/{session_id}")
async def get_session(
    session_id: str,
    user: dict = Depends(verify_api_key)
):
    """Get a session with full conversation history."""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[session_id]
    
    if session.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "session_id": session["id"],
        "name": session["name"],
        "created_at": session["created_at"],
        "system_prompt": session.get("system_prompt"),
        "messages": session["messages"]
    }


@app.put("/api/sessions/{session_id}")
async def update_session(
    session_id: str,
    name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user: dict = Depends(verify_api_key)
):
    """Update session name or system prompt."""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[session_id]
    
    if session.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if name:
        session["name"] = name
    if system_prompt:
        session["system_prompt"] = system_prompt
    
    return {"status": "updated", "session_id": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user: dict = Depends(verify_api_key)
):
    """Delete a session."""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[session_id]
    
    if session.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    del sessions_store[session_id]
    
    return {"status": "deleted", "session_id": session_id}


# =============================================================================
# Configuration Endpoints
# =============================================================================

@app.get("/api/config")
async def get_config(user: dict = Depends(verify_api_key)):
    """Get user's configuration (or defaults if not set)."""
    user_config = configs_store.get(user["user_id"], DEFAULT_CONFIG.copy())
    
    return {
        "config": user_config,
        "available_models": [
            {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "description": "Fast and capable"},
            {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "description": "Fastest responses"}
        ],
        "parameter_ranges": {
            "temperature": {"min": 0, "max": 2, "default": 0.7},
            "top_p": {"min": 0, "max": 1, "default": 0.9},
            "top_k": {"min": 1, "max": 100, "default": 50},
            "max_tokens": {"min": 1, "max": 4096, "default": 150}
        }
    }


@app.put("/api/config")
async def update_config(
    config: ConfigUpdate,
    user: dict = Depends(verify_api_key)
):
    """Update user's configuration."""
    user_id = user["user_id"]
    
    if user_id not in configs_store:
        configs_store[user_id] = DEFAULT_CONFIG.copy()
    
    # Update only provided fields
    if config.system_prompt is not None:
        configs_store[user_id]["system_prompt"] = config.system_prompt
    if config.default_temperature is not None:
        configs_store[user_id]["default_temperature"] = config.default_temperature
    if config.default_top_p is not None:
        configs_store[user_id]["default_top_p"] = config.default_top_p
    if config.default_top_k is not None:
        configs_store[user_id]["default_top_k"] = config.default_top_k
    if config.default_max_tokens is not None:
        configs_store[user_id]["default_max_tokens"] = config.default_max_tokens
    
    return {
        "status": "updated",
        "config": configs_store[user_id]
    }


@app.post("/api/config/reset")
async def reset_config(user: dict = Depends(verify_api_key)):
    """Reset user's configuration to defaults."""
    user_id = user["user_id"]
    configs_store[user_id] = DEFAULT_CONFIG.copy()
    
    return {
        "status": "reset",
        "config": configs_store[user_id]
    }


# =============================================================================
# System Prompt Templates (Bonus!)
# =============================================================================

PROMPT_TEMPLATES = {
    "helpful": {
        "name": "Helpful Assistant",
        "prompt": "You are a helpful AI assistant. Be concise, clear, and friendly."
    },
    "teacher": {
        "name": "Patient Teacher",
        "prompt": "You are a patient teacher who explains concepts step by step. Use analogies and examples. Check for understanding."
    },
    "coder": {
        "name": "Code Assistant",
        "prompt": "You are a coding assistant. Provide clean, well-commented code. Explain your approach briefly. Prefer simple solutions."
    },
    "creative": {
        "name": "Creative Writer",
        "prompt": "You are a creative writer with a vivid imagination. Use descriptive language and engaging narratives."
    },
    "analyst": {
        "name": "Data Analyst",
        "prompt": "You are a data analyst. Be precise, use numbers when possible, and clearly state assumptions and limitations."
    }
}


@app.get("/api/templates")
async def get_templates():
    """Get available system prompt templates."""
    return {
        "templates": [
            {"id": k, **v} for k, v in PROMPT_TEMPLATES.items()
        ]
    }


# =============================================================================
# Run with: uvicorn main:app --reload --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
