"""
FastAPI application for the conversational AI system
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .graph_definition import ConversationGraph
from .persistence import ChatPersistence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KisanAI - Conversational AI for Agriculture",
    description="Speech-to-text conversational agent with RAG and multilingual support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
conversation_graph = ConversationGraph()
persistence = ChatPersistence()

# Pydantic models
class SessionCreate(BaseModel):
    session_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message_count: int

class MessageResponse(BaseModel):
    role: str
    text: str
    language: Optional[str] = None
    timestamp: str

class ConversationResponse(BaseModel):
    session_id: str
    language: str
    user_text: str
    ai_text: str
    context_docs: List[Dict[str, Any]]
    timestamp: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "KisanAI - Conversational AI for Agriculture", "status": "healthy"}

@app.post("/start_session", response_model=SessionResponse)
async def start_session(session_data: SessionCreate = None):
    """Start a new conversation session"""
    try:
        session_id = session_data.session_id if session_data and session_data.session_id else str(uuid.uuid4())
        
        # Create session in persistence
        success = persistence.create_session(session_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        # Get session info
        sessions = persistence.get_all_sessions(limit=1)
        session_info = next((s for s in sessions if s["session_id"] == session_id), None)
        
        if not session_info:
            raise HTTPException(status_code=500, detail="Session created but not found")
        
        return SessionResponse(
            session_id=session_id,
            created_at=session_info["created_at"],
            message_count=session_info["message_count"]
        )
        
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(limit: int = 20):
    """Get list of all sessions"""
    try:
        sessions = persistence.get_all_sessions(limit=limit)
        return [
            SessionResponse(
                session_id=s["session_id"],
                created_at=s["created_at"],
                message_count=s["message_count"]
            )
            for s in sessions
        ]
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/history", response_model=List[MessageResponse])
async def get_session_history(session_id: str, limit: int = 50):
    """Get conversation history for a session"""
    try:
        messages = persistence.get_session_history(session_id, limit=limit)
        return [
            MessageResponse(
                role=msg["role"],
                text=msg["text"],
                language=msg.get("language"),
                timestamp=msg["timestamp"]
            )
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages"""
    try:
        success = persistence.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Disconnect WebSocket if active
        manager.disconnect(session_id)
        
        return {"message": "Session deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        stats = persistence.get_session_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time conversation
@app.websocket("/conversation/{session_id}")
async def conversation_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time conversation"""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process the conversation
            result = await conversation_graph.process_conversation(
                audio_data=data,
                session_id=session_id,
                language="auto"
            )
            
            # Send response back to client (without audio_response to avoid JSON serialization issues)
            response_data = result.copy()
            if "audio_response" in response_data:
                # Encode audio as base64 for JSON serialization
                import base64
                audio_base64 = base64.b64encode(response_data["audio_response"]).decode('utf-8')
                response_data["audio_response"] = audio_base64
            
            # Debug logging
            logger.info(f"Sending conversation response: user_text='{result.get('user_text', '')[:50]}...', ai_text='{result.get('ai_text', '')[:50]}...', audio_length={len(result.get('audio_response', b''))}")
            
            await manager.send_message(session_id, {
                "type": "conversation_response",
                "data": response_data
            })
            
            # Send partial transcription if available
            if result.get("user_text"):
                await manager.send_message(session_id, {
                    "type": "transcription",
                    "data": {
                        "text": result["user_text"],
                        "language": result["language"],
                        "is_final": True
                    }
                })
            
            # Send AI response
            if result.get("ai_text"):
                logger.info(f"Sending AI response: '{result['ai_text'][:100]}...'")
                await manager.send_message(session_id, {
                    "type": "ai_response",
                    "data": {
                        "text": result["ai_text"],
                        "language": result["language"]
                    }
                })
            
            # Send audio response
            if result.get("audio_response"):
                # Encode audio data as base64 for JSON serialization
                import base64
                audio_base64 = base64.b64encode(result["audio_response"]).decode('utf-8')
                
                # Get the actual sample rate from the TTS result
                # The TTS engine now returns the correct sample rate
                sample_rate = result.get('audio_sample_rate', 22050)  # Default fallback
                
                logger.info(f"Sending audio response: {len(result['audio_response'])} bytes, sample rate: {sample_rate}Hz")
                
                await manager.send_message(session_id, {
                    "type": "audio_response",
                    "data": {
                        "audio_data": audio_base64,
                        "language": result["language"],
                        "sample_rate": sample_rate
                    }
                })
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(session_id)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"message": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "path": str(request.url)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
