import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uuid
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import uvicorn
import google.generativeai as genai

load_dotenv()

class APIConfig:
    """Configuration management for API settings"""
    API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = 'gemini-2.0-flash-001'  
    MAX_HISTORY_LENGTH = 10
    REQUEST_TIMEOUT = timedelta(seconds=30)
    MAX_RESPONSE_LENGTH = 5000  

class ChatMessage(BaseModel):
    """Represents a single chat message"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str = Field(...) 
    content: str = Field(..., min_length=1, max_length=2000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant']:
            raise ValueError('Role must be either "user" or "assistant"')
        return v

class UserQuery(BaseModel):
    """Input model for user queries"""
    query: str = Field(..., min_length=1, max_length=500, 
                       description="User's query about SHAP values and feature importance")
    context: Optional[Dict[str, str]] = Field(default=None, 
                                             description="Optional context for query")

class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str = Field(..., description="Chatbot's response", max_length=2000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Optional[Dict[str, str]] = Field(default=None)

class ChatHistory:
    """Manages chat session history"""
    def __init__(self, max_length: int = 10):
        self.messages: List[ChatMessage] = []
        self.max_length = max_length

    def add_message(self, message: ChatMessage):
        """Add a message to chat history"""
        if len(self.messages) >= self.max_length:
            self.messages.pop(0)
        self.messages.append(message)

    def get_context(self) -> List[Dict[str, str]]:
        """Convert history to context for model"""
        return [{"role": msg.role, "parts": [msg.content]} for msg in self.messages]

class SHAPExplanationChatbot:
    """Main chatbot class handling SHAP value explanations"""
    def __init__(self, api_key: str): 
        genai.configure(api_key=api_key)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            APIConfig.MODEL_NAME, 
            safety_settings=safety_settings
        )
        self.chat_session = self.model.start_chat(history=[])
        self.history = ChatHistory(max_length=APIConfig.MAX_HISTORY_LENGTH)

    def process_query(self, user_query: UserQuery) -> ChatResponse:
        """Process user query and generate response about SHAP values"""
        try:
            full_prompt = f"""
            You are advanced Machine Learning and Deep Learning model explainer of nuvix.
            nuvix is the platform to calculate risk on any credit.
            We use SHAP waterfall plot to explain the predictions of any ML or DL model.
            
            Query: {user_query.query}

            Response Guidelines:
            - Give the brief, positive and convincing explanation for higher SHAP values of the features
            - Don't mention that you are using SHAP for the explanation.
            - Be precise with the responses.
            - Focus only on feature importance and SHAP value explanations.
            - Do not respond to any location-based queries.
            """

            user_message = ChatMessage(role='user', content=full_prompt)
            self.history.add_message(user_message)
            response = self.chat_session.send_message(full_prompt)
            response_text = response.text[:APIConfig.MAX_RESPONSE_LENGTH]
            assistant_message = ChatMessage(
                role='assistant', 
                content=response_text
            )
            self.history.add_message(assistant_message)
            return ChatResponse(
                message=response_text,
                context=user_query.context
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

app = FastAPI(
    title="SHAP Value Explanation API",
    description="AI-powered chatbot for explaining SHAP values and feature importance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_chatbot():
    """Dependency to get chatbot instance"""
    if not APIConfig.API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return SHAPExplanationChatbot(APIConfig.API_KEY)

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "SHAP Value Explanation API",
        "status": "Running",
        "version": "1.0.0",
        "description": "AI-powered explanations for SHAP values and feature importance"
    }

@app.post("/explain", response_model=ChatResponse)
async def explain_endpoint(
    user_query: UserQuery, 
    chatbot: SHAPExplanationChatbot = Depends(get_chatbot)
):
    """Main endpoint for SHAP value explanations"""
    return chatbot.process_query(user_query)

if __name__ == "__main__":
    if not APIConfig.API_KEY:
        raise ValueError("GEMINI_API_KEY must be set in .env file")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )