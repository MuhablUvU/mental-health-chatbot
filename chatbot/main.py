from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import json

app = FastAPI(
    title="Mental Health Bilingual Chatbot",
    description="A mental health support chatbot for English and Arabic",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and tokenizer
MODEL_NAME = "facebook/blenderbot-400M-distill"  # You can replace with other multilingual models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Mental health related keywords and responses
MENTAL_HEALTH_PATTERNS = {
    "en": {
        "anxiety": ["I understand you're feeling anxious.", "Let's talk about what's causing your anxiety."],
        "depression": ["I hear that you're going through a difficult time.", "You're not alone in this."],
        "stress": ["It's normal to feel stressed.", "Let's discuss some stress management techniques."],
        "help": ["I'm here to support you.", "You're taking a positive step by reaching out."]
    },
    "ar": {
        "قلق": ["أتفهم شعورك بالقلق.", "دعنا نتحدث عما يسبب قلقك."],
        "اكتئاب": ["أسمع أنك تمر بوقت صعب.", "أنت لست وحدك في هذا."],
        "توتر": ["من الطبيعي أن تشعر بالتوتر.", "دعنا نناقش بعض تقنيات إدارة التوتر."],
        "مساعدة": ["أنا هنا لدعمك.", "أنت تتخذ خطوة إيجابية بالتواصل معنا."]
    }
}

class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    language: str
    timestamp: str
    support_resources: List[dict]
    confidence_score: float

class ChatSession:
    def __init__(self):
        self.conversation_history = []
        
    def add_message(self, message: str, is_user: bool = True):
        self.conversation_history.append({
            "content": message,
            "is_user": is_user,
            "timestamp": datetime.utcnow().isoformat()
        })

# Store chat sessions
chat_sessions = {}

def get_mental_health_resources(language: str) -> List[dict]:
    if language == "en":
        return [
            {
                "name": "24/7 Crisis Helpline",
                "contact": "1-800-273-8255",
                "type": "emergency"
            },
            {
                "name": "Online Counseling",
                "url": "https://www.betterhelp.com",
                "type": "counseling"
            }
        ]
    else:  # Arabic resources
        return [
            {
                "name": "خط المساعدة النفسية",
                "contact": "920033360",
                "type": "emergency"
            },
            {
                "name": "الاستشارات النفسية عبر الإنترنت",
                "url": "https://www.nafsi.health",
                "type": "counseling"
            }
        ]

async def process_message(message: str, language: str, session_id: str) -> tuple[str, float]:
    # Get or create session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession()
    
    session = chat_sessions[session_id]
    session.add_message(message)
    
    # Check for mental health keywords
    response = None
    for keyword, responses in MENTAL_HEALTH_PATTERNS[language].items():
        if keyword in message.lower():
            response = torch.randint(0, len(responses), (1,)).item()
            response = responses[response]
            break
    
    if not response:
        # Use the model for general conversation
        inputs = tokenizer(message, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    session.add_message(response, is_user=False)
    return response, 0.85

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Mental Health Support Chatbot",
        "supported_languages": ["English (en)", "Arabic (ar)"],
        "status": "operational",
        "last_updated": "2025-04-14"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if request.language not in ["en", "ar"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported language. Please use 'en' for English or 'ar' for Arabic"
            )
        
        session_id = request.session_id or f"session_{datetime.utcnow().timestamp()}"
        response_text, confidence = await process_message(
            request.message,
            request.language,
            session_id
        )
        
        return ChatResponse(
            response=response_text,
            language=request.language,
            timestamp=datetime.utcnow().isoformat(),
            support_resources=get_mental_health_resources(request.language),
            confidence_score=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": chat_sessions[session_id].conversation_history}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
