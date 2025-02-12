import os
import httpx
import asyncio
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class TextRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 1024
    temperature: float = 1.0
    stream: bool = False
    json_mode: bool = False
    moderation: bool = False
    top_p: float = 1.0
    seed: int | None = None
    stop: str | None = None

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running."}

@app.post("/generate-text/")
async def generate_text(request: TextRequest):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": request.stream,
    }

    if request.seed is not None:
        payload["seed"] = request.seed

    if request.stop:
        payload["stop"] = request.stop

    if request.moderation:
        payload["moderation"] = request.moderation

    if request.json_mode:
        payload["response_format"] = "json"

    if not request.stream:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            return response.json()

    # ✅ FIXED STREAMING HANDLING
    async def response_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                async for chunk in response.aiter_bytes():
                    chunk_data = chunk.decode("utf-8").strip()
                    
                    # ✅ Extract content from JSON response
                    for line in chunk_data.split("\n"):
                        if line.startswith("data:"):
                            try:
                                parsed_data = json.loads(line[5:].strip())  # Remove "data:" prefix and parse JSON
                                if "choices" in parsed_data:
                                    for choice in parsed_data["choices"]:
                                        if "delta" in choice and "content" in choice["delta"]:
                                            yield choice["delta"]["content"]  # Extract content only
                            except json.JSONDecodeError:
                                continue

    return StreamingResponse(response_generator(), media_type="text/event-stream")
