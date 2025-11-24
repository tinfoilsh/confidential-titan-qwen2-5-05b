#!/usr/bin/env python3
"""
GGUF-based API server for title generation using llama-cpp-python
OpenAI-compatible /v1/chat/completions endpoint
CPU-only, optimized for TiTan-Qwen2.5-0.5B-GGUF or similar small models.
"""

import os
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from llama_cpp import Llama
import string

# ----------------------------
# Environment configuration
# ----------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "theprint/TiTan-Qwen2.5-0.5B-GGUF")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "titan-qwen2-5-05b")
MODEL_FILE = os.getenv("MODEL_FILE", "")  # optional explicit gguf path
PORT = int(os.getenv("PORT", "8000"))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "8192"))

app = FastAPI(title="Title Generation API (GGUF CPU)")

llm: Optional[Llama] = None


# ----------------------------
# Request / response models
# ----------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 8
    stop: Optional[List[str]] = None


# ----------------------------
# Model loading helpers
# ----------------------------

def find_gguf_file(path_or_repo: str) -> str:
    """
    Resolve a GGUF file from either:
    - a local file path
    - a local directory containing *.gguf
    - a Hugging Face repo ID containing *.gguf
    """
    p = Path(path_or_repo)

    # Local path handling
    if p.exists():
        if p.is_file() and p.suffix == ".gguf":
            return str(p)
        if p.is_dir():
            ggufs = sorted(p.glob("*.gguf"))
            if not ggufs:
                raise ValueError(f"No GGUF files found in directory: {path_or_repo}")
            return str(ggufs[0])

    # Hugging Face repo ID
    if "/" in path_or_repo:
        try:
            from huggingface_hub import list_repo_files, hf_hub_download
        except ImportError as e:
            raise ValueError(
                "huggingface_hub not installed. Install it or set MODEL_FILE to a local GGUF path."
            ) from e

        files = list_repo_files(path_or_repo)
        gguf_files = sorted(f for f in files if f.endswith(".gguf"))
        if not gguf_files:
            raise ValueError(f"No GGUF files found in HF repo: {path_or_repo}")
        filename = gguf_files[0]
        print(f"Downloading GGUF file '{filename}' from repo '{path_or_repo}'")
        return hf_hub_download(repo_id=path_or_repo, filename=filename)

    raise ValueError(f"Invalid GGUF path or repo: {path_or_repo}")


def load_model() -> None:
    """Load the GGUF model into llama-cpp (CPU)."""
    global llm

    print(f"[load_model] Loading model: {MODEL_NAME}")

    model_path = MODEL_FILE or find_gguf_file(MODEL_NAME)
    if not Path(model_path).exists():
        raise ValueError(f"Resolved model file does not exist: {model_path}")

    print(f"[load_model] Using GGUF file: {model_path}")

    llm = Llama(
        model_path=model_path,
        n_ctx=CONTEXT_SIZE,
        n_threads=None,   # auto: use all cores
        verbose=False,
    )

    print(f"[load_model] Model loaded successfully: {MODEL_NAME}")


@app.on_event("startup")
async def startup_event():
    """FastAPI startup hook: load model once."""
    load_model()


# ----------------------------
# Utility: title post-processing
# ----------------------------

TRAILING_STOPWORDS = {
    "and", "or", "for", "of", "in", "on", "at", "to", "with", "by", "the", "a", "an"
}


def clean_title(raw: str) -> str:
    """
    Minimal, strict cleanup:
    - remove markdown / prefixes (###, Title:, bullets)
    - enforce <= 5 words
    - enforce <= 40 chars (cut at word boundary where possible)
    - remove punctuation
    - trim trailing stopwords like 'in', 'for', 'and' so it doesn't look incomplete
    """
    if not raw:
        return ""

    t = raw.strip()

    # Remove obvious prefixes
    for prefix in ("###", "##", "#", "Title:", "title:", "-", "*"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()

    # Remove surrounding quotes
    t = t.strip('"').strip("'").strip()

    # Remove all special characters that shouldn't be in titles
    # Remove: quotes, brackets, markdown characters, and other problematic chars
    # Keep: basic punctuation like commas, periods, colons, semicolons, hyphens, ampersands
    problematic = '"\'()[]{}<>*_`~^'
    t = t.translate(str.maketrans("", "", problematic))

    # Collapse multiple spaces
    t = " ".join(t.split())

    if not t:
        return ""

    # Enforce word limit
    words = t.split()
    if len(words) > 5:
        words = words[:5]
        t = " ".join(words)

    # Enforce char limit, try not to cut mid-word
    if len(t) > 40:
        truncated = t[:40]
        # try cut back to last space
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        t = truncated.strip()

    # Re-split to handle case where truncation changed words
    words = t.split()
    # Remove trailing stopword if it makes it feel cut off
    if words:
        last = words[-1].lower()
        if last in TRAILING_STOPWORDS and len(words) > 1:
            words = words[:-1]
    
    # Remove trailing commas from all words
    words = [w.rstrip(',') for w in words]
    
    t = " ".join(words).strip()
    
    # Remove incomplete endings (single words that seem cut off)
    if words and len(words) > 1:
        last_word = words[-1].lower()
        # If last word is very short and seems incomplete, consider removing it
        if len(last_word) <= 3 and last_word not in ["ai", "pc", "it"]:
            words = words[:-1]
            t = " ".join(words).strip()

    return t


# ----------------------------
# API endpoints
# ----------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": SERVED_MODEL_NAME,
        "device": "cpu",
        "format": "gguf",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": 1234567890,
                "owned_by": "llama-cpp",
                "root": MODEL_NAME,
                "parent": None,
                "max_model_len": CONTEXT_SIZE,
                "permission": [],
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global llm

    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # We ignore any user-provided system prompt and enforce our own,
    # because this endpoint is *only* for title generation.
    conversation_text = ""

    # In your pipeline you typically stuff the full conversation into the last user message.
    # If there are multiple user messages, we join them.
    user_chunks = [m.content for m in request.messages if m.role == "user"]
    if user_chunks:
        conversation_text = "\n".join(user_chunks).strip()
    else:
        raise HTTPException(status_code=400, detail="User message required")

    # Strong system prompt to force short, clean titles
    system_prompt = """### Task:

Generate a concise, 3-5 word title summarizing the chat history.

### Guidelines:

- The title should clearly represent the main theme or subject of the conversation.
- Prioritize accuracy over excessive creativity; keep it clear and simple.
- Write the title in the chat's primary language; default to English if multilingual.
- Output ONLY the title text, nothing else - no JSON, no formatting, no prefixes.
- NO code (no Python, no functions, no imports, no ```)
- NO markdown formatting (no ###, no ##, no #)
- NO prefixes like "Title:" or "title:"
- NO explanations or additional text
- Maximum 5 words
- Maximum 40 characters
- Must be a complete, meaningful phrase - do not cut off mid-sentence

### Examples:

- Stock Market Trends
- Perfect Chocolate Chip Recipe
- Evolution of Music Streaming
- Remote Work Productivity Tips
- Artificial Intelligence in Healthcare
- Video Game Development Insights
"""

    # Try simpler prompt format - direct instruction
    prompt = f"""{system_prompt}

Conversation:
{conversation_text}

Title:"""

    # Sampling parameters: allow enough tokens for complete 5-word titles
    max_tokens = min(request.max_tokens or 15, 20)
    
    # Retry logic for empty responses
    max_retries = 3
    raw_text = ""
    result = None
    
    for attempt in range(max_retries):
        try:
            result = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.2 + (attempt * 0.1),  # Increase temperature on retries
                stop=["\n\n", "###", "Title:", "title:", "</s>", "[INST]", "```", "def ", "import ", "class ", "from ", "/SYS", "SYS", "\n"],
                echo=False,
            )
            raw_text = result["choices"][0]["text"].strip()
            
            # Remove code patterns immediately
            if raw_text.startswith("```") or "def " in raw_text or "import " in raw_text or "class " in raw_text:
                raw_text = ""
            
            # Remove common prefixes that might appear
            for prefix in ["Sure", "I'll", "I will", "SYS", "/SYS", "System:", "User:", "[", "("]:
                if raw_text.startswith(prefix):
                    raw_text = raw_text[len(prefix):].strip()
                    # Remove colon if present
                    if raw_text.startswith(":"):
                        raw_text = raw_text[1:].strip()
            
            if raw_text:
                break
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Model error: {e}")
            continue
    
    title = clean_title(raw_text)

    # It's possible (rare) that we end up with empty after cleaning;
    # in that case, fall back to the raw text cleaned only minimally.
    if not title and raw_text:
        title = raw_text.strip().strip('"').strip("'").strip()
        # Remove any remaining problematic prefixes
        for prefix in ["###", "##", "#", "Title:", "title:"]:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()

    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    response = {
        "id": f"chatcmpl-{os.urandom(8).hex()}",
        "object": "chat.completion",
        "created": 1234567890,
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": title,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    return JSONResponse(content=response)


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    print(f"Starting GGUF API server for {MODEL_NAME} on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
