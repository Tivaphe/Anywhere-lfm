import asyncio
import base64
import os
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers.image_utils import load_image

from core.config import DEFAULT_SETTINGS
from core.generate import generate
from core.models import SUPPORTED_MODELS, ModelCache, UnsupportedModelError

API_KEY = os.environ.get("LIQUIDAI_API_KEY")
HOST = os.environ.get("LIQUIDAI_HOST", "127.0.0.1")
PORT = int(os.environ.get("LIQUIDAI_PORT", "8000"))
PRELOAD_MODEL = os.environ.get("LIQUIDAI_PRELOAD_MODEL")

model_cache = ModelCache()


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = DEFAULT_SETTINGS["temperature"]
    min_p: Optional[float] = DEFAULT_SETTINGS["min_p"]
    repetition_penalty: Optional[float] = DEFAULT_SETTINGS["repetition_penalty"]
    max_tokens: Optional[int] = DEFAULT_SETTINGS["max_new_tokens"]
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]


class UnloadRequest(BaseModel):
    model: Optional[str] = None


def _decode_image(url: str):
    if url.startswith("data:image"):
        encoded = url.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(encoded)))
    return load_image(url)


def parse_messages(messages: List[ChatMessage]):
    history = []
    image = None
    for message in messages:
        if isinstance(message.content, str):
            history.append({"role": message.role, "content": message.content})
            continue

        texts = []
        for part in message.content:
            if part.type == "text" and part.text:
                texts.append(part.text)
            elif part.type == "image_url" and part.image_url:
                url = part.image_url.get("url")
                if not url:
                    continue
                if image is not None:
                    raise HTTPException(status_code=400, detail="Une seule image par requête est supportée.")
                image = _decode_image(url)
        history.append({"role": message.role, "content": " ".join(texts).strip()})
    return history, image


async def verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Jeton d'API invalide ou manquant.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD_MODEL:
        print(f"Préchargement du modèle {PRELOAD_MODEL}...")
        try:
            await asyncio.to_thread(model_cache.get, PRELOAD_MODEL)
        except Exception as exc:
            print(f"AVERTISSEMENT : préchargement impossible ({exc}).")
    yield
    model_cache.unload()


app = FastAPI(title="Anywhere-LFM API", lifespan=lifespan)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    _: None = Depends(verify_api_key),
):
    if request.stream:
        raise HTTPException(status_code=400, detail="Le streaming n'est pas encore implémenté.")

    try:
        loaded = await asyncio.to_thread(model_cache.get, request.model)
    except UnsupportedModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle : {exc}") from exc

    try:
        history, image = parse_messages(request.messages)
        settings = {
            "temperature": request.temperature if request.temperature is not None else DEFAULT_SETTINGS["temperature"],
            "min_p": request.min_p if request.min_p is not None else DEFAULT_SETTINGS["min_p"],
            "repetition_penalty": (
                request.repetition_penalty
                if request.repetition_penalty is not None
                else DEFAULT_SETTINGS["repetition_penalty"]
            ),
            "max_new_tokens": request.max_tokens or DEFAULT_SETTINGS["max_new_tokens"],
        }

        response_text, _ = await asyncio.to_thread(
            generate,
            loaded,
            history,
            settings,
            image,
            None,
            None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du texte : {exc}") from exc

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
    )


@app.get("/v1/models")
async def list_models():
    loaded = set(model_cache.loaded_names())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "owned_by": "liquid-ai",
                "loaded": name in loaded,
            }
            for name in SUPPORTED_MODELS
        ],
    }


@app.post("/v1/models/unload")
async def unload_models(payload: UnloadRequest, _: None = Depends(verify_api_key)):
    await asyncio.to_thread(model_cache.unload, payload.model)
    return {"status": "ok", "loaded": model_cache.loaded_names()}


@app.get("/")
def read_root():
    loaded_models = model_cache.loaded_names()
    return {
        "status": "Le serveur de l'API LiquidAI est en ligne.",
        "bind": f"{HOST}:{PORT}",
        "auth_required": bool(API_KEY),
        "models_available": SUPPORTED_MODELS,
        "models_loaded_in_cache": loaded_models or "Aucun",
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
