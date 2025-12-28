import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
                          AutoModelForImageTextToText)
from transformers.image_utils import load_image
import time
import uuid
import base64
from io import BytesIO
from PIL import Image

# --- Modèles Pydantic pour la validation des requêtes et réponses ---
# Similaire à la structure de l'API OpenAI

# --- Modèles Pydantic pour la validation des requêtes et réponses ---
# Adapté pour supporter le format multimodal (texte + image)

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    # Pour l'image, on attend une URL ou une chaîne base64
    image_url: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    # D'autres paramètres comme top_p, max_tokens, etc. peuvent être ajoutés ici
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

# --- Gestion dynamique des modèles ---

# Cache pour les modèles chargés (pour éviter de recharger depuis le disque)
model_cache = {}

# Liste des modèles autorisés pour le chargement
SUPPORTED_MODELS = [
    # Text Models
    "LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B",
    "LiquidAI/LFM2-8B-A1B", "LiquidAI/LFM2-2.6B", "LiquidAI/LFM2-2.6B-Exp",
    "LiquidAI/LFM2-1.2B-Extract", "LiquidAI/LFM2-350M-Extract",
    "LiquidAI/LFM2-1.2B-RAG", "LiquidAI/LFM2-1.2B-Tool", "LiquidAI/LFM2-350M-Math",
    # Vision-Language Models
    "LiquidAI/LFM2-VL-3B", "LiquidAI/LFM2-VL-1.6B", "LiquidAI/LFM2-VL-450M"
]

def get_model(model_name: str):
    """
    Charge un modèle et son processeur/tokenizer, en utilisant un cache.
    """
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Modèle non supporté. Les modèles disponibles sont : {', '.join(SUPPORTED_MODELS)}")

    if model_name in model_cache:
        return model_cache[model_name]

    print(f"Chargement du modèle '{model_name}'...")
    try:
        is_vl_model = "VL" in model_name
        if is_vl_model:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            model_cache[model_name] = (model, processor)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto"
            )
            model_cache[model_name] = (model, tokenizer)

        print(f"Modèle '{model_name}' chargé et mis en cache.")
        return model_cache[model_name]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle '{model_name}': {e}")

app = FastAPI()

# Pré-chargement du modèle par défaut pour accélérer le premier appel (optionnel)
DEFAULT_MODEL = "LiquidAI/LFM2-1.2B"
@app.on_event("startup")
async def startup_event():
    print("Démarrage de l'API et pré-chargement du modèle par défaut...")
    try:
        get_model(DEFAULT_MODEL)
    except Exception as e:
        print(f"AVERTISSEMENT : Impossible de pré-charger le modèle par défaut '{DEFAULT_MODEL}'. L'API démarrera sans modèle pré-chargé. Erreur : {e}")

# --- Endpoint de l'API ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        model, processor = get_model(request.model)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")

    if request.stream:
        raise HTTPException(status_code=400, detail="Le streaming n'est pas encore implémenté.")

    # --- Préparation de l'input pour le modèle ---
    try:
        is_vl_model = "VL" in model.config._name_or_path
        conversation_history = []
        image_input = None

        for msg in request.messages:
            if isinstance(msg.content, str):
                conversation_history.append({"role": msg.role, "content": msg.content})
            else: # C'est une liste de parties (multimodal)
                text_content = ""
                for part in msg.content:
                    if part.type == "text":
                        text_content += part.text + " "
                    elif part.type == "image_url":
                        if image_input is not None:
                            raise HTTPException(status_code=400, detail="Une seule image par message est supportée.")

                        image_data = part.image_url['url']
                        if image_data.startswith("data:image"):
                            # Image en base64
                            base64_str = image_data.split(",")[1]
                            image_input = Image.open(BytesIO(base64.b64decode(base64_str)))
                        else:
                            # URL de l'image
                            image_input = load_image(image_data)

                conversation_history.append({"role": msg.role, "content": text_content.strip()})

        if is_vl_model and image_input:
             # Le processeur VL attend que le dernier message contienne l'image
            last_message = conversation_history[-1]
            last_message['content'] = [
                {"type": "image", "image": image_input},
                {"type": "text", "text": last_message.get('content', '')}
            ]

        # --- Génération de la réponse ---
        if is_vl_model:
            inputs = processor.apply_chat_template(conversation_history, add_generation_prompt=True, return_tensors="pt").to(model.device)
            generation_kwargs = dict(**inputs)
        else: # Modèle texte seulement
            inputs = processor.apply_chat_template(conversation_history, add_generation_prompt=True, return_tensors="pt").to(model.device)
            generation_kwargs = dict(input_ids=inputs)

        generation_kwargs.update(
            max_new_tokens=512,
            do_sample=True,
            temperature=request.temperature
        )

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        input_ids_length = generation_kwargs['input_ids'].shape[-1]
        new_tokens = outputs[0][input_ids_length:]
        response_text = processor.decode(new_tokens, skip_special_tokens=True)

        # Créer la réponse au format OpenAI
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ]
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du texte : {e}")


@app.get("/")
def read_root():
    loaded_models = list(model_cache.keys())
    return {
        "status": "Le serveur de l'API LiquidAI est en ligne.",
        "models_available": SUPPORTED_MODELS,
        "models_loaded_in_cache": loaded_models if loaded_models else "Aucun"
    }

if __name__ == "__main__":
    # Lancer le serveur
    uvicorn.run(app, host="0.0.0.0", port=8000)
