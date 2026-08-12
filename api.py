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
import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, list_repo_files

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
    "LiquidAI/LFM2-VL-3B", "LiquidAI/LFM2-VL-1.6B", "LiquidAI/LFM2-VL-450M",
    # GGUF Models
    "LiquidAI/LFM2-2.6B-GGUF", "LiquidAI/LFM2-8B-A1B-GGUF",
    "LiquidAI/LFM2-1.2B-GGUF", "LiquidAI/LFM2-700M-GGUF", "LiquidAI/LFM2-350M-GGUF"
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
        is_gguf_model = "GGUF" in model_name
        is_vl_model = "VL" in model_name

        if is_gguf_model:
            print(f"Recherche du fichier GGUF pour {model_name}...")
            repo_files = list_repo_files(model_name)
            gguf_files = [f for f in repo_files if f.endswith(".gguf")]
            if not gguf_files:
                raise FileNotFoundError(f"Aucun fichier GGUF trouvé dans le repository {model_name}")

            model_file = next((f for f in gguf_files if "Q4_K_M" in f.upper()), gguf_files[0])
            print(f"Téléchargement du fichier GGUF : {model_file}...")

            model_path = hf_hub_download(
                repo_id=model_name,
                filename=model_file,
                local_dir=os.path.join(os.getcwd(), 'models'),
                local_dir_use_symlinks=False
            )

            print(f"Chargement de {model_file} avec llama.cpp...")
            model = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=True)
            model_cache[model_name] = (model, None) # Llama models don't have a separate processor object

        elif is_vl_model:
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
        is_gguf_model = isinstance(model, Llama)
        is_vl_model = not is_gguf_model and "VL" in model.config._name_or_path

        # Le format de l'historique pour llama.cpp est déjà correct
        conversation_history = [msg.dict() for msg in request.messages]

        if is_gguf_model:
            # --- Génération avec Llama.cpp ---
            output = model.create_chat_completion(
                messages=conversation_history,
                temperature=request.temperature,
                max_tokens=512
            )
            response_text = output['choices'][0]['message']['content']

        else:
            # --- Génération avec Transformers (VL ou Texte) ---
            image_input = None
            # Pour les modèles transformers, on doit extraire le texte et l'image
            parsed_conversation = []
            for msg in request.messages:
                if isinstance(msg.content, str):
                    parsed_conversation.append({"role": msg.role, "content": msg.content})
                else: # Multimodal
                    text_content = ""
                    for part in msg.content:
                        if part.type == "text":
                            text_content += part.text + " "
                        elif part.type == "image_url":
                            if image_input: raise HTTPException(status_code=400, detail="Une seule image par message.")
                            img_data = part.image_url['url']
                            image_input = Image.open(BytesIO(base64.b64decode(img_data.split(",")[1]))) if img_data.startswith("data:image") else load_image(img_data)
                    parsed_conversation.append({"role": msg.role, "content": text_content.strip()})

            if is_vl_model and image_input:
                last_msg = parsed_conversation[-1]
                last_msg['content'] = [{"type": "image", "image": image_input}, {"type": "text", "text": last_msg.get('content', '')}]

            inputs = processor.apply_chat_template(parsed_conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=request.temperature
                )

            response_text = processor.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        # --- Créer la réponse au format OpenAI ---
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
