import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import uuid

# --- Modèles Pydantic pour la validation des requêtes et réponses ---
# Similaire à la structure de l'API OpenAI

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str # Nom du modèle à utiliser
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
    "LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B",
    "LiquidAI/LFM2-8B-A1B", "LiquidAI/LFM2-2.6B", "LiquidAI/LFM2-2.6B-Exp",
    "LiquidAI/LFM2-1.2B-Extract", "LiquidAI/LFM2-350M-Extract",
    "LiquidAI/LFM2-1.2B-RAG", "LiquidAI/LFM2-1.2B-Tool", "LiquidAI/LFM2-350M-Math"
]

def get_model(model_name: str):
    """
    Charge un modèle et son tokenizer, en utilisant un cache pour éviter les rechargements.
    """
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Modèle non supporté. Les modèles disponibles sont : {', '.join(SUPPORTED_MODELS)}")

    # Si le modèle est déjà dans le cache, on le retourne
    if model_name in model_cache:
        return model_cache[model_name]

    # Sinon, on le charge
    print(f"Chargement du modèle '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto"
        )
        model_cache[model_name] = (model, tokenizer)
        print(f"Modèle '{model_name}' chargé et mis en cache.")
        return model, tokenizer
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
    # Charger dynamiquement le modèle demandé
    try:
        model, tokenizer = get_model(request.model)
    except HTTPException as e:
        # Propage l'exception HTTP si le modèle n'est pas supporté ou ne peut pas être chargé
        raise e
    except Exception as e:
        # Gère les autres erreurs potentielles
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")

    if request.stream:
        raise HTTPException(status_code=400, detail="Le streaming n'est pas encore implémenté.")

    # Convertir les messages Pydantic en format attendu par le tokenizer
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    try:
        # Appliquer le template de chat
        input_ids = tokenizer.apply_chat_template(
            conversation_history, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # Générer la réponse
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=request.temperature
            )

        # Décoder seulement les nouveaux tokens
        new_tokens = outputs[0][input_ids.shape[-1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

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
