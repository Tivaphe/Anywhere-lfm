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

# --- Initialisation de l'application et du modèle ---

print("Initialisation de l'API et chargement du modèle...")

# Modèle par défaut au démarrage
DEFAULT_MODEL = "LiquidAI/LFM2-350M"

try:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    print(f"Modèle par défaut '{DEFAULT_MODEL}' chargé avec succès.")
except Exception as e:
    print(f"ERREUR CRITIQUE : Impossible de charger le modèle par défaut. L'API ne pourra pas fonctionner. Erreur : {e}")
    model = None
    tokenizer = None

app = FastAPI()

# --- Endpoint de l'API ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas disponible ou n'a pas pu être chargé.")

    if request.model != DEFAULT_MODEL:
         # Pour l'instant, on ne supporte qu'un seul modèle chargé à la fois.
         # On pourrait ajouter une logique pour charger dynamiquement d'autres modèles ici.
        raise HTTPException(status_code=400, detail=f"Modèle non supporté. Seul le modèle '{DEFAULT_MODEL}' est actuellement chargé.")

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
    return {"status": "Le serveur de l'API LiquidAI est en ligne.", "model_loaded": DEFAULT_MODEL if model else "Aucun"}

if __name__ == "__main__":
    # Lancer le serveur
    uvicorn.run(app, host="0.0.0.0", port=8000)
