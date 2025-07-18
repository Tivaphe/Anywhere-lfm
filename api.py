import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import time
import uuid
import os
import json
import markdown2

# RAG specific imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
DEFAULT_MODEL = "LiquidAI/LFM2-350M"
RAG_DOCUMENTS_PATH = "documents/"
CONVERSATIONS_PATH = "conversations/"
os.makedirs(RAG_DOCUMENTS_PATH, exist_ok=True)
os.makedirs(CONVERSATIONS_PATH, exist_ok=True)

# --- Global Variables ---
model = None
tokenizer = None
vector_store = None
conversations = {}

# --- FastAPI App Initialization ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    rag_enabled: bool
    settings: dict

class Settings(BaseModel):
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.3
    min_p: float = 0.15
    repetition_penalty: float = 1.05
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50

# --- Model Loading ---
def load_model(model_name: str):
    global model, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto"
        )
        print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. API will not work. Error: {e}")
        model = None
        tokenizer = None

# --- RAG Functions ---
def initialize_rag():
    global vector_store
    try:
        docs = []
        for filename in os.listdir(RAG_DOCUMENTS_PATH):
            file_path = os.path.join(RAG_DOCUMENTS_PATH, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs.extend(loader.load())
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                docs.extend(loader.load())

        if not docs:
            print("No documents found for RAG.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vector_store = FAISS.from_documents(splits, embeddings)
        print("RAG vector store created successfully.")
    except Exception as e:
        print(f"Error creating RAG index: {e}")
        vector_store = None

# --- Conversation Management ---
def load_conversations():
    global conversations
    for filename in os.listdir(CONVERSATIONS_PATH):
        if filename.endswith(".json"):
            conv_id = filename.replace(".json", "")
            with open(os.path.join(CONVERSATIONS_PATH, filename), "r", encoding="utf-8") as f:
                conversations[conv_id] = json.load(f)

def save_conversation(conv_id: str):
    with open(os.path.join(CONVERSATIONS_PATH, f"{conv_id}.json"), "w", encoding="utf-8") as f:
        json.dump(conversations[conv_id], f, ensure_ascii=False, indent=2)

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not available.")

    conv_id = request.conversation_id
    if conv_id not in conversations:
        conversations[conv_id] = [{"role": "system", "content": request.settings.get("system_prompt", "You are a helpful assistant.")}]

    conversations[conv_id].append({"role": "user", "content": request.message})

    rag_context = ""
    if request.rag_enabled and vector_store:
        try:
            docs = vector_store.similarity_search(request.message, k=3)
            rag_context = "\n\nContext from documents:\n" + "\n---\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"RAG search error: {e}")

    conversation_history = list(conversations[conv_id])
    if rag_context:
        conversation_history[-1]["content"] = f"{rag_context}\n\nQuestion: {request.message}"

    input_ids = tokenizer.apply_chat_template(
        conversation_history, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=request.settings.get("temperature", 0.3),
        top_p=request.settings.get("min_p", 0.15) if request.settings.get("min_p", 0.15) > 0 else None,
        repetition_penalty=request.settings.get("repetition_penalty", 1.05)
    )

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
        new_tokens = outputs[0][input_ids.shape[-1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    conversations[conv_id].append({"role": "assistant", "content": response_text})
    save_conversation(conv_id)

    return JSONResponse({"response": markdown2.markdown(response_text, extras=["fenced-code-blocks", "tables"])})

@app.post("/api/conversations")
async def new_conversation():
    conv_id = str(uuid.uuid4())
    conversations[conv_id] = [{"role": "system", "content": "You are a helpful assistant."}]
    save_conversation(conv_id)
    return JSONResponse({"conversation_id": conv_id, "history": conversations[conv_id]})

@app.get("/api/conversations")
async def get_all_conversations():
    return JSONResponse(conversations)

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if conv_id in conversations:
        del conversations[conv_id]
        filepath = os.path.join(CONVERSATIONS_PATH, f"{conv_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        return JSONResponse({"status": "deleted"})
    raise HTTPException(status_code=404, detail="Conversation not found")

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    destination = os.path.join(RAG_DOCUMENTS_PATH, file.filename)
    with open(destination, "wb") as buffer:
        buffer.write(await file.read())
    initialize_rag()
    return JSONResponse({"status": "documents updated"})

@app.get("/api/models")
async def get_models():
    # In a real scenario, this could scan a directory or a config file
    return JSONResponse(["LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B"])

@app.post("/api/eject")
async def eject_model():
    global model, tokenizer
    model = None
    tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return JSONResponse({"status": "model ejected"})

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    load_model(DEFAULT_MODEL)
    initialize_rag()
    load_conversations()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
