"""Chargement, cache et déchargement des modèles LFM2."""

from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

SUPPORTED_MODELS: List[str] = [
    # Text
    "LiquidAI/LFM2-350M",
    "LiquidAI/LFM2-700M",
    "LiquidAI/LFM2-1.2B",
    "LiquidAI/LFM2-8B-A1B",
    "LiquidAI/LFM2-2.6B",
    "LiquidAI/LFM2-2.6B-Exp",
    "LiquidAI/LFM2-1.2B-Extract",
    "LiquidAI/LFM2-350M-Extract",
    "LiquidAI/LFM2-1.2B-RAG",
    "LiquidAI/LFM2-1.2B-Tool",
    "LiquidAI/LFM2-350M-Math",
    # Vision-Language
    "LiquidAI/LFM2-VL-3B",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-450M",
    # GGUF
    "LiquidAI/LFM2-2.6B-GGUF",
    "LiquidAI/LFM2-8B-A1B-GGUF",
    "LiquidAI/LFM2-1.2B-GGUF",
    "LiquidAI/LFM2-700M-GGUF",
    "LiquidAI/LFM2-350M-GGUF",
]

StatusCallback = Optional[Callable[[str], None]]


@dataclass
class LoadedModel:
    name: str
    kind: str  # "text" | "vl" | "gguf"
    model: object
    processor: object = None


class UnsupportedModelError(ValueError):
    pass


def is_gguf_model(name: str) -> bool:
    return "GGUF" in (name or "")


def is_vl_model(name: str) -> bool:
    return "VL" in (name or "")


def _emit(callback: StatusCallback, message: str) -> None:
    if callback:
        callback(message)


def load_model(
    model_name: str,
    status_callback: StatusCallback = None,
    offload_folder: str = "./offload",
    allowed_models: Optional[List[str]] = None,
) -> LoadedModel:
    catalog = allowed_models if allowed_models is not None else SUPPORTED_MODELS
    if catalog and model_name not in catalog:
        raise UnsupportedModelError(
            f"Modèle non supporté: {model_name}. "
            f"Disponibles: {', '.join(catalog)}"
        )

    os.makedirs(offload_folder, exist_ok=True)

    if is_gguf_model(model_name):
        return _load_gguf(model_name, status_callback)
    if is_vl_model(model_name):
        return _load_vl(model_name, offload_folder, status_callback)
    return _load_text(model_name, offload_folder, status_callback)


def _load_gguf(model_name: str, status_callback: StatusCallback) -> LoadedModel:
    from huggingface_hub import hf_hub_download, list_repo_files
    from llama_cpp import Llama

    _emit(status_callback, "Recherche du fichier GGUF sur le Hugging Face Hub...")
    repo_files = list_repo_files(model_name)
    gguf_files = [name for name in repo_files if name.endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError(f"Aucun fichier GGUF trouvé dans {model_name}")

    model_file = next(
        (name for name in gguf_files if "Q4_K_M" in name.upper()),
        gguf_files[0],
    )
    _emit(status_callback, f"Téléchargement du modèle GGUF : {model_file}...")
    model_path = hf_hub_download(
        repo_id=model_name,
        filename=model_file,
        local_dir=os.path.join(os.getcwd(), "models"),
    )
    _emit(status_callback, f"Chargement de {model_file} avec llama.cpp...")
    model = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    return LoadedModel(name=model_name, kind="gguf", model=model, processor=None)


def _load_vl(model_name: str, offload_folder: str, status_callback: StatusCallback) -> LoadedModel:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _emit(status_callback, f"Chargement du modèle vision {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        offload_folder=offload_folder,
    )
    return LoadedModel(name=model_name, kind="vl", model=model, processor=processor)


def _load_text(model_name: str, offload_folder: str, status_callback: StatusCallback) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _emit(status_callback, f"Chargement du modèle texte {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        offload_folder=offload_folder,
    )
    return LoadedModel(name=model_name, kind="text", model=model, processor=tokenizer)


def unload_model(loaded: Optional[LoadedModel]) -> None:
    if loaded is None:
        return
    model = getattr(loaded, "model", None)
    if model is not None and hasattr(model, "close"):
        try:
            model.close()
        except Exception:
            pass
    loaded.model = None
    loaded.processor = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class ModelCache:
    """Garde un seul modèle en mémoire pour éviter les OOM."""

    def __init__(self) -> None:
        self._models: Dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str, status_callback: StatusCallback = None) -> LoadedModel:
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]
            if model_name not in SUPPORTED_MODELS:
                raise UnsupportedModelError(
                    f"Modèle non supporté: {model_name}. "
                    f"Disponibles: {', '.join(SUPPORTED_MODELS)}"
                )
            for name in list(self._models):
                unload_model(self._models.pop(name))
            loaded = load_model(model_name, status_callback=status_callback)
            self._models[model_name] = loaded
            return loaded

    def unload(self, model_name: Optional[str] = None) -> None:
        with self._lock:
            names = [model_name] if model_name else list(self._models)
            for name in names:
                loaded = self._models.pop(name, None)
                unload_model(loaded)

    def loaded_names(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())
