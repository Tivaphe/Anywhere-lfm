"""Chargement, cache et déchargement des modèles LFM2."""

from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .catalog import (
    fallback_repo_ids,
    get_cached_repo_ids,
    is_allowed_model,
    list_gguf_files,
    parse_model_ref,
    pick_gguf_file,
)

SUPPORTED_MODELS: List[str] = fallback_repo_ids()

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
    repo, spec = parse_model_ref(name)
    if spec and str(spec).lower().endswith(".gguf"):
        return True
    return "GGUF" in (repo or name or "")


def is_vl_model(name: str) -> bool:
    if is_gguf_model(name):
        return False
    repo, _ = parse_model_ref(name)
    short = (repo or name or "").split("/", 1)[-1]
    return "VL" in short


def available_model_ids() -> List[str]:
    return get_cached_repo_ids() or list(SUPPORTED_MODELS)


def _emit(callback: StatusCallback, message: str) -> None:
    if callback:
        callback(message)


def load_model(
    model_name: str,
    status_callback: StatusCallback = None,
    offload_folder: str = "./offload",
    allowed_models: Optional[List[str]] = None,
) -> LoadedModel:
    repo_id, _ = parse_model_ref(model_name)
    if allowed_models is not None:
        allowed = model_name in allowed_models or repo_id in allowed_models
        if not allowed:
            raise UnsupportedModelError(
                f"Modèle non supporté: {model_name}. "
                f"Disponibles: {', '.join(allowed_models)}"
            )
    elif not is_allowed_model(model_name):
        raise UnsupportedModelError(
            f"Modèle non supporté: {model_name}. "
            "Seuls les dépôts LiquidAI/LFM* (texte, VL, GGUF) sont acceptés."
        )

    os.makedirs(offload_folder, exist_ok=True)

    if is_gguf_model(model_name):
        return _load_gguf(model_name, status_callback)
    if is_vl_model(model_name):
        return _load_vl(repo_id, offload_folder, status_callback)
    return _load_text(repo_id, offload_folder, status_callback)


def _load_gguf(model_name: str, status_callback: StatusCallback) -> LoadedModel:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    repo_id, spec = parse_model_ref(model_name)
    _emit(status_callback, f"Recherche des fichiers GGUF dans {repo_id}...")
    gguf_files = list_gguf_files(repo_id)
    if not gguf_files:
        raise FileNotFoundError(f"Aucun fichier GGUF trouvé dans {repo_id}")

    model_file = pick_gguf_file(gguf_files, spec)
    loaded_name = f"{repo_id}:{os.path.basename(model_file)}"
    _emit(status_callback, f"Téléchargement du modèle GGUF : {model_file}...")
    model_path = hf_hub_download(
        repo_id=repo_id,
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
    return LoadedModel(name=loaded_name, kind="gguf", model=model, processor=None)


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
            if not is_allowed_model(model_name):
                raise UnsupportedModelError(
                    f"Modèle non supporté: {model_name}. "
                    "Seuls les dépôts LiquidAI/LFM* sont acceptés."
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
