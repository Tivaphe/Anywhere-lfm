"""Couche métier partagée entre la GUI et l'API."""

from .config import DEFAULT_SETTINGS, load_settings, save_settings
from .device import describe_device
from .generate import build_sampling_kwargs, generate
from .catalog import (
    Catalog,
    ModelEntry,
    get_catalog,
    is_allowed_model,
    refresh_catalog,
)
from .models import (
    SUPPORTED_MODELS,
    LoadedModel,
    ModelCache,
    available_model_ids,
    is_gguf_model,
    is_vl_model,
    load_model,
    unload_model,
)
from .rag import RagIndex

__all__ = [
    "DEFAULT_SETTINGS",
    "SUPPORTED_MODELS",
    "Catalog",
    "LoadedModel",
    "ModelCache",
    "ModelEntry",
    "RagIndex",
    "available_model_ids",
    "build_sampling_kwargs",
    "describe_device",
    "generate",
    "get_catalog",
    "is_allowed_model",
    "is_gguf_model",
    "is_vl_model",
    "load_model",
    "load_settings",
    "refresh_catalog",
    "save_settings",
    "unload_model",
]
