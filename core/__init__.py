"""Couche métier partagée entre la GUI et l'API."""

from .config import DEFAULT_SETTINGS, load_settings, save_settings
from .device import describe_device
from .generate import build_sampling_kwargs, generate
from .models import (
    SUPPORTED_MODELS,
    LoadedModel,
    ModelCache,
    is_gguf_model,
    is_vl_model,
    load_model,
    unload_model,
)
from .rag import RagIndex

__all__ = [
    "DEFAULT_SETTINGS",
    "SUPPORTED_MODELS",
    "LoadedModel",
    "ModelCache",
    "RagIndex",
    "build_sampling_kwargs",
    "describe_device",
    "generate",
    "is_gguf_model",
    "is_vl_model",
    "load_model",
    "load_settings",
    "save_settings",
    "unload_model",
]
