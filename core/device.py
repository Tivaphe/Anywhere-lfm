"""Détection du périphérique d'inférence."""

from __future__ import annotations


def describe_device() -> str:
    try:
        import torch
    except ImportError:
        return "CPU"

    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "GPU: Apple MPS"
    return "CPU"


def infer_model_device(model):
    import torch

    if hasattr(model, "device"):
        try:
            return model.device
        except Exception:
            pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")
