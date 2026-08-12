"""Paramètres par défaut et persistance locale."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

SETTINGS_PATH = os.path.join(os.getcwd(), "settings.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "system_prompt": "You are a helpful assistant.",
    "temperature": 0.3,
    "min_p": 0.15,
    "repetition_penalty": 1.05,
    "max_new_tokens": 512,
    "rag_chunk_size": 500,
    "rag_chunk_overlap": 50,
}


def load_settings(path: str = SETTINGS_PATH) -> Dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if not os.path.exists(path):
        return settings
    try:
        with open(path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if isinstance(stored, dict):
            settings.update(stored)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)
    return settings


def save_settings(settings: Dict[str, Any], path: str = SETTINGS_PATH) -> None:
    payload = dict(DEFAULT_SETTINGS)
    payload.update(settings)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
