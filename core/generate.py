"""Génération unifiée (texte, vision, GGUF) pour la GUI et l'API."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .device import infer_model_device
from .models import LoadedModel

TokenCallback = Optional[Callable[[str], None]]


def build_sampling_kwargs(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Construit les kwargs Transformers. `min_p` n'est plus mappé sur `top_p`."""
    temperature = float(settings.get("temperature", 0.3))
    min_p = float(settings.get("min_p", 0.15))
    repetition_penalty = float(settings.get("repetition_penalty", 1.05))
    max_new_tokens = int(settings.get("max_new_tokens", 512))

    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    if temperature <= 0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        if min_p > 0:
            kwargs["min_p"] = min_p
    return kwargs


def as_text_messages(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalise un historique multimodal vers du texte pour llama.cpp / RAG."""
    messages: List[Dict[str, str]] = []
    for message in history:
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" or "text" in part:
                        parts.append(part.get("text") or "")
                elif isinstance(part, str):
                    parts.append(part)
            content = " ".join(part for part in parts if part).strip()
        messages.append({"role": message.get("role", "user"), "content": content or ""})
    return messages


class CallbackStreamer:
    """Interface compatible TextStreamer : pousse les tokens vers un callback."""

    def __init__(self, tokenizer, on_token: Callable[[str], None], skip_prompt: bool = True, **decode_kwargs):
        self.tokenizer = tokenizer
        self.on_token = on_token
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.token_cache: List[int] = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("CallbackStreamer ne supporte que batch_size=1")
        if len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        printable = text if self.print_len == 0 else text[self.print_len:]
        self.print_len = len(text)
        if printable:
            self.on_token(printable)

    def end(self):
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        printable = text[self.print_len:]
        if printable:
            self.on_token(printable)
        self.next_tokens_are_prompt = True
        self.token_cache = []
        self.print_len = 0


def _apply_chat_template(processor, conversation, device):
    import torch

    kwargs = {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "tokenize": True,
        "return_dict": True,
    }
    try:
        inputs = processor.apply_chat_template(conversation, **kwargs)
    except TypeError:
        kwargs.pop("return_dict", None)
        try:
            inputs = processor.apply_chat_template(conversation, **kwargs)
        except TypeError:
            kwargs.pop("tokenize", None)
            inputs = processor.apply_chat_template(conversation, **kwargs)

    if hasattr(inputs, "to"):
        try:
            inputs = inputs.to(device)
        except Exception:
            pass

    if torch.is_tensor(inputs):
        return {"input_ids": inputs.to(device)}

    if hasattr(inputs, "keys"):
        moved = {}
        for key, value in dict(inputs).items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved

    raise TypeError(f"Format d'entrée inattendu depuis apply_chat_template: {type(inputs)}")


def _resolve_image(image=None, image_path: Optional[str] = None):
    if image is not None:
        return image
    if image_path:
        from transformers.image_utils import load_image

        return load_image(image_path)
    return None


def _build_vl_conversation(history: List[Dict[str, Any]], image) -> List[Dict[str, Any]]:
    if not history:
        content = [{"type": "image", "image": image}]
        return [{"role": "user", "content": content}]

    conversation = list(history[:-1])
    last = history[-1]
    last_text = last.get("content", "")
    if isinstance(last_text, list):
        last_text = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in last_text
        ).strip()

    content = [{"type": "image", "image": image}]
    if last_text:
        content.append({"type": "text", "text": last_text})
    conversation.append({"role": last.get("role", "user"), "content": content})
    return conversation


def _generate_gguf(
    loaded: LoadedModel,
    history: List[Dict[str, Any]],
    settings: Dict[str, Any],
    on_token: TokenCallback,
) -> Tuple[str, int]:
    messages = as_text_messages(history)
    temperature = float(settings.get("temperature", 0.3))
    min_p = float(settings.get("min_p", 0.15))
    repeat_penalty = float(settings.get("repetition_penalty", 1.05))
    max_tokens = int(settings.get("max_new_tokens", 512))

    kwargs = {
        "messages": messages,
        "temperature": max(temperature, 0.0),
        "top_p": 1.0,
        "repeat_penalty": repeat_penalty,
        "max_tokens": max_tokens,
        "stream": bool(on_token),
    }
    if min_p > 0:
        kwargs["min_p"] = min_p

    try:
        output = loaded.model.create_chat_completion(**kwargs)
    except TypeError:
        kwargs.pop("min_p", None)
        output = loaded.model.create_chat_completion(**kwargs)

    if kwargs["stream"]:
        chunks = []
        for chunk in output:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content")
            if token:
                chunks.append(token)
                on_token(token)
        result = "".join(chunks)
    else:
        result = output["choices"][0]["message"]["content"] or ""

    if result:
        try:
            num_tokens = len(loaded.model.tokenize(result.encode("utf-8")))
        except Exception:
            num_tokens = max(1, len(result.split()))
    else:
        num_tokens = 0
    return result, num_tokens


def _generate_transformers(
    loaded: LoadedModel,
    history: List[Dict[str, Any]],
    settings: Dict[str, Any],
    image=None,
    image_path: Optional[str] = None,
    on_token: TokenCallback = None,
) -> Tuple[str, int]:
    import torch

    resolved_image = _resolve_image(image=image, image_path=image_path)
    if loaded.kind == "vl" and resolved_image is not None:
        conversation = _build_vl_conversation(history, resolved_image)
    else:
        conversation = as_text_messages(history)

    device = infer_model_device(loaded.model)
    inputs = _apply_chat_template(loaded.processor, conversation, device)
    sampling = build_sampling_kwargs(settings)

    # Le streaming Transformers n'est fiable que sur les modèles texte.
    if on_token and loaded.kind == "text":
        sampling["streamer"] = CallbackStreamer(loaded.processor, on_token, skip_prompt=True)

    with torch.no_grad():
        outputs = loaded.model.generate(**inputs, **sampling)

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_len:]
    result = loaded.processor.decode(new_tokens, skip_special_tokens=True)
    return result, int(new_tokens.shape[-1]) if hasattr(new_tokens, "shape") else len(new_tokens)


def generate(
    loaded: LoadedModel,
    conversation_history: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
    image=None,
    image_path: Optional[str] = None,
    on_token: TokenCallback = None,
) -> Tuple[str, float]:
    """
    Génère une réponse.

    Retourne (texte, tokens_par_seconde).
    """
    settings = settings or {}
    started = time.time()

    if loaded.kind == "gguf":
        text, num_tokens = _generate_gguf(loaded, conversation_history, settings, on_token)
    else:
        text, num_tokens = _generate_transformers(
            loaded,
            conversation_history,
            settings,
            image=image,
            image_path=image_path,
            on_token=on_token,
        )

    duration = time.time() - started
    tokens_per_sec = (num_tokens / duration) if duration > 0 else 0.0
    return text, tokens_per_sec
