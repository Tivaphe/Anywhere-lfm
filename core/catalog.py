"""Découverte automatique des modèles LiquidAI / LFM sur Hugging Face."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

HF_AUTHOR = "LiquidAI"
HF_API = "https://huggingface.co/api/models"
CACHE_PATH = os.path.join("models", "catalog.json")
CACHE_TTL_SECONDS = 12 * 60 * 60
USER_AGENT = "Anywhere-LFM/1.0"

PREFERRED_QUANTS = ("Q4_K_M", "Q5_K_M", "Q4_0", "Q4_K_S", "Q6_K", "Q8_0", "Q5_0", "Q3_K_M")

EXCLUDE_TOKENS = (
    "MLX",
    "ONNX",
    "OPENVINO",
    "WEBGPU",
    "LEAPBUNDLES",
    "ENCODER",
    "ENCODERS",
)

FALLBACK_REPOS: List[str] = [
    # LFM2 texte
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
    # LFM2.5 texte
    "LiquidAI/LFM2.5-350M",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "LiquidAI/LFM2.5-1.2B-Thinking",
    "LiquidAI/LFM2.5-2.6B",
    # Vision
    "LiquidAI/LFM2-VL-3B",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-450M",
    # GGUF LFM2
    "LiquidAI/LFM2-350M-GGUF",
    "LiquidAI/LFM2-700M-GGUF",
    "LiquidAI/LFM2-1.2B-GGUF",
    "LiquidAI/LFM2-2.6B-GGUF",
    "LiquidAI/LFM2-2.6B-Exp-GGUF",
    "LiquidAI/LFM2-8B-A1B-GGUF",
    "LiquidAI/LFM2-24B-A2B-GGUF",
    "LiquidAI/LFM2-350M-Extract-GGUF",
    "LiquidAI/LFM2-1.2B-Extract-GGUF",
    "LiquidAI/LFM2-1.2B-RAG-GGUF",
    "LiquidAI/LFM2-1.2B-Tool-GGUF",
    "LiquidAI/LFM2-350M-Math-GGUF",
    "LiquidAI/LFM2-2.6B-Transcript-GGUF",
    # GGUF LFM2.5
    "LiquidAI/LFM2.5-230M-GGUF",
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
    "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
    "LiquidAI/LFM2.5-2.6B-GGUF",
]

StatusCallback = Optional[Callable[[str], None]]


@dataclass
class ModelEntry:
    repo_id: str
    kind: str
    downloads: int = 0
    last_modified: str = ""
    pipeline_tag: str = ""
    gguf_files: List[str] = field(default_factory=list)

    @property
    def short_name(self) -> str:
        return self.repo_id.split("/", 1)[-1]

    @property
    def label(self) -> str:
        prefix = {"gguf": "GGUF", "vl": "VL", "text": "TXT"}.get(self.kind, self.kind.upper())
        return f"[{prefix}] {self.short_name}"


@dataclass
class Catalog:
    models: List[ModelEntry] = field(default_factory=list)
    updated_at: float = 0.0
    source: str = "fallback"

    def find(self, repo_id: str) -> Optional[ModelEntry]:
        for entry in self.models:
            if entry.repo_id == repo_id:
                return entry
        return None

    def repo_ids(self) -> List[str]:
        return [entry.repo_id for entry in self.models]

    def is_stale(self, ttl: int = CACHE_TTL_SECONDS) -> bool:
        if not self.updated_at or self.source == "fallback":
            return True
        return (time.time() - self.updated_at) > ttl


def parse_model_ref(ref: str) -> Tuple[str, Optional[str]]:
    """Sépare `LiquidAI/repo` et un éventuel fichier / quant (`:Q4_K_M` ou `:file.gguf`)."""
    value = (ref or "").strip()
    if not value:
        return "", None
    if value.count("/") >= 1 and ":" in value:
        repo, spec = value.split(":", 1)
        spec = spec.strip() or None
        return repo.strip(), spec
    return value, None


def is_lfm_repo(repo_id: str) -> bool:
    if not repo_id.startswith(f"{HF_AUTHOR}/"):
        return False
    name = repo_id.split("/", 1)[-1]
    return name.upper().startswith("LFM")


def is_excluded_repo(
    repo_id: str,
    tags: Optional[Iterable[str]] = None,
    library_name: Optional[str] = None,
) -> bool:
    name = repo_id.split("/", 1)[-1]
    haystack = name.upper()
    if any(token in haystack for token in EXCLUDE_TOKENS):
        return True
    if name.endswith("-Base") or name.endswith("-base"):
        return True
    lib = (library_name or "").lower()
    if lib in {"mlx", "onnx", "openvino"}:
        return True
    tags_l = {str(tag).lower() for tag in (tags or [])}
    if tags_l & {"mlx", "onnx", "openvino"} and "gguf" not in tags_l:
        return True
    return False


def is_allowed_model(ref: str) -> bool:
    repo, _ = parse_model_ref(ref)
    return is_lfm_repo(repo) and not is_excluded_repo(repo)


def classify_kind(
    repo_id: str,
    tags: Optional[Iterable[str]] = None,
    pipeline_tag: Optional[str] = None,
    library_name: Optional[str] = None,
    files: Optional[Iterable[str]] = None,
) -> str:
    tags_l = {str(tag).lower() for tag in (tags or [])}
    name = repo_id.upper()
    file_list = list(files or [])
    has_gguf = any(str(item).lower().endswith(".gguf") for item in file_list)
    if (
        "GGUF" in name
        or "gguf" in tags_l
        or (library_name or "").lower() == "gguf"
        or has_gguf
    ):
        return "gguf"
    short = repo_id.split("/", 1)[-1].upper()
    if "VL" in short or pipeline_tag == "image-text-to-text":
        return "vl"
    return "text"


def pick_gguf_file(files: List[str], preferred: Optional[str] = None) -> str:
    if not files:
        raise FileNotFoundError("Aucun fichier .gguf dans ce dépôt.")
    if preferred:
        needle = preferred.lower()
        for name in files:
            if name.lower() == needle or name.lower().endswith("/" + needle):
                return name
        for name in files:
            stem = os.path.basename(name).lower()
            if needle in stem or needle.replace(".gguf", "") in stem:
                return name
    ranked = {quant.upper(): index for index, quant in enumerate(PREFERRED_QUANTS)}

    def score(name: str) -> Tuple[int, str]:
        upper = os.path.basename(name).upper()
        for quant, index in ranked.items():
            if quant in upper:
                return index, upper
        return len(ranked) + 1, upper

    return sorted(files, key=score)[0]


def gguf_quant_label(filename: str) -> str:
    stem = os.path.basename(filename)
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    upper = stem.upper()
    for quant in list(PREFERRED_QUANTS) + ("BF16", "F16", "F32", "Q4_1", "IQ4_XS"):
        if quant in upper:
            if quant == "Q4_K_M":
                return f"{quant} (recommandé)"
            return quant
    return stem


def fallback_entries() -> List[ModelEntry]:
    entries = []
    for repo_id in FALLBACK_REPOS:
        entries.append(
            ModelEntry(
                repo_id=repo_id,
                kind=classify_kind(repo_id),
            )
        )
    return _sort_entries(entries)


def fallback_repo_ids() -> List[str]:
    return list(FALLBACK_REPOS)


def _sort_entries(entries: List[ModelEntry]) -> List[ModelEntry]:
    order = {"text": 0, "vl": 1, "gguf": 2}
    return sorted(
        entries,
        key=lambda item: (order.get(item.kind, 9), item.short_name.lower()),
    )


def _emit(callback: StatusCallback, message: str) -> None:
    if callback:
        callback(message)


def _sibling_names(payload: Dict[str, Any]) -> List[str]:
    names = []
    for sibling in payload.get("siblings") or []:
        if isinstance(sibling, str):
            names.append(sibling)
        elif isinstance(sibling, dict):
            name = sibling.get("rfilename") or sibling.get("filename")
            if name:
                names.append(name)
    return names


def _http_get_json(url: str, timeout: int = 45) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_list_models() -> List[Dict[str, Any]]:
    base = f"{HF_API}?author={HF_AUTHOR}&limit=500&sort=lastModified&direction=-1"
    errors = []
    for url in (f"{base}&expand=siblings", base):
        try:
            payload = _http_get_json(url)
            if isinstance(payload, list):
                return payload
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
    raise RuntimeError(f"API Hugging Face injoignable ({'; '.join(errors) or 'erreur inconnue'})")


def _hub_list_models() -> List[Dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        infos = list(api.list_models(author=HF_AUTHOR, expand=["siblings"]))
    except TypeError:
        infos = list(api.list_models(author=HF_AUTHOR))
    payloads = []
    for info in infos:
        siblings = []
        raw_siblings = getattr(info, "siblings", None) or []
        for sibling in raw_siblings:
            name = getattr(sibling, "rfilename", None) or str(sibling)
            if name:
                siblings.append(name)
        payloads.append(
            {
                "id": info.id,
                "tags": list(getattr(info, "tags", None) or []),
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "library_name": getattr(info, "library_name", None),
                "downloads": getattr(info, "downloads", 0) or 0,
                "lastModified": str(getattr(info, "last_modified", "") or ""),
                "siblings": siblings,
            }
        )
    return payloads


def fetch_model_payloads() -> List[Dict[str, Any]]:
    try:
        return _hub_list_models()
    except Exception:
        return _http_list_models()


def list_gguf_files(repo_id: str) -> List[str]:
    try:
        from huggingface_hub import list_repo_files

        files = list(list_repo_files(repo_id))
    except Exception:
        try:
            payload = _http_get_json(f"{HF_API}/{repo_id}")
            files = _sibling_names(payload) if isinstance(payload, dict) else []
        except Exception as exc:
            raise FileNotFoundError(f"Impossible de lister les fichiers de {repo_id}: {exc}") from exc
    return sorted(name for name in files if str(name).lower().endswith(".gguf"))


def entry_from_payload(payload: Dict[str, Any]) -> Optional[ModelEntry]:
    repo_id = payload.get("id") or payload.get("modelId") or ""
    if not is_lfm_repo(repo_id):
        return None
    tags = payload.get("tags") or []
    library_name = payload.get("library_name")
    if is_excluded_repo(repo_id, tags, library_name):
        return None
    files = [name for name in _sibling_names(payload) if str(name).lower().endswith(".gguf")]
    kind = classify_kind(
        repo_id,
        tags=tags,
        pipeline_tag=payload.get("pipeline_tag"),
        library_name=library_name,
        files=files,
    )
    return ModelEntry(
        repo_id=repo_id,
        kind=kind,
        downloads=int(payload.get("downloads") or 0),
        last_modified=str(payload.get("lastModified") or payload.get("last_modified") or ""),
        pipeline_tag=str(payload.get("pipeline_tag") or ""),
        gguf_files=files,
    )


def refresh_catalog(
    status_callback: StatusCallback = None,
    fetch_missing_gguf_files: bool = True,
) -> Catalog:
    _emit(status_callback, "Interrogation du Hub Hugging Face (LiquidAI)...")
    payloads = fetch_model_payloads()
    entries: List[ModelEntry] = []
    for payload in payloads:
        entry = entry_from_payload(payload)
        if entry:
            entries.append(entry)

    if fetch_missing_gguf_files:
        missing = [entry for entry in entries if entry.kind == "gguf" and not entry.gguf_files]
        if missing:
            _emit(status_callback, f"Lecture des fichiers GGUF ({len(missing)} dépôts)...")
            for entry in missing:
                try:
                    entry.gguf_files = list_gguf_files(entry.repo_id)
                except Exception:
                    entry.gguf_files = []

    if not entries:
        raise RuntimeError("Aucun modèle LFM n'a été trouvé sur Hugging Face.")

    catalog = Catalog(
        models=_sort_entries(entries),
        updated_at=time.time(),
        source="huggingface",
    )
    save_catalog(catalog)
    _emit(
        status_callback,
        f"Catalogue mis à jour : {len(catalog.models)} modèles "
        f"({sum(1 for item in catalog.models if item.kind == 'gguf')} GGUF).",
    )
    return catalog


def save_catalog(catalog: Catalog, path: str = CACHE_PATH) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "updated_at": catalog.updated_at,
        "source": catalog.source,
        "models": [asdict(entry) for entry in catalog.models],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_cached_catalog(path: str = CACHE_PATH) -> Optional[Catalog]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        models = [ModelEntry(**item) for item in payload.get("models") or []]
        if not models:
            return None
        return Catalog(
            models=_sort_entries(models),
            updated_at=float(payload.get("updated_at") or 0),
            source=str(payload.get("source") or "cache"),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def get_catalog(prefer_cache: bool = True) -> Catalog:
    if prefer_cache:
        cached = load_cached_catalog()
        if cached:
            return cached
    return Catalog(models=fallback_entries(), updated_at=0.0, source="fallback")


def get_cached_repo_ids() -> List[str]:
    catalog = get_catalog(prefer_cache=True)
    return catalog.repo_ids()
