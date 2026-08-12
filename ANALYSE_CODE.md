# Analyse de code — Anywhere-LFM (LiquidAI Chat)

**Dépôt :** [Tivaphe/Anywhere-lfm](https://github.com/Tivaphe/Anywhere-lfm)  
**Périmètre :** `main.py`, `api.py`, `settings.py`, scripts d’install/lancement, `requirements.txt`, README  
**Taille :** ~1 100 lignes Python, 3 modules, pas de tests  
**Date de revue :** 12 août 2026  
**Statut :** les correctifs P0/P1 listés en §10 ont été appliqués (module `core/`, `min_p` réel, pipeline VL, install Unix, API locale, settings persistés).

---

## 1. De quoi il s’agit

Anywhere-LFM est une appli locale pour parler aux modèles **LiquidAI LFM2** (texte, vision, GGUF) sans ligne de commande. Deux faces :

| Face | Entrée | Techno |
|------|--------|--------|
| Bureau | `main.py` via `run.sh` / `run.bat` | PyQt6 + workers QThread |
| API « OpenAI-like » | `api.py` via `run_api.sh` | FastAPI + uvicorn, `0.0.0.0:8000` |

Fonctionnalités réellement présentes dans le code :

- chat texte avec historique JSON persisté ;
- modèles VL (image + texte) ;
- GGUF via `llama-cpp-python` ;
- RAG (PDF / TXT / DOCX → embeddings MiniLM → FAISS) ;
- paramètres de génération (température, « Min P », repetition penalty, prompt système) ;
- streaming token-par-token (texte Transformers + GGUF seulement) ;
- éjection du modèle pour libérer la VRAM.

Verdict global : **prototype fonctionnel et ambitieux**, mais tout est concentré dans deux gros fichiers, avec de la duplication, quelques bugs de génération réels, et une install Linux incomplète.

---

## 2. Architecture actuelle

```
┌──────────────────┐     ┌──────────────────┐
│   main.py GUI    │     │     api.py       │
│  LiquidAIApp     │     │   FastAPI        │
│  ModelWorker     │     │   get_model()    │
│  GenerationWorker│     │   /v1/chat/...   │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         │   logique dupliquée    │
         │   (load GGUF / VL /    │
         │    texte, generate)    │
         ▼                        ▼
   transformers + torch + llama_cpp + huggingface_hub
```

Il n’y a **pas de couche métier partagée**. La liste des modèles, le chargement (HF / GGUF / VL) et la génération sont copiés-collés entre `main.py` et `api.py`. `settings.py` n’est qu’un dialogue Qt.

Conséquence : tout correctif (nouveau modèle, `min_p`, VL `return_dict`, unload) doit être fait deux fois, et les deux chemins ont déjà divergé.

---

## 3. Tour des fichiers

### 3.1 `main.py` (~794 lignes) — cœur de l’appli

Trois responsabilités mélangées dans un seul module :

1. **Workers**
   - `ModelWorker` : charge Transformers, VL ou GGUF dans un `QThread`.
   - `PyQtStreamer` : `TextStreamer` + `QObject` pour pousser les tokens vers l’UI.
   - `GenerationWorker` : trois branches (`llama.cpp` / VL / texte).

2. **UI** (`LiquidAIApp`) : historique, chat HTML, sélecteur de modèle, zone image, RAG, paramètres.

3. **Persistance** : `conversations/<uuid>.json`, copie des docs dans `documents/`.

Points bien vus :

- chargement et génération hors thread UI ;
- signal `status_update` pendant le download GGUF ;
- `eject_model()` + `torch.cuda.empty_cache()` ;
- RAG optionnel, pas imposé.

Points faibles structurels :

- classe dieu (~500 lignes) ;
- commentaires « CORRECTION 1 / 2 » laissés dans le code ;
- détection de type de modèle par sous-chaîne `"VL"` / `"GGUF"` dans le nom ;
- pas d’arrêt de génération, pas de multi-ligne, pas de thème.

### 3.2 `api.py` (~241 lignes) — serveur local

- Schémas Pydantic proches d’OpenAI (`ChatCompletionRequest`, etc.).
- Cache global `model_cache` (jamais évincé).
- Précharge `LiquidAI/LFM2-1.2B` au startup.
- Endpoint unique `POST /v1/chat/completions` + `GET /`.

Compatibilité OpenAI **partielle** : pas de `/v1/models`, pas de `stream`, pas de `max_tokens` exposé, `generate()` synchrone dans un handler `async` (bloque la boucle).

### 3.3 `settings.py` (~72 lignes)

Dialogue propre et lisible. Limites :

- réglages **non persistés** (perdus à la fermeture) ;
- pas de validation `chunk_overlap < chunk_size` ;
- le champ s’appelle « Min P » mais n’est **pas** branché sur `min_p` (voir §5).

### 3.4 Scripts et dépendances

| Fichier | Rôle | Problème |
|---------|------|----------|
| `install.bat` | venv + `pip install -r requirements.txt` + lance l’app | OK, le plus abouti |
| `install.sh` | venv + **liste hardcodée** de paquets | **n’utilise pas** `requirements.txt` ; rate RAG, GGUF, Pillow, etc. |
| `run.sh` / `run.bat` | active le venv, lance `main.py` | OK |
| `run_api.sh` | lance `api.py` | pas de garde si le venv n’existe pas |
| `requirements.txt` | pin `transformers` sur un commit git | `install.sh` installe `transformers@main` (instable) |
| `.gitignore` | pyc + `documents/` | manque `venv/`, `models/`, `offload/`, `conversations/`, `__pycache__` partiel |
| README | clair, en français | promet un auto-lancement Linux que `install.sh` ne fait pas ; badge MIT sans fichier `LICENSE` |

---

## 4. Flux critiques

### Chargement de modèle (GUI)

`refresh_model_list()` → `on_model_change()` → `load_model()` → `ModelWorker`.

Effet de bord important : **au démarrage, le premier modèle de la liste (`LFM2-350M`) est chargé automatiquement**. Changer l’item du combo relance un download/load immédiat, même pour « juste regarder » la liste. Sur CPU (cas de la capture d’écran), c’est long et bloquant côté UX (`set_ui_enabled(False)`).

### Génération texte

`apply_chat_template(..., return_tensors="pt")` → `model.generate(..., streamer=PyQtStreamer)`.

Le streamer émet des tokens ; en fin de run, `display_current_conversation()` **réécrit tout le HTML** (markdown via `markdown2`). D’où un léger flash : d’abord texte brut streamé, puis re-rendu.

### Génération VL

L’image n’est injectée que dans **le dernier** message user. L’historique sauvegardé ne contient **que du texte**. Un second tour VL n’a plus l’image précédente → le multi-tour vision est cassé.

De plus, l’appel officiel LiquidAI / Transformers pour LFM2-VL est :

```python
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)
model.generate(**inputs, ...)
```

Le code GUI fait `return_tensors="pt"` **sans** `return_dict=True` / `tokenize=True`, puis `dict(**inputs, ...)`.  
L’API fait la même chose et passe `inputs` en **positionnel** à `generate()`.  
Sans `pixel_values`, un modèle VL ne « voit » pas l’image, ou lève une exception.

### RAG

1. Copie des fichiers vers `documents/`.
2. Recharge **tous** les fichiers du dossier (pas incrémental).
3. `RecursiveCharacterTextSplitter` + `HuggingFaceEmbeddings('all-MiniLM-L6-v2')` + FAISS **en mémoire**, **sur le thread UI** (`QApplication.processEvents()` pour faire semblant de rester fluide).
4. À l’envoi : `similarity_search(..., k=3)` concaténé dans le dernier message user.

Le README dit « modèles texte uniquement » ; le code n’interdit pas le RAG sur un VL.

---

## 5. Bugs fonctionnels (par gravité)

### Critique

**B1 — `min_p` n’est pas `min_p`.**  
LiquidAI recommande `temperature=0.3`, `min_p=0.15`, `repetition_penalty=1.05`. L’UI expose bien ces valeurs par défaut, mais `GenerationWorker` les mappe ainsi :

```python
top_p=self.settings["min_p"] if self.settings["min_p"] > 0 else None
```

`min_p` et `top_p` sont deux samplers différents. Le réglage affiché « Min P » pilote en réalité le nucleus sampling. Les modèles LFM2 ne tournent donc **pas** avec les hyperparamètres officiels. Même erreur côté GGUF (`top_p=...min_p`). L’API n’envoie ni `min_p` ni `repetition_penalty`.

**B2 — Pipeline VL incomplet.**  
Manque `tokenize=True` + `return_dict=True`. L’API décode ensuite avec `processor.decode(outputs[0][len(inputs[0]):], ...)` : si `inputs` est un `BatchEncoding`, `len(inputs[0])` n’est pas la longueur des `input_ids`. Risque d’erreur ou de texte tronqué / pollué par le prompt.

**B3 — `install.sh` ne rend pas l’app utilisable.**  
N’installe pas `langchain*`, `sentence-transformers`, `faiss-cpu`, `pypdf`, `python-docx`, `llama-cpp-python`, `huggingface-hub`, `Pillow`, `torchvision`. Un utilisateur macOS/Linux qui suit le README cassera dès le RAG, le GGUF, ou même l’import de `main.py`.

### Important

**B4 — Handler FastAPI async + `model.generate()` synchrone.**  
Une requête bloque tout le serveur (pas de second client, pas de `GET /` pendant la gen).

**B5 — Cache API sans unload.**  
Chaque modèle demandé reste en RAM/VRAM. Enchaîner 1.2B + VL-3B + GGUF = OOM quasi certain. Pas d’endpoint pour éjecter.

**B6 — Génération bloquante / non interruptible.**  
Pas de bouton Stop. `max_new_tokens=512` partout, non configurable. `do_sample=True` même si température = 0.

**B7 — Injection HTML dans le chat.**  
Les messages user sont interpolés bruts dans du HTML Qt :

```python
html = f"<p style='margin: 0;'>{content}</p>"
```

Un message contenant `<b>`, `<img>` ou des balises casse le rendu (et le contexte RAG affiché aussi).

**B8 — Historique VL amnésique.**  
Seuls les textes sont sérialisés. Impossible de rejouer une conversation image.

**B9 — `on_model_change` trop agressif.**  
Clear du chat + load immédiat. Mauvaise UX, et double chargement possible si l’utilisateur clique vite.

### Mineur / dette

- Typo UI : « L’IA réfléchi... » → « réfléchit ».
- Titres d’historique : `msg['content'][:30] + "..."` même si le texte est plus court.
- `save_conversations()` réécrit **tous** les JSON à chaque message.
- `msg.dict()` et `@app.on_event("startup")` : APIs Pydantic v2 / FastAPI dépréciées.
- `local_dir_use_symlinks=False` déprécié dans `huggingface_hub`.
- `HuggingFaceEmbeddings` (langchain-community) est le chemin déprécié.
- `check_device()` ignore MPS (Apple Silicon) et ROCm.
- Couleurs `#4f4f4f` sans couleur de texte : illisible si le thème OS est clair.
- Saisie `QLineEdit` : pas de messages multi-lignes.
- Précharge API du 1.2B même si le client demandera un autre modèle.
- Capture d’écran de 686 Ko commitée à la racine.

---

## 6. Sécurité

Ce n’est pas un service exposé publiquement *par design*, mais `api.py` écoute sur **toutes les interfaces** sans auth.

| Risque | Détail |
|--------|--------|
| Accès réseau ouvert | `uvicorn.run(..., host="0.0.0.0", port=8000)` — n’importe qui sur le LAN peut faire tourner des modèles (CPU/GPU hijack). |
| `trust_remote_code=True` | Exécute le code custom du repo HF. OK pour LiquidAI officiel, dangereux si la liste de modèles s’ouvre. |
| SSRF | `load_image(url)` côté API suit n’importe quelle URL. |
| Pas de limite | Ni `max_tokens` client réellement borné côté politique, ni rate limit, ni taille d’image. |
| Données locales | `conversations/` et `documents/` ne sont pas ignorés par git (seul `documents/` l’est). Risque de commit de chats privés. |

Pour un usage desktop 100 % local, le risque majeur est surtout **l’API bind 0.0.0.0 sans token**.

---

## 7. Qualité de code

| Critère | État |
|---------|------|
| Modularité | Faible — 2 fichiers font tout |
| DRY | Mauvaise — load + generate ×2 |
| Types | Quasi absents hors Pydantic API |
| Tests | Aucun |
| Gestion d’erreurs | GUI : traceback dans le chat (bien pour debug, brut pour l’utilisateur). API : 500 générique. |
| Threads | Intention correcte (QThread). RAG et embeddings restent sur l’UI. Streamer Qt depuis le thread generate : ça passe via queued signals, mais c’est fragile. |
| Config | Valeurs magiques (`512`, `2048`, `k=3`, `n_gpu_layers=-1`) |
| i18n | UI en français, prompt système en anglais, OK |
| Licence | Badge MIT, pas de `LICENSE` |

Style : lisible, noms clairs, peu de « clever code ». On sent une appli qui a grandi par PRs successives (RAG, VL, GGUF, eject) sans refonte.

---

## 8. Écarts README ↔ code

1. Linux : install incomplète + pas de lancement auto.  
2. « Streaming » : vrai pour texte/GGUF GUI ; faux pour VL et pour l’API (`400` si `stream=true`).  
3. « API compatible OpenAI » : squelette seulement.  
4. RAG « texte uniquement » : non enforce.  
5. Python 3.8+ : trop optimiste (PyQt6 + torch + transformers git).  
6. Licence MIT : fichier manquant.

---

## 9. Ce qui est bien

- Objectif produit clair : rendre LFM2 utilisable « anywhere », sans CLI.
- Scripts Windows soignés (messages d’erreur, pause, vérif PyQt6).
- Séparation load / generate en threads dans la GUI.
- Support de trois backends (Transformers, VL, llama.cpp) dans un même sélecteur.
- Historique disque simple, suppression via menu contextuel.
- README pédagogique (exemples curl texte + vision).
- Défauts de sampling (0.3 / 0.15 / 1.05) alignés sur la doc LiquidAI — même s’ils sont mal branchés.

---

## 10. Refactor recommandé (par priorité)

### P0 — faire marcher ce qui est promis

1. **Brancher `min_p` pour de vrai** dans GUI et API (`model.generate(..., min_p=...)`, et l’équivalent llama.cpp si dispo ; ne plus le mapper sur `top_p`).
2. **Corriger le chemin VL** : `tokenize=True`, `return_dict=True`, `model.generate(**inputs)`.
3. **`install.sh` → `pip install -r requirements.txt`**, comme le `.bat`.
4. Ajouter `LICENSE` (MIT) et ignorer `venv/`, `models/`, `offload/`, `conversations/`.

### P1 — assainir l’archi

5. Extraire un module `core/` :
   - `models.py` : registre, load, unload, cache unique ;
   - `generate.py` : une fonction de génération (sync + générateur stream) ;
   - `rag.py` : index / search.
   GUI et API ne font plus que de l’I/O.
6. Ne plus charger un modèle au simple changement de combo : bouton « Charger ».
7. `run_in_executor` (ou worker thread) dans FastAPI ; `host=127.0.0.1` par défaut + token optionnel.
8. Persister `settings.json`. Échapper le HTML user (`html.escape`).

### P2 — produit

9. Stop génération, `max_new_tokens` dans les settings, saisie multi-ligne.
10. Stream SSE côté API (`text/event-stream` façon OpenAI).
11. Persister les images des tours VL (chemin ou miniature) pour le multi-tour.
12. RAG dans un worker, index incremental, bouton « vider les documents ».
13. Détection device : CUDA / MPS / CPU.
14. Tests smoke : parsing des schémas API, mapping des settings, splitter RAG.

Cible raisonnable après P0+P1 :

```
core/models.py
core/generate.py
core/rag.py
core/config.py
ui/app.py
ui/settings.py
ui/workers.py
api/server.py
```

---

## 11. Synthèse

Anywhere-LFM est une **bonne maquette produit** : une GUI PyQt + une API locale autour des LFM2, avec RAG et VL greffés au fil de l’eau. Le code se lit facilement et les intentions sont les bonnes (threads, cache, eject, scripts Windows).

En l’état ce n’est **pas encore une base saine** :

- la logique modèle est dupliquée et déjà divergente ;
- le paramètre phare `min_p` est mal câblé ;
- le chemin vision ne suit pas l’API Transformers officielle ;
- l’install Unix est cassée par rapport au README ;
- l’API bloque, n’évince rien, et écoute sur toutes les interfaces.

Corriger P0 (min_p, VL, install) rendrait l’outil réellement aligné avec ce qu’il annonce. Extraire un `core` ensuite éviterait que la prochaine feature (tools, nouveau modèle, stream API) double encore la dette.
