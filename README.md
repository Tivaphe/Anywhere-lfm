<img width="1442" height="841" alt="Capture d'écran 2025-07-15 204919" src="https://github.com/user-attachments/assets/885a78a4-9102-4d1c-83fb-607610486a95" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Application de Chat LiquidAI avec Transformers

Une application de bureau et une API locale pour interagir avec les modèles LFM2 de LiquidAI, optimisée par la bibliothèque `transformers`.

## Fonctionnalités Principales

- **Interface Graphique Intuitive** : Une application de bureau simple et efficace construite avec PyQt6 pour interagir avec les modèles.
- **Gestion Dynamique des Modèles** : Chargez et déchargez les modèles de langue à la volée pour libérer les ressources (VRAM/RAM).
- **Support RAG (Retrieval-Augmented Generation)** : Améliorez les réponses du modèle en lui fournissant le contexte de vos propres documents (`.txt`, `.pdf`, `.docx`).
- **API Compatible OpenAI** : Exposez le modèle via une API locale qui imite la structure de l'API OpenAI, vous permettant de connecter vos outils et scripts existants.
- **Streaming de Texte** : Obtenez des réponses en temps réel, mot par mot, pour une expérience plus fluide.
- **Paramètres Personnalisables** : Ajustez finement les paramètres de génération comme la température, le `min_p` et la pénalité de répétition via une interface dédiée.
- **Historique des Conversations** : Toutes vos discussions sont sauvegardées localement et peuvent être rechargées.

## Prérequis

- **Python 3.8+**
- **Git**

### ⚠️ Prérequis Important pour les Utilisateurs Windows

Pour que l'installation des dépendances fonctionne correctement, vous **DEVEZ** activer le support des chemins de fichiers longs sur votre système. C'est une opération unique et sans danger.

1.  **Ouvrez PowerShell en tant qu'administrateur** et exécutez la commande suivante :
    ```powershell
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ```
2.  **Redémarrez votre ordinateur**. C'est une étape cruciale pour que le changement soit pris en compte.

## Installation

Les scripts d'installation créent un environnement virtuel et installent toutes les dépendances nécessaires.

#### Pour Windows

1.  **Première fois :** Double-cliquez sur `install.bat`. L'application se lancera automatiquement après l'installation.
2.  **Les fois suivantes :** Double-cliquez sur `run.bat` pour un démarrage rapide.

#### Pour macOS et Linux

1.  **Première fois :**
    *   Rendez les scripts exécutables : `chmod +x install.sh run.sh run_api.sh`.
    *   Exécutez le script d'installation : `./install.sh`. L'application se lancera automatiquement.
2.  **Les fois suivantes :** Exécutez `./run.sh`.

## Comment Utiliser

Ce projet offre deux modes d'utilisation principaux : une interface de bureau et une API.

### 1. Interface Graphique de Bureau (GUI)

Lancez l'application en utilisant `run.bat` (Windows) ou `./run.sh` (macOS/Linux).

- **Panneau de Gauche (Historique)** :
  - Affiche toutes vos conversations passées.
  - Cliquez sur "Nouvelle Discussion" pour en commencer une nouvelle.
  - Faites un clic droit sur une conversation pour la supprimer.

- **Panneau de Droite (Chat)** :
  - **Sélection de Modèle** : Choisissez un modèle dans la liste déroulante. L'application le téléchargera (si nécessaire) et le chargera en mémoire.
  - **Éjecter le Modèle** : Cliquez sur ce bouton pour décharger le modèle de la VRAM/RAM et libérer les ressources.
  - **Paramètres** : Ouvre une fenêtre pour ajuster le *prompt système*, la *température*, le *min_p*, et d'autres options de génération.
  - **Zone de Chat** : Affiche la conversation en cours.
  - **Champ de Saisie** : Tapez votre message et appuyez sur Entrée ou cliquez sur "Envoyer".

### 2. Fonctionnalité RAG (Retrieval-Augmented Generation)

La fonctionnalité RAG permet au modèle de répondre à des questions sur des informations contenues dans vos documents personnels.

1.  **Charger des documents** : Cliquez sur le bouton "Charger Documents" pour sélectionner un ou plusieurs fichiers (`.txt`, `.pdf`, `.docx`).
2.  **Création de l'index** : L'application traite ces documents, les découpe en morceaux, et les transforme en vecteurs numériques à l'aide d'un modèle d'embedding local. Ces vecteurs sont stockés dans un index en mémoire (FAISS).
3.  **Activer le RAG** : Cochez la case "Activer RAG".
4.  **Posez votre question** : Lorsque le RAG est actif, l'application recherche les morceaux de documents les plus pertinents et les injecte dans le contexte avant d'interroger le modèle.

### 3. API Compatible OpenAI

Lancez le serveur d'API avec `run_api.bat` (Windows) ou `./run_api.sh` (macOS/Linux). Le serveur démarrera sur `http://localhost:8000`.

Vous pouvez maintenant utiliser cette URL dans n'importe quel client ou bibliothèque compatible avec l'API OpenAI.

**Exemple avec `curl` :**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "LiquidAI/LFM2-350M",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explique le concept de RAG en une phrase."}
  ]
}'
```

**Réponse attendue :**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "LiquidAI/LFM2-350M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Le RAG (Retrieval-Augmented Generation) est une technique où un modèle de langue récupère des informations pertinentes dans une base de connaissances externe avant de générer une réponse."
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Dépendances

Les dépendances sont listées dans le fichier `requirements.txt` et sont installées automatiquement. Les principales bibliothèques utilisées sont :

- **GUI** : `PyQt6`
- **Modèles IA** : `transformers`, `torch`, `accelerate`
- **API** : `fastapi`, `uvicorn`
- **RAG** : `langchain`, `langchain-community`, `sentence-transformers`, `faiss-cpu`, `pypdf`, `python-docx`
- **Autres** : `markdown2`
