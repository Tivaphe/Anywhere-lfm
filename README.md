<img width="1442" height="841" alt="Capture d'écran 2025-07-15 204919" src="https://github.com/user-attachments/assets/885a78a4-9102-4d1c-83fb-607610486a95" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Application de Chat LiquidAI avec Transformers

Une application de bureau simple pour discuter avec les modèles d'IA de LiquidAI, optimisée pour la bibliothèque `transformers`.

## ⚠️ Prérequis Important pour les Utilisateurs Windows

Pour que l'installation fonctionne, vous **DEVEZ** activer le support des chemins de fichiers longs sur votre système. C'est une opération unique et sans danger.

1.  **Ouvrez l'Éditeur de Stratégie de Groupe** :
    -   Appuyez sur `Win + R` (la touche Windows et la lettre R en même temps).
    -   Tapez `gpedit.msc` et appuyez sur Entrée.
    -   *Si cela ne fonctionne pas (versions Famille de Windows), ouvrez PowerShell en tant qu'administrateur et collez cette commande :*
        ```powershell
        New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
        ```

2.  **Naviguez dans l'Éditeur de Stratégie** (si `gpedit.msc` a fonctionné) :
    -   Allez à `Configuration ordinateur > Modèles d'administration > Système > Système de fichiers`.

3.  **Activez le support** :
    -   Dans le volet de droite, double-cliquez sur `Activer la prise en charge des chemins d'accès longs Win32`.
    -   Sélectionnez `Activé` et cliquez sur `OK`.

4.  **Redémarrez votre ordinateur**. Cette étape est cruciale pour que le changement soit pris en compte.

## Installation

### Autres Prérequis

-   **Python 3.8+**
-   **Git**

### Instructions d'Installation

#### Pour Windows

1.  Assurez-vous d'avoir suivi le prérequis important ci-dessus.
2.  Double-cliquez sur `install.bat`.

#### Pour macOS et Linux

1.  Rendez les scripts exécutables (une seule fois) : `chmod +x install.sh run.sh run_api.sh`
2.  Exécutez le script d'installation : `./install.sh`

## Deux Modes d'Utilisation

Ce projet peut être utilisé de deux manières : via une interface graphique de bureau, ou via une API compatible avec OpenAI.

### 1. Interface Graphique de Bureau

#### Lancement
-   **Windows** : Double-cliquez sur `run.bat`.
-   **macOS / Linux** : Exécutez `./run.sh`.

#### Utilisation
1.  **Lancez l'application**.
2.  **Sélectionnez un modèle** dans la liste.
3.  **(Optionnel) Paramètres** : Cliquez sur "Paramètres" pour ajuster le prompt système et les options de génération.
4.  **Discutez** !

### 2. API Compatible OpenAI

#### Lancement
-   **Windows** : Double-cliquez sur `run_api.bat`.
-   **macOS / Linux** : Exécutez `./run_api.sh`.

Le serveur démarrera sur `http://localhost:8000`.

#### Utilisation

Vous pouvez maintenant utiliser ce serveur comme un remplacement direct de l'API OpenAI dans vos outils et scripts. Le point de terminaison principal est `/v1/chat/completions`.

**Exemple avec `curl` :**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "LiquidAI/LFM2-350M",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is C. elegans?"}
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
        "content": "C. elegans, or Caenorhabditis elegans, is a type of nematode worm..."
      },
      "finish_reason": "stop"
    }
  ]
}
```
