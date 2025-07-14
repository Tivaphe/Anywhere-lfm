# Application de Chat LiquidAI

Une application de bureau simple pour discuter avec les modèles d'IA de LiquidAI, supportant les modèles Transformers et GGUF.

## Fonctionnalités

-   Interface graphique simple et conviviale.
-   Support des modèles Transformers (via Hugging Face) et GGUF (via llama-cpp).
-   Chargement de modèles locaux depuis un dossier `models`.
-   Fenêtre de paramètres pour personnaliser le prompt système et les hyperparamètres de génération.
-   Détection du matériel (CPU/GPU) pour une performance optimale.
-   Compatible avec Windows, macOS, et Linux.

## Installation

### Prérequis

-   **Python 3.8+**
-   **Git**
-   Un **compilateur C++** :
    -   **Windows** : Les "Build Tools for Visual Studio" sont généralement requis.
    -   **macOS** : Les "Command Line Tools for Xcode" sont nécessaires (`xcode-select --install`).
    -   **Linux** : Un paquet comme `build-essential` (sur Debian/Ubuntu) est nécessaire.

### Instructions par Système d'Exploitation

#### Pour Windows

1.  **Exécutez `install.bat`**
    -   Double-cliquez sur le fichier `install.bat`.
    -   Une fenêtre de terminal s'ouvrira et installera tout ce qui est nécessaire. Cela peut prendre un certain temps.
    -   Si l'installation de `llama-cpp-python` échoue, assurez-vous d'avoir les "Build Tools for Visual Studio" installées.
    -   Attendez que le message "Installation terminée avec succès!" apparaisse, puis appuyez sur n'importe quelle touche pour fermer la fenêtre.

2.  **Lancez l'application**
    -   Double-cliquez sur `run.bat`.

#### Pour macOS et Linux

1.  **Rendez les scripts exécutables** (si nécessaire) :
    ```bash
    chmod +x install.sh run.sh
    ```

2.  **Exécutez le script d'installation** :
    ```bash
    ./install.sh
    ```
    -   Le script détectera votre système pour tenter une compilation optimisée de `llama-cpp-python` (Metal pour macOS, CUDA pour Linux avec GPU NVIDIA).
    -   Suivez les instructions si des dépendances sont manquantes.

3.  **Lancez l'application** :
    ```bash
    ./run.sh
    ```

## Comment l'utiliser

1.  **(Optionnel) Ajoutez vos propres modèles**
    -   Placez vos fichiers de modèle `.gguf` dans le dossier `models` qui a été créé à la racine du projet.

2.  **Lancez l'application**
    -   Utilisez `run.bat` ou `./run.sh`.

3.  **Utilisation de l'interface**
    -   **Actualiser** : Cliquez sur le bouton "Actualiser" pour scanner le dossier `models` et mettre à jour la liste des modèles disponibles.
    -   **Sélectionnez un modèle** : Choisissez un modèle dans la liste. `[HF]` indique un modèle qui sera téléchargé depuis Hugging Face, et `[Local]` un modèle de votre dossier `models`.
    -   **Paramètres** : Cliquez sur "Paramètres" pour ajuster le prompt système, la température, et d'autres options de génération.
    -   **Discutez** : Tapez votre message et appuyez sur Entrée !
