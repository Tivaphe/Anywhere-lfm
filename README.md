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

### Prérequis Essentiels

-   **Python 3.8+**
-   **Git**
-   **(Windows Uniquement) Support des chemins longs activé :** L'installation de `transformers` depuis la source peut échouer sur Windows si cette option n'est pas activée.
    1.  Ouvrez l'Éditeur de Stratégie de Groupe : `Win + R`, tapez `gpedit.msc`.
    2.  Naviguez vers `Configuration ordinateur > Modèles d'administration > Système > Système de fichiers`.
    3.  Double-cliquez sur `Activer la prise en charge des chemins d'accès longs Win32` et mettez-le sur `Activé`.
    4.  Redémarrez votre ordinateur.

### Instructions par Système d'Exploitation

L'installation est gérée par des scripts qui tentent de s'adapter à votre système.

#### Pour Windows

1.  **Exécutez `install.bat`**
    -   Double-cliquez sur le fichier. Le script installera toutes les dépendances nécessaires.
    -   Il tentera d'installer `llama-cpp-python` pour le support des modèles GGUF.

2.  **Lancez l'application**
    -   Double-cliquez sur `run.bat`.

#### Pour macOS et Linux

1.  **Rendez les scripts exécutables** (une seule fois) :
    ```bash
    chmod +x install.sh run.sh
    ```

2.  **Exécutez le script d'installation** :
    ```bash
    ./install.sh
    ```

3.  **Lancez l'application** :
    ```bash
    ./run.sh
    ```

### En cas d'échec de l'installation de `llama-cpp-python`

`llama-cpp-python` est la seule dépendance complexe car elle nécessite une compilation.

-   **Que se passe-t-il si l'installation échoue ?**
    -   Le script d'installation affichera un **AVERTISSEMENT**, mais **ne s'arrêtera pas**.
    -   **L'application sera toujours fonctionnelle** pour les modèles standards de Hugging Face (ceux marqués `[HF]`).
    -   Seule la possibilité de charger des modèles locaux `.gguf` sera désactivée.

-   **Comment résoudre le problème ?**
    -   La cause la plus fréquente est l'absence d'un compilateur C++. Assurez-vous d'en avoir un :
        -   **Windows** : Installez les **Build Tools for Visual Studio** (via le "Visual Studio Installer", cochez la charge de travail "Développement desktop en C++").
        -   **macOS** : Installez les **Command Line Tools for Xcode** en tapant `xcode-select --install` dans un terminal.
        -   **Linux** : Installez `build-essential` (ex: `sudo apt-get install build-essential` sur Debian/Ubuntu).
    -   Après avoir installé le compilateur, relancez simplement le script `install.bat` ou `install.sh`.

## Comment l'utiliser

1.  **(Optionnel) Ajoutez vos propres modèles**
    -   Placez vos fichiers de modèle `.gguf` dans le dossier `models`.

2.  **Lancez l'application** (`run.bat` ou `./run.sh`).

3.  **Utilisation de l'interface**
    -   **Actualiser** : Scanne le dossier `models` et met à jour la liste des modèles.
    -   **Sélectionnez un modèle** : `[HF]` = Hugging Face (téléchargé), `[Local]` = votre fichier GGUF.
    -   **Paramètres** : Ouvre une fenêtre pour régler le prompt système, la température, etc.
    -   **Discutez** !
