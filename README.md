# Application d'inférence de modèle LiquidAI

Cette application vous permet d'utiliser les modèles LiquidAI (LFM2-1.2B, LFM2-700M, LFM2-350M) via une interface utilisateur simple dans un notebook Jupyter ou via une API REST.

## Configuration

1.  **Installez Python** : Assurez-vous d'avoir Python 3.7 ou une version ultérieure installée.
2.  **Clonez le référentiel** : `git clone <url-du-repo>`
3.  **Créez un environnement virtuel** (recommandé) :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows, utilisez `venv\\Scripts\\activate`
    ```
4.  **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation de l'interface utilisateur du notebook Jupyter

1.  **Démarrez le serveur Jupyter Notebook** :
    ```bash
    jupyter notebook
    ```
2.  **Ouvrez `liquid_ai_ui.ipynb`** dans votre navigateur.
3.  **Sélectionnez un modèle** dans le menu déroulant.
4.  **Entrez votre texte** dans la zone de texte.
5.  **Cliquez sur le bouton "Générer"** pour voir la sortie du modèle.

## Utilisation de l'API

1.  **Démarrez le serveur Flask** :
    ```bash
    python api.py
    ```
    Le serveur démarrera sur `http://localhost:5000`.

2.  **Envoyez une requête POST** à l'endpoint `/generate`.

    **Exemple avec `curl`** :
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"text": "Bonjour, quel est ton nom?"}' http://localhost:5000/generate
    ```

    **Pour utiliser un modèle différent**, ajoutez le champ `model` au corps de la requête :
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"text": "Bonjour, quel est ton nom?", "model": "LiquidAI/LFM2-1.2B"}' http://localhost:5000/generate
    ```

    **Réponse** :
    ```json
    {
      "generated_text": "La sortie générée par le modèle"
    }
    ```

## Fichiers du projet

-   `requirements.txt`: Contient les dépendances Python nécessaires.
-   `model_loader.py`: Script pour charger les modèles et les tokenizers.
-   `liquid_ai_ui.ipynb`: Notebook Jupyter pour l'interface utilisateur interactive.
-   `api.py`: Application Flask pour l'API REST.
-   `README.md`: Ce fichier.
