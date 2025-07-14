from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name):
    """
    Charge un modèle et un tokenizer à partir de Hugging Face.
    """
    if model_name not in ["LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-350M"]:
        raise ValueError("Modèle non valide. Veuillez choisir parmi : LiquidAI/LFM2-1.2B, LiquidAI/LFM2-700M, LiquidAI/LFM2-350M")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

if __name__ == '__main__':
    # Exemple d'utilisation
    try:
        tokenizer, model = load_model("LiquidAI/LFM2-700M")
        print("Modèle LiquidAI/LFM2-700M chargé avec succès !")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
