from flask import Flask, request, jsonify
from model_loader import load_model
import torch

app = Flask(__name__)

# Charger le modèle par défaut au démarrage
try:
    tokenizer, model = load_model("LiquidAI/LFM2-700M")
    print("Modèle LiquidAI/LFM2-700M chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    tokenizer, model = None, None

@app.route('/generate', methods=['POST'])
def generate():
    if not model or not tokenizer:
        return jsonify({"error": "Modèle non chargé"}), 500

    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Le champ 'text' est manquant"}), 400

    text = data['text']
    model_name = data.get('model', "LiquidAI/LFM2-700M")

    try:
        # Recharger le modèle si un autre est demandé
        global tokenizer, model
        if model.config._name_or_path != model_name:
            tokenizer, model = load_model(model_name)
            print(f"Modèle {model_name} chargé.")

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"generated_text": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
