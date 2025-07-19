from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Charger le tokenizer et le modèle
# Remplacer "microsoft/DialoGPT-medium" par le modèle LFM2 de votre choix
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models")

# Route pour servir la page principale du chatbot
@app.route('/')
def index():
    return render_template('index.html')

# Route pour l'API de chat
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')

    # Encoder l'entrée et générer une réponse
    input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
