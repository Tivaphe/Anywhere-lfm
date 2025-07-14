import sys
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QComboBox, QLabel,
                             QHBoxLayout, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import torch

from settings import SettingsWindow

# --- Workers pour le chargement et la génération en arrière-plan ---

class TransformersModelWorker(QThread):
    model_loaded = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto"
            )
            self.model_loaded.emit(model, tokenizer)
        except Exception as e:
            self.error.emit(str(e))

class GGUFModelWorker(QThread):
    model_loaded = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            # Pour les modèles GGUF, le "tokenizer" est intégré dans l'objet Llama
            model = Llama(model_path=self.model_path, n_ctx=2048, n_gpu_layers=-1) # -1 pour tout mettre sur le GPU
            self.model_loaded.emit(model, None) # Pas de tokenizer séparé
        except Exception as e:
            self.error.emit(str(e))

class GenerationWorker(QThread):
    generation_complete = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, conversation_history, settings):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = conversation_history
        self.settings = settings

    def run(self):
        try:
            # Inférence avec Transformers
            if isinstance(self.model, PreTrainedModel):
                input_ids = self.tokenizer.apply_chat_template(
                    self.conversation_history,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=self.settings["temperature"],
                        top_p=self.settings["min_p"] if self.settings["min_p"] > 0 else None, # top_p et min_p sont souvent la même chose
                        repetition_penalty=self.settings["repetition_penalty"]
                    )

                new_tokens = outputs[0][input_ids.shape[-1]:]
                result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Inférence avec Llama.cpp
            elif isinstance(self.model, Llama):
                result_iterator = self.model.create_chat_completion(
                    messages=self.conversation_history,
                    temperature=self.settings["temperature"],
                    top_p=self.settings["min_p"],
                    repeat_penalty=self.settings["repetition_penalty"],
                    stream=False # Pour l'instant, on ne gère pas le streaming
                )
                result = result_iterator['choices'][0]['message']['content']

            self.generation_complete.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# --- Application Principale ---

class LiquidAIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidAI Chat")
        self.setGeometry(100, 100, 700, 500)

        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.settings = {
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.3,
            "min_p": 0.15,
            "repetition_penalty": 1.05
        }

        self.init_ui()
        self.check_device()
        self.refresh_model_list()

    def init_ui(self):
        layout = QVBoxLayout()

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)

        # Zone de contrôle des modèles
        model_controls_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        refresh_button = QPushButton("Actualiser")
        refresh_button.clicked.connect(self.refresh_model_list)
        settings_button = QPushButton("Paramètres")
        settings_button.clicked.connect(self.open_settings)
        model_controls_layout.addWidget(QLabel("Modèle:"))
        model_controls_layout.addWidget(self.model_selector)
        model_controls_layout.addWidget(refresh_button)
        model_controls_layout.addWidget(settings_button)
        layout.addLayout(model_controls_layout)

        # Zone de saisie
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Posez votre question ici...")
        self.input_field.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Envoyer")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)
        self.setLayout(layout)

    def check_device(self):
        if torch.cuda.is_available():
            device = f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            device = "CPU"
        self.chat_area.append(f"<i>Utilisation de l'appareil : {device}</i>")

    def refresh_model_list(self):
        self.model_selector.clear()

        # Modèles téléchargeables
        huggingface_models = ["LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B"]
        self.model_selector.addItems([f"[HF] {name}" for name in huggingface_models])

        # Modèles locaux
        models_dir = "models"
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.lower().endswith(".gguf"):
                    self.model_selector.addItem(f"[Local] {filename}")

    def open_settings(self):
        dialog = SettingsWindow(self)
        dialog.set_settings(self.settings)
        if dialog.exec():
            self.settings = dialog.get_settings()
            self.chat_area.append("<i>Paramètres mis à jour.</i>")
            # Re-initialiser la conversation avec le nouveau prompt système
            self.on_model_change(self.model_selector.currentText())

    def on_model_change(self, model_identifier):
        if not model_identifier:
            return
        self.chat_area.clear()
        self.check_device() # Rappeler l'état du device
        self.load_model(model_identifier)

    def load_model(self, model_identifier):
        self.set_ui_enabled(False)
        self.chat_area.append(f"<i>Chargement du modèle {model_identifier}... Veuillez patienter.</i>")

        if model_identifier.startswith("[HF]"):
            model_name = model_identifier.replace("[HF] ", "")
            self.worker = TransformersModelWorker(model_name)
        elif model_identifier.startswith("[Local]"):
            model_name = model_identifier.replace("[Local] ", "")
            model_path = os.path.join("models", model_name)
            self.worker = GGUFModelWorker(model_path)
        else:
            self.on_error("Type de modèle non reconnu.")
            return

        self.worker.model_loaded.connect(self.on_model_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_model_loaded(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        model_name = self.model.config._name_or_path if hasattr(self.model, 'config') else self.model_selector.currentText()
        self.chat_area.append(f"<i>Modèle {model_name} chargé.</i>")
        self.set_ui_enabled(True)

        self.conversation_history = []
        if self.settings["system_prompt"]:
            self.conversation_history.append({"role": "system", "content": self.settings["system_prompt"]})

    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message or not self.model:
            return

        self.chat_area.append(f"<b>Vous:</b> {user_message}")
        self.input_field.clear()
        self.conversation_history.append({"role": "user", "content": user_message})

        self.chat_area.append("<i>L'IA est en train d'écrire...</i>")
        self.set_ui_enabled(False)

        self.generation_worker = GenerationWorker(self.model, self.tokenizer, self.conversation_history, self.settings)
        self.generation_worker.generation_complete.connect(self.on_generation_complete)
        self.generation_worker.error.connect(self.on_error)
        self.generation_worker.start()

    def on_generation_complete(self, response):
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self.chat_area.setTextCursor(cursor)

        self.chat_area.append(f"<b>LiquidAI:</b> {response}")
        self.conversation_history.append({"role": "assistant", "content": response})
        self.set_ui_enabled(True)
        self.input_field.setFocus()

    def on_error(self, error_message):
        self.chat_area.append(f"<font color='red'>Erreur : {error_message}</font>")
        self.set_ui_enabled(True)

    def set_ui_enabled(self, enabled):
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.model_selector.setEnabled(enabled)

if __name__ == "__main__":
    # Correction pour un bug potentiel avec Llama.cpp et les threads sur certains systèmes
    from llama_cpp.llama import K_QUANTS
    K_QUANTS.update({
        "IQ2_XXS": 17, "IQ2_XS": 18, "IQ2_S": 19, "IQ2_M": 20,
        "IQ1_S": 21, "IQ1_M": 22, "IQ3_XXS": 23, "IQ3_S": 24, "IQ3_M": 25,
        "IQ4_XS": 26, "IQ4_NL": 27
    })

    app = QApplication(sys.argv)
    window = LiquidAIApp()
    window.show()
    sys.exit(app.exec())
