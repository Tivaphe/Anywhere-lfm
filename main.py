import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QComboBox, QLabel,
                             QHBoxLayout)
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
import torch

# Définition de l'architecture LFM2 (simplifiée)
class LFM2Config(PretrainedConfig):
    model_type = "lfm2"

class LFM2Model(PreTrainedModel):
    config_class = LFM2Config
    def __init__(self, config):
        super().__init__(config)
        # Ici, on aurait normalement les couches du modèle (embeddings, décodeurs, etc.)
        # Pour le chargement, une définition minimale suffit.
        self.dummy_layer = torch.nn.Linear(1, 1)

# Enregistrement de l'architecture LFM2
CONFIG_MAPPING.register("lfm2", LFM2Config)
if isinstance(AutoModelForCausalLM, _BaseAutoModelClass):
    AutoModelForCausalLM.register(LFM2Config, LFM2Model)


class ModelWorker(QThread):
    model_loaded = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
            self.model_loaded.emit(tokenizer, model)
        except Exception as e:
            self.error.emit(str(e))

class GenerationWorker(QThread):
    generation_complete = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, tokenizer, model, text):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.text = text

    def run(self):
        try:
            inputs = self.tokenizer(self.text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.generation_complete.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class LiquidAIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidAI Chat")
        self.setGeometry(100, 100, 600, 400)
        self.init_ui()

        self.tokenizer = None
        self.model = None
        self.load_model("LiquidAI/LFM2-350M") # Charger le plus petit modèle par défaut

    def init_ui(self):
        layout = QVBoxLayout()

        # Zone de discussion
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)

        # Sélection du modèle
        model_layout = QHBoxLayout()
        model_label = QLabel("Modèle:")
        self.model_selector = QComboBox()
        self.model_selector.addItems(["LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B"])
        self.model_selector.currentTextChanged.connect(self.load_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        layout.addLayout(model_layout)

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

    def load_model(self, model_name):
        self.chat_area.append(f"<i>Chargement du modèle {model_name}...</i>")
        self.model_worker = ModelWorker(model_name)
        self.model_worker.model_loaded.connect(self.on_model_loaded)
        self.model_worker.error.connect(self.on_error)
        self.model_worker.start()

    def on_model_loaded(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.chat_area.append(f"<i>Modèle {self.model.config._name_or_path} chargé.</i>")

    def send_message(self):
        user_message = self.input_field.text()
        if not user_message or not self.model:
            return

        self.chat_area.append(f"<b>Vous:</b> {user_message}")
        self.input_field.clear()
        self.chat_area.append("<i>L'IA est en train d'écrire...</i>")

        self.generation_worker = GenerationWorker(self.tokenizer, self.model, user_message)
        self.generation_worker.generation_complete.connect(self.on_generation_complete)
        self.generation_worker.error.connect(self.on_error)
        self.generation_worker.start()

    def on_generation_complete(self, response):
        # Enlever le message "L'IA est en train d'écrire..."
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar() # Enlever le retour à la ligne
        self.chat_area.setTextCursor(cursor)

        self.chat_area.append(f"<b>LiquidAI:</b> {response}")

    def on_error(self, error_message):
        self.chat_area.append(f"<font color='red'>Erreur : {error_message}</font>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiquidAIApp()
    window.show()
    sys.exit(app.exec())
