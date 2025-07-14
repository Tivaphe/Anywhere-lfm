import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QComboBox, QLabel,
                             QHBoxLayout)
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ModelWorker(QThread):
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
            self.model_loaded.emit(tokenizer, model)
        except Exception as e:
            self.error.emit(str(e))

class GenerationWorker(QThread):
    generation_complete = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, tokenizer, model, conversation_history):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.conversation_history = conversation_history

    def run(self):
        try:
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
                    temperature=0.3,
                    min_p=0.15,
                    repetition_penalty=1.05
                )

            # Décode seulement les nouveaux tokens générés
            new_tokens = outputs[0][input_ids.shape[-1]:]
            result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            self.generation_complete.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class LiquidAIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidAI Chat")
        self.setGeometry(100, 100, 700, 500)
        self.conversation_history = []
        self.init_ui()
        self.load_model(self.model_selector.currentText())

    def init_ui(self):
        layout = QVBoxLayout()

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        layout.addWidget(self.chat_area)

        model_layout = QHBoxLayout()
        model_label = QLabel("Modèle:")
        self.model_selector = QComboBox()
        self.model_selector.addItems(["LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B"])
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        layout.addLayout(model_layout)

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

    def on_model_change(self, model_name):
        self.chat_area.clear()
        self.conversation_history = []
        self.load_model(model_name)

    def load_model(self, model_name):
        self.chat_area.append(f"<i>Chargement du modèle {model_name}... Veuillez patienter.</i>")
        self.set_ui_enabled(False)
        self.model_worker = ModelWorker(model_name)
        self.model_worker.model_loaded.connect(self.on_model_loaded)
        self.model_worker.error.connect(self.on_error)
        self.model_worker.start()

    def on_model_loaded(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.chat_area.append(f"<i>Modèle {self.model.config._name_or_path} chargé.</i>")
        self.set_ui_enabled(True)
        # Ajout du prompt système au début de la conversation
        self.conversation_history.append({"role": "system", "content": "You are a helpful assistant trained by Liquid AI."})


    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message or not self.model:
            return

        self.chat_area.append(f"<b>Vous:</b> {user_message}")
        self.input_field.clear()
        self.conversation_history.append({"role": "user", "content": user_message})

        self.chat_area.append("<i>L'IA est en train d'écrire...</i>")
        self.set_ui_enabled(False)

        self.generation_worker = GenerationWorker(self.tokenizer, self.model, self.conversation_history)
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
    app = QApplication(sys.argv)
    window = LiquidAIApp()
    window.show()
    sys.exit(app.exec())
