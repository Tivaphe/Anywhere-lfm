import sys
import os
import json
import uuid
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QComboBox, QLabel, QMenu,
                             QHBoxLayout, QSplitter, QListWidget, QListWidgetItem,
                             QFileDialog, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, TextStreamer
import torch
import time
import markdown2

# RAG specific imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- CORRECTION 1: Mise à jour de l'import pour suivre l'avertissement ---
from langchain_community.vectorstores import FAISS # Ancien import: from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from settings import SettingsWindow

# --- Workers pour le chargement et la génération en arrière-plan ---
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
                self.model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto"
            )
            self.model_loaded.emit(model, tokenizer)
        except Exception as e:
            self.error.emit(f"Erreur Transformers : {e}")


class PyQtStreamer(TextStreamer, QObject):
    new_token = pyqtSignal(str)

    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        TextStreamer.__init__(self, tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        QObject.__init__(self)
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("PyQtStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        if self.print_len == 0:
            printable_text = text
        else:
            printable_text = text[self.print_len:]

        self.print_len = len(text)
        if printable_text:
            self.new_token.emit(printable_text)

    def end(self):
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        printable_text = text[self.print_len:]
        if printable_text:
            self.new_token.emit(printable_text)
        self.next_tokens_are_prompt = True
        self.token_cache = []
        self.print_len = 0

class GenerationWorker(QThread):
    generation_complete = pyqtSignal(str)
    new_token = pyqtSignal(str)
    error = pyqtSignal(str)
    stats = pyqtSignal(float)

    def __init__(self, model, tokenizer, conversation_history, settings):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = conversation_history
        self.settings = settings
        self.streamer = PyQtStreamer(self.tokenizer, skip_prompt=True)
        self.streamer.new_token.connect(self.new_token)

    def run(self):
        try:
            input_ids = self.tokenizer.apply_chat_template(
                self.conversation_history, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            generation_kwargs = dict(
                input_ids=input_ids,
                streamer=self.streamer,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.settings["temperature"],
                top_p=self.settings["min_p"] if self.settings["min_p"] > 0 else None,
                repetition_penalty=self.settings["repetition_penalty"]
            )

            start_time = time.time()
            outputs = self.model.generate(**generation_kwargs)
            end_time = time.time()

            new_tokens = outputs[0][input_ids.shape[-1]:]
            num_new_tokens = len(new_tokens)
            result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            duration = end_time - start_time
            tokens_per_sec = num_new_tokens / duration if duration > 0 else 0

            self.stats.emit(tokens_per_sec)
            self.generation_complete.emit(result)

        except Exception as e:
            self.error.emit(f"Erreur de génération : {e}")

# --- Application Principale ---

class LiquidAIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidAI Chat")
        self.setGeometry(100, 100, 1000, 700)

        self.model = None
        self.tokenizer = None
        self.current_conversation_id = None
        self.conversations = {}
        self.settings = {
            "system_prompt": "You are a helpful assistant.", "temperature": 0.3,
            "min_p": 0.15, "repetition_penalty": 1.05,
            "rag_chunk_size": 500, "rag_chunk_overlap": 50
        }
        # RAG attributes
        self.rag_enabled = False
        self.vector_store = None
        self.rag_documents_path = "documents/"
        os.makedirs(self.rag_documents_path, exist_ok=True)
        self.current_assistant_message = ""

        self.init_ui()
        self.load_conversations()
        self.check_device()
        self.refresh_model_list()
        if not self.conversations:
            self.start_new_conversation()
        else:
            self.history_list.setCurrentRow(0)
            self.load_selected_conversation(self.history_list.item(0))


    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Panneau de gauche (Historique) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.history_list = QListWidget(left_panel)
        self.history_list.itemClicked.connect(self.load_selected_conversation)
        self.history_list.itemDoubleClicked.connect(self.rename_conversation_item)
        self.history_list.itemChanged.connect(self.on_conversation_renamed)
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self.show_conversation_context_menu)
        left_layout.addWidget(self.history_list)
        new_chat_button = QPushButton("Nouvelle Discussion")
        new_chat_button.clicked.connect(self.start_new_conversation)
        left_layout.addWidget(new_chat_button)
        
        # --- CORRECTION 2: Le code ci-dessous a été ré-indenté pour faire partie de la fonction init_ui ---

        # --- Panneau de droite (Chat) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("font-size: 14px;")
        right_layout.addWidget(self.chat_area)

        # --- Contrôles du modèle ---
        model_controls_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        settings_button = QPushButton("Paramètres")
        settings_button.clicked.connect(self.open_settings)
        self.eject_button = QPushButton("Ejecter")
        self.eject_button.clicked.connect(self.eject_model)
        self.eject_button.setEnabled(False)
        model_controls_layout.addWidget(QLabel("Modèle:"))
        model_controls_layout.addWidget(self.model_selector)
        model_controls_layout.addWidget(self.eject_button)
        model_controls_layout.addWidget(settings_button)
        right_layout.addLayout(model_controls_layout)

        # --- Zone de saisie utilisateur ---
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setStyleSheet("font-size: 14px;")
        self.input_field.setPlaceholderText("Posez votre question ici...")
        self.input_field.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Envoyer")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        right_layout.addLayout(input_layout)

        # --- RAG Controls ---
        rag_layout = QHBoxLayout()
        self.load_docs_button = QPushButton("Charger Documents")
        self.load_docs_button.clicked.connect(self.load_documents)
        self.rag_toggle_checkbox = QCheckBox("Activer RAG")
        self.rag_toggle_checkbox.stateChanged.connect(self.toggle_rag)
        self.rag_status_label = QLabel("RAG: Inactif - Aucun document chargé")
        self.stats_label = QLabel("")
        rag_layout.addWidget(self.load_docs_button)
        rag_layout.addWidget(self.rag_toggle_checkbox)
        rag_layout.addWidget(self.rag_status_label)
        rag_layout.addStretch()
        rag_layout.addWidget(self.stats_label)
        right_layout.addLayout(rag_layout)

        # --- Splitter pour séparer les panneaux ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 750])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def show_conversation_context_menu(self, position):
        item = self.history_list.itemAt(position)
        if not item:
            return

        context_menu = QMenu(self)
        delete_action = context_menu.addAction("Supprimer")
        action = context_menu.exec(self.history_list.mapToGlobal(position))

        if action == delete_action:
            self.delete_conversation(item)

    def delete_conversation(self, item):
        conv_id = item.data(Qt.ItemDataRole.UserRole)

        file_path = f"conversations/{conv_id}.json"
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                self.on_error(f"Impossible de supprimer le fichier {file_path}: {e}")
                return

        if conv_id in self.conversations:
            del self.conversations[conv_id]

        row = self.history_list.row(item)
        self.history_list.takeItem(row)

        if self.current_conversation_id == conv_id:
            if self.history_list.count() > 0:
                first_item = self.history_list.item(0)
                self.history_list.setCurrentItem(first_item)
                self.load_selected_conversation(first_item)
            else:
                self.start_new_conversation()

    def check_device(self):
        device = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        self.chat_area.append(f"<i>Utilisation de l'appareil : {device}</i>")

    def refresh_model_list(self):
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        huggingface_models = ["LiquidAI/LFM2-350M", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B"]
        self.model_selector.addItems(huggingface_models)
        self.model_selector.blockSignals(False)
        self.on_model_change(self.model_selector.currentText())

    def open_settings(self):
        dialog = SettingsWindow(self)
        dialog.set_settings(self.settings)
        if dialog.exec():
            self.settings = dialog.get_settings()
            self.chat_area.append("<i>Paramètres mis à jour.</i>")
            if self.current_conversation_id:
                self.conversations[self.current_conversation_id][0] = {"role": "system", "content": self.settings["system_prompt"]}
                self.display_current_conversation()

    def on_model_change(self, model_identifier):
        if not model_identifier: return
        self.chat_area.clear()
        self.check_device()
        self.load_model(model_identifier)

    def load_model(self, model_identifier):
        self.set_ui_enabled(False)
        self.chat_area.append(f"<i>Chargement du modèle {model_identifier}...</i>")
        self.worker = ModelWorker(model_identifier)
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_model_loaded(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        model_name = self.model_selector.currentText()
        if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
            model_name = model.config._name_or_path
        self.chat_area.append(f"<i>Modèle {model_name} chargé.</i>")
        self.set_ui_enabled(True)
        self.eject_button.setEnabled(True)
        if self.current_conversation_id:
            self.display_current_conversation()

    def eject_model(self):
        if self.model is None:
            return
        model_name = self.model_selector.currentText()
        self.chat_area.append(f"<i>Déchargement du modèle {model_name}...</i>")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.chat_area.append("<i>Cache GPU vidé.</i>")
        self.chat_area.append("<i>Modèle déchargé. Sélectionnez un modèle pour commencer.</i>")
        self.eject_button.setEnabled(False)
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)

    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message or not self.model or not self.current_conversation_id: return

        rag_context = ""
        if self.rag_enabled and self.vector_store:
            try:
                docs = self.vector_store.similarity_search(user_message, k=3)
                rag_context = "\n\nContexte des documents:\n" + "\n---\n".join([doc.page_content for doc in docs])
                self.chat_area.append(f"<i>Contexte RAG trouvé:\n{rag_context}</i>")
            except Exception as e:
                self.on_error(f"Erreur de recherche RAG: {e}")

        self.append_message("user", user_message, save=True)
        self.input_field.clear()

        self.chat_area.append("<i>L'IA réfléchi...</i>")
        self.set_ui_enabled(False)

        conversation_history = list(self.conversations[self.current_conversation_id])
        if rag_context:
            conversation_history[-1]["content"] = f"{rag_context}\n\nQuestion: {user_message}"

        self.current_assistant_message = ""
        self.stats_label.setText("")
        self.generation_worker = GenerationWorker(self.model, self.tokenizer, conversation_history, self.settings)
        self.generation_worker.new_token.connect(self.on_new_token)
        self.generation_worker.generation_complete.connect(self.on_generation_complete)
        self.generation_worker.stats.connect(self.on_stats_update)
        self.generation_worker.error.connect(self.on_error)
        self.generation_worker.start()

    def on_new_token(self, token):
        if not self.current_assistant_message:
            cursor = self.chat_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.select(cursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
            self.chat_area.setTextCursor(cursor)
            self.append_message("assistant", token, save=False)
        else:
            cursor = self.chat_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(token)
        self.current_assistant_message += token

    def on_stats_update(self, tokens_per_sec):
        self.stats_label.setText(f"{tokens_per_sec:.2f} tokens/s")

    def on_generation_complete(self, response):
        self.conversations[self.current_conversation_id].append({"role": "assistant", "content": response})
        self.save_conversations()
        self.display_current_conversation()
        self.set_ui_enabled(True)
        self.input_field.setFocus()
        self.current_assistant_message = ""

    def on_error(self, error_message):
        self.chat_area.append(f"<font color='red'>Erreur : {error_message}</font>")
        self.set_ui_enabled(True)

    def set_ui_enabled(self, enabled):
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.model_selector.setEnabled(enabled)
        self.history_list.setEnabled(enabled)
        self.load_docs_button.setEnabled(enabled)
        self.rag_toggle_checkbox.setEnabled(enabled and self.vector_store is not None)

    def toggle_rag(self, state):
        self.rag_enabled = (state == Qt.CheckState.Checked.value)
        if self.rag_enabled:
            self.rag_status_label.setText("RAG: Actif")
            self.chat_area.append("<i>RAG activé. Les documents chargés seront utilisés comme contexte.</i>")
        else:
            self.rag_status_label.setText("RAG: Inactif")
            self.chat_area.append("<i>RAG désactivé.</i>")

    def load_documents(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Sélectionner des documents", "", "Documents (*.txt *.pdf *.docx)"
        )
        if not files:
            return

        self.chat_area.append(f"<i>Chargement de {len(files)} document(s)...</i>")
        QApplication.processEvents()

        for file_path in files:
            filename = os.path.basename(file_path)
            destination = os.path.join(self.rag_documents_path, filename)
            try:
                with open(file_path, 'rb') as f_in, open(destination, 'wb') as f_out:
                    f_out.write(f_in.read())
            except Exception as e:
                self.on_error(f"Erreur lors de la copie du fichier {filename}: {e}")
                return

        self.chat_area.append("<i>Création de la base de données vectorielle...</i>")
        QApplication.processEvents()

        try:
            docs = []
            for filename in os.listdir(self.rag_documents_path):
                file_path = os.path.join(self.rag_documents_path, filename)
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                elif filename.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    docs.extend(loader.load())
                elif filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs.extend(loader.load())

            if not docs:
                self.on_error("Aucun document valide trouvé à traiter.")
                return

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings["rag_chunk_size"],
                chunk_overlap=self.settings["rag_chunk_overlap"]
            )
            splits = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
            self.vector_store = FAISS.from_documents(splits, embeddings)

            self.rag_status_label.setText("RAG: Prêt")
            self.rag_toggle_checkbox.setEnabled(True)
            self.chat_area.append("<i>Base de données vectorielle créée avec succès. Vous pouvez maintenant activer le RAG.</i>")

        except Exception as e:
            self.on_error(f"Erreur lors de la création de l'index RAG : {e}")
            self.rag_status_label.setText("RAG: Erreur")
            self.vector_store = None
            self.rag_toggle_checkbox.setEnabled(False)

    def rename_conversation_item(self, item):
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.history_list.editItem(item)

    def on_conversation_renamed(self, item):
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        if conv_id in self.conversations:
            new_title = item.text()
            self.conversations[conv_id]["title"] = new_title
            self.save_conversations()
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)


    def start_new_conversation(self):
        self.current_conversation_id = str(uuid.uuid4())
        new_title = f"Discussion du {time.strftime('%Y-%m-%d %H:%M')}"
        self.conversations[self.current_conversation_id] = {
            "title": new_title,
            "messages": [{"role": "system", "content": self.settings["system_prompt"]}]
        }

        item = QListWidgetItem(new_title)
        item.setData(Qt.ItemDataRole.UserRole, self.current_conversation_id)
        self.history_list.insertItem(0, item)
        self.history_list.setCurrentItem(item)

        self.display_current_conversation()

    def load_selected_conversation(self, item):
        self.current_conversation_id = item.data(Qt.ItemDataRole.UserRole)
        self.display_current_conversation()

    def display_current_conversation(self):
        self.chat_area.clear()
        self.check_device()
        if self.model:
            model_name = self.model_selector.currentText()
            if hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path'):
                model_name = self.model.config._name_or_path
            self.chat_area.append(f"<i>Modèle {model_name} chargé.</i>")

        if not self.current_conversation_id: return

        conversation_data = self.conversations.get(self.current_conversation_id, {})
        history = conversation_data.get("messages", [])

        # Afficher le message système en premier s'il existe
        system_prompt = next((msg["content"] for msg in history if msg["role"] == "system"), None)
        if system_prompt:
             self.append_message("system", "Prompt système : " + system_prompt, save=False)

        for message in history:
             if message["role"] != "system":
                self.append_message(message["role"], message["content"], save=False)

    def append_message(self, role, content, save=True):
        if not self.current_conversation_id: return

        if save:
            # Assurez-vous que la conversation a une structure pour stocker les messages
            if "messages" not in self.conversations[self.current_conversation_id]:
                self.conversations[self.current_conversation_id]["messages"] = []
            self.conversations[self.current_conversation_id]["messages"].append({"role": role, "content": content})


        if role == "user":
            # Alignement à droite pour l'utilisateur
            html = f"""
            <div style='margin-left: 50px; margin-right: 5px; background-color: #2b5278; padding: 10px; border-radius: 10px; margin-bottom: 5px;'>
                <p style='color: white; margin: 0;'>{content}</p>
            </div>
            """
        elif role == "assistant":
            # Alignement à gauche pour l'IA
            formatted_content = markdown2.markdown(content, extras=["fenced-code-blocks", "tables"])
            html = f"""
            <div style='margin-right: 50px; margin-left: 5px; background-color: #3a3a3a; padding: 10px; border-radius: 10px; margin-bottom: 5px;'>
                <b style='color: #e0e0e0;'>LiquidAI:</b>
                <div style='color: white;'>{formatted_content}</div>
            </div>
            """
        else: # system messages
            html = f"<div style='text-align: center; color: grey; margin-bottom: 5px;'><i>{content}</i></div>"

        self.chat_area.append(html)
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())


        if save:
            self.save_conversations()

    def save_conversations(self):
        os.makedirs("conversations", exist_ok=True)
        for conv_id, history in self.conversations.items():
            with open(f"conversations/{conv_id}.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    def load_conversations(self):
        if not os.path.exists("conversations"):
            return

        sorted_files = sorted(
            [f for f in os.listdir("conversations") if f.endswith(".json")],
            key=lambda x: os.path.getmtime(os.path.join("conversations", x)),
            reverse=True
        )

        for filename in sorted_files:
            conv_id = filename.replace(".json", "")
            try:
                with open(f"conversations/{filename}", "r", encoding="utf-8") as f:
                    conv_data = json.load(f)
                    self.conversations[conv_id] = conv_data

                # Gérer l'ancien et le nouveau format
                if isinstance(conv_data, list):
                     # Ancien format: une liste de messages
                    title = f"Ancienne Discussion {conv_id[:8]}"
                    user_message = next((msg['content'] for msg in conv_data if msg['role'] == 'user'), None)
                    if user_message:
                        title = user_message[:30] + "..."
                    self.conversations[conv_id] = {
                        "title": title,
                        "messages": conv_data
                    }
                else:
                    # Nouveau format: un dictionnaire avec "title" et "messages"
                    title = conv_data.get("title", f"Discussion {conv_id[:8]}")


                item = QListWidgetItem(title)
                item.setData(Qt.ItemDataRole.UserRole, conv_id)
                self.history_list.addItem(item)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Erreur de chargement ou format invalide pour {filename}: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiquidAIApp()
    window.show()
    sys.exit(app.exec())