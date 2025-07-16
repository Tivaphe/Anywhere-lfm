import sys
import os
import json
import uuid
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QComboBox, QLabel,
                             QHBoxLayout, QSplitter, QListWidget, QListWidgetItem,
                             QFileDialog, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, TextStreamer
import torch
import time
import markdown2

# RAG specific imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # Updated import
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

from PyQt6.QtCore import QObject

class PyQtStreamer(TextStreamer, QObject):  # Swapped order: TextStreamer first, then QObject
    new_token = pyqtSignal(str)

    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        TextStreamer.__init__(self, tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        QObject.__init__(self)
        # Initialize additional attributes to match TextStreamer's logic
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens, decodes them, and emits the decoded text via signal.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("PyQtStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Extend token cache
        self.token_cache.extend(value.tolist())

        # Decode the current cache
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # Get the new printable text
        if self.print_len == 0:
            printable_text = text
        else:
            printable_text = text[self.print_len:]

        # Update print_len to the length of the decoded text (in characters)
        self.print_len = len(text)

        # Emit the new text chunk
        if printable_text:
            self.new_token.emit(printable_text)

    def end(self):
        """Finalize and emit any remaining text."""
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        printable_text = text[self.print_len:]
        if printable_text:
            self.new_token.emit(printable_text)
        self.next_tokens_are_prompt = True
        self.token_cache = []
        self.print_len = 0

class GenerationWorker(QThread):
    generation_complete = pyqtSignal(str) # Still needed for the full response
    new_token = pyqtSignal(str)
    error = pyqtSignal(str)
    stats = pyqtSignal(float) # For tokens/sec

    def __init__(self, model, tokenizer, conversation_history, settings):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = conversation_history
        self.settings = settings
        self.streamer = PyQtStreamer(self.tokenizer, skip_prompt=True)  # Enable skip_prompt if desired
        self.streamer.new_token.connect(self.new_token) # Forward the signal

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
            # Capture the outputs from the generation with streamer
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
        self.setGeometry(100, 100, 1000, 700) # Taille augmentée

        self.model = None
        self.tokenizer = None
        self.current_conversation_id = None
        self.conversations = {} # Stocke les historiques de conversation
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
        self.start_new_conversation()
        self.refresh_model_list()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Panneau de gauche (Historique) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.load_selected_conversation)
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self.show_conversation_context_menu)
        left_layout.addWidget(self.history_list)

        new_chat_button = QPushButton("Nouvelle Discussion")
        new_chat_button.clicked.connect(self.start_new_conversation)
        left_layout.addWidget(new_chat_button)

    def show_conversation_context_menu(self, position):
        # Récupérer l'élément sur lequel l'utilisateur a cliqué
        item = self.history_list.itemAt(position)
        if not item:
            return # Ne rien faire si le clic est dans le vide

        # Créer le menu
        context_menu = QMenu(self)

        # Créer l'action "Supprimer"
        delete_action = context_menu.addAction("Supprimer")

        # Exécuter le menu et attendre que l'utilisateur choisisse une action
        action = context_menu.exec(self.history_list.mapToGlobal(position))

        # Si l'utilisateur a cliqué sur "Supprimer"
        if action == delete_action:
            self.delete_conversation(item)

    def delete_conversation(self, item):
        # Récupérer l'ID de la conversation à partir de l'élément de la liste
        conv_id = item.data(Qt.ItemDataRole.UserRole)

        # --- Étape 1: Supprimer le fichier de sauvegarde ---
        file_path = f"conversations/{conv_id}.json"
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                self.on_error(f"Impossible de supprimer le fichier {file_path}: {e}")
                return

    # --- Étape 2: Supprimer l'entrée du dictionnaire interne ---
        if conv_id in self.conversations:
            del self.conversations[conv_id]

    # --- Étape 3: Retirer l'élément de la QListWidget ---
        row = self.history_list.row(item)
        self.history_list.takeItem(row)

    # --- Étape 4: Gérer le cas où on supprime la conversation active ---
        if self.current_conversation_id == conv_id:
            # Si la liste n'est pas vide, on charge la première conversation
            if self.history_list.count() > 0:
                first_item = self.history_list.item(0)
                self.history_list.setCurrentItem(first_item)
                self.load_selected_conversation(first_item)
            else:
                # Sinon, on en crée une nouvelle
                self.start_new_conversation()

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
        splitter.setSizes([250, 750])  # Tailles initiales

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.check_device()

    # ... (le reste du code sera ajouté dans les prochaines étapes)
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
                # Mettre à jour le prompt système de la conversation actuelle
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

        # Accès plus sûr au nom du modèle
        model_name = self.model_selector.currentText()
        if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
            model_name = model.config._name_or_path

        self.chat_area.append(f"<i>Modèle {model_name} chargé.</i>")
        self.set_ui_enabled(True)
        self.eject_button.setEnabled(True) # Activer le bouton Ejecter
        if self.current_conversation_id:
            self.display_current_conversation()

    def eject_model(self):
        if self.model is None:
            return

        model_name = self.model_selector.currentText()
        self.chat_area.append(f"<i>Déchargement du modèle {model_name}...</i>")

        # Supprimer les objets modèle et tokenizer
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        # Vider le cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.chat_area.append("<i>Cache GPU vidé.</i>")

        self.chat_area.append("<i>Modèle déchargé. Sélectionnez un modèle pour commencer.</i>")
        self.eject_button.setEnabled(False) # Désactiver après l'éjection
        self.input_field.setEnabled(False) # Désactiver la saisie
        self.send_button.setEnabled(False)

    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message or not self.model or not self.current_conversation_id: return

        # RAG Logic
        rag_context = ""
        if self.rag_enabled and self.vector_store:
            try:
                docs = self.vector_store.similarity_search(user_message, k=3)
                rag_context = "\n\nContexte des documents:\n" + "\n---\n".join([doc.page_content for doc in docs])
                self.chat_area.append(f"<i>Contexte RAG trouvé:\n{rag_context}</i>")
            except Exception as e:
                self.on_error(f"Erreur de recherche RAG: {e}")


        # Mettre à jour l'historique avec le message utilisateur
        # Note: Le contexte RAG n'est PAS ajouté à l'historique sauvegardé,
        # il est injecté temporairement dans le prompt.
        self.append_message("user", user_message, save=True)
        self.input_field.clear()


        self.chat_area.append("<i>L'IA réfléchi...</i>")
        self.set_ui_enabled(False)

        # Préparer l'historique pour le modèle
        conversation_history = list(self.conversations[self.current_conversation_id]) # Copie

        # Injecter le contexte RAG dans le dernier message utilisateur
        if rag_context:
            conversation_history[-1]["content"] = f"{rag_context}\n\nQuestion: {user_message}"


        self.current_assistant_message = "" # Reset for new message
        self.stats_label.setText("") # Clear stats
        self.generation_worker = GenerationWorker(self.model, self.tokenizer, conversation_history, self.settings)
        self.generation_worker.new_token.connect(self.on_new_token)
        self.generation_worker.generation_complete.connect(self.on_generation_complete)
        self.generation_worker.stats.connect(self.on_stats_update)
        self.generation_worker.error.connect(self.on_error)
        self.generation_worker.start()

    def on_new_token(self, token):
        if not self.current_assistant_message: # First token
            # Remove the "L'IA réfléchi..." message
            cursor = self.chat_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.select(cursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
            self.chat_area.setTextCursor(cursor)
            self.append_message("assistant", token, save=False)
        else:
            # Append the new token to the existing message
            cursor = self.chat_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(token)
        self.current_assistant_message += token

    def on_stats_update(self, tokens_per_sec):
        self.stats_label.setText(f"{tokens_per_sec:.2f} tokens/s")

    def on_generation_complete(self, response):
        # The streaming is done, now we save the full response and re-render it with markdown
        self.conversations[self.current_conversation_id].append({"role": "assistant", "content": response})
        self.save_conversations()

        # Re-display the whole conversation to get proper markdown rendering
        self.display_current_conversation()

        self.set_ui_enabled(True)
        self.input_field.setFocus()
        self.current_assistant_message = "" # Clear for next time

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

    # --- RAG Fonctions ---
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
        QApplication.processEvents() # Pour que le message s'affiche

        # Copier les fichiers dans le dossier local
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
            # 1. Charger les documents depuis le dossier
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

            # 2. Splitter les documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings["rag_chunk_size"],
                chunk_overlap=self.settings["rag_chunk_overlap"]
            )
            splits = text_splitter.split_documents(docs)

            # 3. Créer les embeddings et l'index FAISS
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


    # --- Fonctions de gestion de l'historique ---

    def start_new_conversation(self):
        self.current_conversation_id = str(uuid.uuid4())
        self.conversations[self.current_conversation_id] = [
            {"role": "system", "content": self.settings["system_prompt"]}
        ]

        item = QListWidgetItem(f"Nouvelle Discussion - {self.current_conversation_id[:8]}")
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

        history = self.conversations.get(self.current_conversation_id, [])
        for message in history:
            # On utilise une astuce: on appelle append_message sans sauvegarder
            # pour réutiliser la logique d'affichage.
            self.append_message(message["role"], message["content"], save=False)


    def append_message(self, role, content, save=True):
        if not self.current_conversation_id: return

        if save:
            self.conversations[self.current_conversation_id].append({"role": role, "content": content})

        if role == "user":
            html = f"""
            <div style='background-color: #4f4f4f; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>
                <b>Vous:</b>
                <p style='margin: 0;'>{content}</p>
            </div>
            """
        elif role == "assistant":
            # Le contenu de l'assistant est déjà du HTML de markdown2
            formatted_content = markdown2.markdown(content, extras=["fenced-code-blocks", "tables"])
            html = f"""
            <div style='background-color: #4f4f4f; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>
                <b>LiquidAI:</b>
                {formatted_content}
            </div>
            """
        else: # system, error...
            html = f"<i>{content}</i>"

        self.chat_area.append(html)

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

        for filename in os.listdir("conversations"):
            if filename.endswith(".json"):
                conv_id = filename.replace(".json", "")
                with open(f"conversations/{filename}", "r", encoding="utf-8") as f:
                    self.conversations[conv_id] = json.load(f)

                # Le titre est le premier message utilisateur, ou un titre par défaut
                title = f"Discussion {conv_id[:8]}"
                for msg in self.conversations[conv_id]:
                    if msg['role'] == 'user':
                        title = msg['content'][:30] + "..."
                        break

                item = QListWidgetItem(title)
                item.setData(Qt.ItemDataRole.UserRole, conv_id)
                self.history_list.addItem(item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiquidAIApp()
    window.show()
    sys.exit(app.exec())
