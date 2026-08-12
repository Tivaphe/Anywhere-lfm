import html
import json
import os
import sys
import traceback
import uuid

import markdown2
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import load_settings, save_settings
from core.device import describe_device
from core.generate import generate
from core.models import (
    SUPPORTED_MODELS,
    LoadedModel,
    is_vl_model,
    load_model,
    unload_model,
)
from core.rag import RagIndex
from settings import SettingsWindow


class ModelWorker(QThread):
    model_loaded = pyqtSignal(object)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            loaded = load_model(self.model_name, status_callback=self.status_update.emit)
            self.model_loaded.emit(loaded)
        except Exception as exc:
            self.error.emit(
                f"Erreur de chargement du modèle : {exc}\n\nTraceback:\n{traceback.format_exc()}"
            )


class GenerationWorker(QThread):
    generation_complete = pyqtSignal(str)
    new_token = pyqtSignal(str)
    error = pyqtSignal(str)
    stats = pyqtSignal(float)

    def __init__(self, loaded_model: LoadedModel, conversation_history, settings, image_path=None):
        super().__init__()
        self.loaded_model = loaded_model
        self.conversation_history = conversation_history
        self.settings = settings
        self.image_path = image_path

    def run(self):
        try:
            result, tokens_per_sec = generate(
                self.loaded_model,
                self.conversation_history,
                self.settings,
                image_path=self.image_path,
                on_token=self.new_token.emit,
            )
            self.stats.emit(tokens_per_sec)
            self.generation_complete.emit(result)
        except Exception as exc:
            self.error.emit(
                f"Erreur de génération : {exc}\n\nTraceback:\n{traceback.format_exc()}"
            )


class RagWorker(QThread):
    status_update = pyqtSignal(str)
    index_ready = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, rag_index: RagIndex, files, chunk_size: int, chunk_overlap: int):
        super().__init__()
        self.rag_index = rag_index
        self.files = files
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(self):
        try:
            self.status_update.emit(f"Copie de {len(self.files)} document(s)...")
            self.rag_index.add_files(self.files)
            self.status_update.emit("Création de la base de données vectorielle...")
            count = self.rag_index.rebuild(self.chunk_size, self.chunk_overlap)
            self.index_ready.emit(count)
        except Exception as exc:
            self.error.emit(
                f"Erreur lors de la création de l'index RAG : {exc}\n\nTraceback:\n{traceback.format_exc()}"
            )


class LiquidAIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiquidAI Chat")
        self.setGeometry(100, 100, 1000, 700)

        self.loaded_model = None
        self.current_conversation_id = None
        self.conversations = {}
        self.settings = load_settings()
        self.rag_enabled = False
        self.rag_index = RagIndex("documents/")
        self.current_assistant_message = ""
        self.selected_image_path = None
        self.busy = False

        self.init_ui()
        self.load_conversations()
        self.check_device()
        self.refresh_model_list()
        self.chat_area.append(
            "<i>Aucun modèle chargé. Choisissez un modèle puis cliquez sur Charger.</i>"
        )
        if not self.conversations:
            self.start_new_conversation()
        else:
            self.history_list.setCurrentRow(0)
            self.load_selected_conversation(self.history_list.item(0))

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.history_list = QListWidget(left_panel)
        self.history_list.itemClicked.connect(self.load_selected_conversation)
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self.show_conversation_context_menu)
        left_layout.addWidget(self.history_list)
        new_chat_button = QPushButton("Nouvelle Discussion")
        new_chat_button.clicked.connect(self.start_new_conversation)
        left_layout.addWidget(new_chat_button)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("font-size: 14px; color: #f2f2f2; background-color: #2b2b2b;")
        right_layout.addWidget(self.chat_area)

        model_controls_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        self.load_button = QPushButton("Charger")
        self.load_button.clicked.connect(self.load_selected_model)
        settings_button = QPushButton("Paramètres")
        settings_button.clicked.connect(self.open_settings)
        self.eject_button = QPushButton("Éjecter")
        self.eject_button.clicked.connect(self.eject_model)
        self.eject_button.setEnabled(False)
        model_controls_layout.addWidget(QLabel("Modèle:"))
        model_controls_layout.addWidget(self.model_selector)
        model_controls_layout.addWidget(self.load_button)
        model_controls_layout.addWidget(self.eject_button)
        model_controls_layout.addWidget(settings_button)
        right_layout.addLayout(model_controls_layout)

        self.image_input_container = QWidget()
        image_input_layout = QHBoxLayout(self.image_input_container)
        image_input_layout.setContentsMargins(0, 5, 0, 5)
        self.select_image_button = QPushButton("Sélectionner une image")
        self.select_image_button.clicked.connect(self.select_image)
        self.image_thumbnail_label = QLabel()
        self.image_thumbnail_label.setFixedSize(64, 64)
        self.image_thumbnail_label.setStyleSheet("border: 1px solid #555;")
        self.image_filename_label = QLabel("Aucune image sélectionnée")
        self.clear_image_button = QPushButton("X")
        self.clear_image_button.setFixedSize(30, 30)
        self.clear_image_button.clicked.connect(self.clear_image)
        image_input_layout.addWidget(self.select_image_button)
        image_input_layout.addWidget(self.image_thumbnail_label)
        image_input_layout.addWidget(self.image_filename_label)
        image_input_layout.addStretch()
        image_input_layout.addWidget(self.clear_image_button)
        right_layout.addWidget(self.image_input_container)
        self.image_input_container.setVisible(False)

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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 750])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.set_ui_enabled(True)

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
            except OSError as exc:
                self.on_error(f"Impossible de supprimer le fichier {file_path}: {exc}")
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
        self.chat_area.append(f"<i>Utilisation de l'appareil : {html.escape(describe_device())}</i>")

    def refresh_model_list(self):
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItems(SUPPORTED_MODELS)
        self.model_selector.blockSignals(False)
        self.on_model_change(self.model_selector.currentText())

    def open_settings(self):
        dialog = SettingsWindow(self)
        dialog.set_settings(self.settings)
        if dialog.exec():
            self.settings = dialog.get_settings()
            save_settings(self.settings)
            self.chat_area.append("<i>Paramètres mis à jour.</i>")
            if self.current_conversation_id:
                history = self.conversations.get(self.current_conversation_id, [])
                if history and history[0].get("role") == "system":
                    history[0] = {"role": "system", "content": self.settings["system_prompt"]}
                else:
                    history.insert(0, {"role": "system", "content": self.settings["system_prompt"]})
                self.display_current_conversation()

    def on_model_change(self, model_identifier: str):
        if not model_identifier:
            return
        vl_selected = is_vl_model(model_identifier)
        self.image_input_container.setVisible(vl_selected)
        if not vl_selected:
            self.clear_image()
        if self.loaded_model and self.loaded_model.name != model_identifier:
            self.chat_area.append(
                f"<i>Modèle sélectionné : {html.escape(model_identifier)}. "
                "Cliquez sur Charger pour l'utiliser.</i>"
            )

    def load_selected_model(self):
        model_identifier = self.model_selector.currentText()
        if not model_identifier:
            return
        if self.loaded_model and self.loaded_model.name == model_identifier:
            self.chat_area.append(f"<i>Le modèle {html.escape(model_identifier)} est déjà chargé.</i>")
            return
        if self.loaded_model:
            unload_model(self.loaded_model)
            self.loaded_model = None
        self.load_model(model_identifier)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner une image", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if file_path:
            self.selected_image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_thumbnail_label.setPixmap(
                pixmap.scaled(
                    64,
                    64,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.image_filename_label.setText(os.path.basename(file_path))

    def clear_image(self):
        self.selected_image_path = None
        self.image_thumbnail_label.clear()
        self.image_filename_label.setText("Aucune image sélectionnée")

    def load_model(self, model_identifier: str):
        self.set_ui_enabled(False)
        self.chat_area.append(f"<i>Préparation du chargement du modèle {html.escape(model_identifier)}...</i>")
        self.worker = ModelWorker(model_identifier)
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.start()

    def on_status_update(self, message: str):
        self.chat_area.append(f"<i>{html.escape(message)}</i>")

    def on_model_loaded(self, loaded: LoadedModel):
        self.loaded_model = loaded
        self.chat_area.append(f"<i>Modèle {html.escape(loaded.name)} chargé.</i>")
        self.set_ui_enabled(True)
        if self.current_conversation_id:
            self.display_current_conversation()

    def eject_model(self):
        if self.loaded_model is None:
            return
        model_name = self.loaded_model.name
        self.chat_area.append(f"<i>Déchargement du modèle {html.escape(model_name)}...</i>")
        unload_model(self.loaded_model)
        self.loaded_model = None
        self.chat_area.append("<i>Modèle déchargé. Choisissez un modèle puis cliquez sur Charger.</i>")
        self.set_ui_enabled(True)

    def send_message(self):
        user_message = self.input_field.text().strip()
        loaded_is_vl = bool(self.loaded_model and self.loaded_model.kind == "vl")

        if loaded_is_vl and not self.selected_image_path:
            self.on_error("Veuillez sélectionner une image pour utiliser ce modèle Vision-Language.")
            return

        if not user_message and not self.selected_image_path:
            return

        if not self.loaded_model or not self.current_conversation_id:
            self.on_error("Chargez d'abord un modèle avant d'envoyer un message.")
            return

        rag_context = ""
        if self.rag_enabled and self.rag_index.ready and not loaded_is_vl and user_message:
            try:
                found = self.rag_index.search(user_message, k=3)
                if found:
                    rag_context = f"\n\nContexte des documents:\n{found}"
                    self.chat_area.append(f"<i>Contexte RAG trouvé:\n{html.escape(rag_context)}</i>")
            except Exception as exc:
                self.on_error(f"Erreur de recherche RAG: {exc}")

        display_message = user_message or "(image)"
        self.append_message("user", display_message, save=True)
        self.input_field.clear()

        self.chat_area.append("<i>L'IA réfléchit...</i>")
        self.set_ui_enabled(False)

        conversation_history = list(self.conversations[self.current_conversation_id])
        if rag_context:
            conversation_history[-1] = dict(conversation_history[-1])
            conversation_history[-1]["content"] = f"{rag_context}\n\nQuestion: {user_message}"

        self.current_assistant_message = ""
        self.stats_label.setText("")

        self.generation_worker = GenerationWorker(
            self.loaded_model,
            conversation_history,
            self.settings,
            self.selected_image_path,
        )
        self.generation_worker.new_token.connect(self.on_new_token)
        self.generation_worker.generation_complete.connect(self.on_generation_complete)
        self.generation_worker.stats.connect(self.on_stats_update)
        self.generation_worker.error.connect(self.on_error)
        self.generation_worker.finished.connect(self.clear_image_after_generation)
        self.generation_worker.start()

    def clear_image_after_generation(self):
        self.clear_image()

    def on_new_token(self, token: str):
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

    def on_stats_update(self, tokens_per_sec: float):
        self.stats_label.setText(f"{tokens_per_sec:.2f} tokens/s")

    def on_generation_complete(self, response: str):
        if not self.current_conversation_id:
            self.set_ui_enabled(True)
            return
        self.conversations[self.current_conversation_id].append(
            {"role": "assistant", "content": response}
        )
        self.save_conversations()
        self.display_current_conversation()
        self.set_ui_enabled(True)
        self.input_field.setFocus()
        self.current_assistant_message = ""

    def on_error(self, error_message: str):
        self.chat_area.append(f"<font color='#ff6b6b'>Erreur : {html.escape(error_message)}</font>")
        self.set_ui_enabled(True)

    def set_ui_enabled(self, enabled: bool):
        self.busy = not enabled
        has_model = self.loaded_model is not None
        self.input_field.setEnabled(enabled and has_model)
        self.send_button.setEnabled(enabled and has_model)
        self.model_selector.setEnabled(enabled)
        self.load_button.setEnabled(enabled)
        self.eject_button.setEnabled(enabled and has_model)
        self.history_list.setEnabled(enabled)
        self.load_docs_button.setEnabled(enabled)
        self.rag_toggle_checkbox.setEnabled(enabled and self.rag_index.ready)

    def toggle_rag(self, state):
        self.rag_enabled = state == Qt.CheckState.Checked.value
        if self.rag_enabled:
            self.rag_status_label.setText("RAG: Actif")
            self.chat_area.append(
                "<i>RAG activé. Les documents chargés seront utilisés comme contexte (modèles texte uniquement).</i>"
            )
        else:
            self.rag_status_label.setText("RAG: Inactif" if not self.rag_index.ready else "RAG: Prêt")
            self.chat_area.append("<i>RAG désactivé.</i>")

    def load_documents(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Sélectionner des documents", "", "Documents (*.txt *.pdf *.docx)"
        )
        if not files:
            return

        self.set_ui_enabled(False)
        self.rag_worker = RagWorker(
            self.rag_index,
            files,
            int(self.settings.get("rag_chunk_size", 500)),
            int(self.settings.get("rag_chunk_overlap", 50)),
        )
        self.rag_worker.status_update.connect(self.on_status_update)
        self.rag_worker.index_ready.connect(self.on_rag_ready)
        self.rag_worker.error.connect(self.on_rag_error)
        self.rag_worker.start()

    def on_rag_ready(self, chunk_count: int):
        self.rag_status_label.setText("RAG: Prêt")
        self.rag_toggle_checkbox.setEnabled(True)
        self.chat_area.append(
            f"<i>Index RAG créé ({chunk_count} morceaux). Vous pouvez maintenant activer le RAG.</i>"
        )
        self.set_ui_enabled(True)

    def on_rag_error(self, message: str):
        self.rag_status_label.setText("RAG: Erreur")
        self.rag_toggle_checkbox.setChecked(False)
        self.rag_enabled = False
        self.on_error(message)

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
        if self.loaded_model:
            self.chat_area.append(f"<i>Modèle {html.escape(self.loaded_model.name)} chargé.</i>")
        else:
            self.chat_area.append(
                "<i>Aucun modèle chargé. Choisissez un modèle puis cliquez sur Charger.</i>"
            )

        if not self.current_conversation_id:
            return

        history = self.conversations.get(self.current_conversation_id, [])
        for message in history:
            self.append_message(message["role"], message.get("content", ""), save=False)

    def _conversation_title(self, history, conv_id: str) -> str:
        for message in history:
            if message.get("role") == "user":
                content = message.get("content", "")
                if not isinstance(content, str):
                    return "Image + texte"
                snippet = content[:30]
                return snippet + ("..." if len(content) > 30 else "")
        return f"Discussion {conv_id[:8]}"

    def append_message(self, role, content, save=True):
        if not self.current_conversation_id:
            return

        text = content if isinstance(content, str) else str(content)
        if save:
            self.conversations[self.current_conversation_id].append({"role": role, "content": text})

        safe = html.escape(text)
        if role == "user":
            bubble = f"""
            <div style='background-color: #3a3a3a; color: #f2f2f2; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>
                <b>Vous:</b>
                <p style='margin: 0;'>{safe}</p>
            </div>
            """
        elif role == "assistant":
            formatted = markdown2.markdown(
                text,
                extras=["fenced-code-blocks", "tables"],
                safe_mode="escape",
            )
            bubble = f"""
            <div style='background-color: #3a3a3a; color: #f2f2f2; padding: 10px; border-radius: 5px; margin-bottom: 5px;'>
                <b>LiquidAI:</b>
                {formatted}
            </div>
            """
        else:
            bubble = f"<i>{safe}</i>"

        self.chat_area.append(bubble)
        if save:
            self.save_conversations()

    def save_conversations(self):
        os.makedirs("conversations", exist_ok=True)
        if not self.current_conversation_id:
            return
        history = self.conversations.get(self.current_conversation_id)
        if history is None:
            return
        path = f"conversations/{self.current_conversation_id}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    def load_conversations(self):
        if not os.path.exists("conversations"):
            return

        for filename in os.listdir("conversations"):
            if not filename.endswith(".json"):
                continue
            conv_id = filename.replace(".json", "")
            path = os.path.join("conversations", filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    self.conversations[conv_id] = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            title = self._conversation_title(self.conversations[conv_id], conv_id)
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, conv_id)
            self.history_list.addItem(item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiquidAIApp()
    window.show()
    sys.exit(app.exec())
