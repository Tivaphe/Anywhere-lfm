from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QLabel,
                             QDialogButtonBox, QDoubleSpinBox, QFormLayout, QSpinBox)

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")

        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Prompt Système
        self.system_prompt_field = QLineEdit()
        form_layout.addRow(QLabel("Prompt Système:"), self.system_prompt_field)

        # Température
        self.temperature_field = QDoubleSpinBox()
        self.temperature_field.setRange(0.0, 2.0)
        self.temperature_field.setSingleStep(0.1)
        form_layout.addRow(QLabel("Température:"), self.temperature_field)

        # Min P
        self.min_p_field = QDoubleSpinBox()
        self.min_p_field.setRange(0.0, 1.0)
        self.min_p_field.setSingleStep(0.05)
        form_layout.addRow(QLabel("Min P:"), self.min_p_field)

        # Repetition Penalty
        self.repetition_penalty_field = QDoubleSpinBox()
        self.repetition_penalty_field.setRange(0.0, 2.0)
        self.repetition_penalty_field.setSingleStep(0.05)
        form_layout.addRow(QLabel("Repetition Penalty:"), self.repetition_penalty_field)

        # RAG Settings
        self.chunk_size_field = QSpinBox()
        self.chunk_size_field.setRange(100, 2000)
        self.chunk_size_field.setSingleStep(50)
        form_layout.addRow(QLabel("RAG Chunk Size:"), self.chunk_size_field)

        self.chunk_overlap_field = QSpinBox()
        self.chunk_overlap_field.setRange(0, 1000)
        self.chunk_overlap_field.setSingleStep(10)
        form_layout.addRow(QLabel("RAG Chunk Overlap:"), self.chunk_overlap_field)


        self.layout.addLayout(form_layout)

        # Boutons OK et Annuler
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.layout.addWidget(self.button_box)

    def get_settings(self):
        return {
            "system_prompt": self.system_prompt_field.text(),
            "temperature": self.temperature_field.value(),
            "min_p": self.min_p_field.value(),
            "repetition_penalty": self.repetition_penalty_field.value(),
            "rag_chunk_size": self.chunk_size_field.value(),
            "rag_chunk_overlap": self.chunk_overlap_field.value()
        }

    def set_settings(self, settings):
        self.system_prompt_field.setText(settings.get("system_prompt", "You are a helpful assistant."))
        self.temperature_field.setValue(settings.get("temperature", 0.3))
        self.min_p_field.setValue(settings.get("min_p", 0.15))
        self.repetition_penalty_field.setValue(settings.get("repetition_penalty", 1.05))
        self.chunk_size_field.setValue(settings.get("rag_chunk_size", 500))
        self.chunk_overlap_field.setValue(settings.get("rag_chunk_overlap", 50))
