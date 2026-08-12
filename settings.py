from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit, QLabel,
    QDialogButtonBox, QDoubleSpinBox, QFormLayout, QSpinBox,
)

from core.config import DEFAULT_SETTINGS


class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")

        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.system_prompt_field = QLineEdit()
        form_layout.addRow(QLabel("Prompt Système:"), self.system_prompt_field)

        self.temperature_field = QDoubleSpinBox()
        self.temperature_field.setRange(0.0, 2.0)
        self.temperature_field.setSingleStep(0.1)
        self.temperature_field.setDecimals(2)
        form_layout.addRow(QLabel("Température:"), self.temperature_field)

        self.min_p_field = QDoubleSpinBox()
        self.min_p_field.setRange(0.0, 1.0)
        self.min_p_field.setSingleStep(0.05)
        self.min_p_field.setDecimals(2)
        self.min_p_field.setToolTip(
            "Seuil min_p officiel LiquidAI (pas top_p). Recommandé : 0.15."
        )
        form_layout.addRow(QLabel("Min P:"), self.min_p_field)

        self.repetition_penalty_field = QDoubleSpinBox()
        self.repetition_penalty_field.setRange(0.0, 2.0)
        self.repetition_penalty_field.setSingleStep(0.05)
        self.repetition_penalty_field.setDecimals(2)
        form_layout.addRow(QLabel("Repetition Penalty:"), self.repetition_penalty_field)

        self.max_tokens_field = QSpinBox()
        self.max_tokens_field.setRange(16, 8192)
        self.max_tokens_field.setSingleStep(64)
        form_layout.addRow(QLabel("Max nouveaux tokens:"), self.max_tokens_field)

        self.chunk_size_field = QSpinBox()
        self.chunk_size_field.setRange(100, 2000)
        self.chunk_size_field.setSingleStep(50)
        self.chunk_size_field.valueChanged.connect(self._sync_overlap_limit)
        form_layout.addRow(QLabel("RAG Chunk Size:"), self.chunk_size_field)

        self.chunk_overlap_field = QSpinBox()
        self.chunk_overlap_field.setRange(0, 1000)
        self.chunk_overlap_field.setSingleStep(10)
        form_layout.addRow(QLabel("RAG Chunk Overlap:"), self.chunk_overlap_field)

        self.layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def _sync_overlap_limit(self, chunk_size: int) -> None:
        self.chunk_overlap_field.setMaximum(max(0, chunk_size - 1))

    def get_settings(self):
        chunk_size = self.chunk_size_field.value()
        overlap = min(self.chunk_overlap_field.value(), max(0, chunk_size - 1))
        return {
            "system_prompt": self.system_prompt_field.text(),
            "temperature": self.temperature_field.value(),
            "min_p": self.min_p_field.value(),
            "repetition_penalty": self.repetition_penalty_field.value(),
            "max_new_tokens": self.max_tokens_field.value(),
            "rag_chunk_size": chunk_size,
            "rag_chunk_overlap": overlap,
        }

    def set_settings(self, settings):
        merged = dict(DEFAULT_SETTINGS)
        merged.update(settings or {})
        self.system_prompt_field.setText(merged.get("system_prompt", DEFAULT_SETTINGS["system_prompt"]))
        self.temperature_field.setValue(merged.get("temperature", 0.3))
        self.min_p_field.setValue(merged.get("min_p", 0.15))
        self.repetition_penalty_field.setValue(merged.get("repetition_penalty", 1.05))
        self.max_tokens_field.setValue(int(merged.get("max_new_tokens", 512)))
        chunk_size = int(merged.get("rag_chunk_size", 500))
        self.chunk_size_field.setValue(chunk_size)
        self._sync_overlap_limit(chunk_size)
        self.chunk_overlap_field.setValue(int(merged.get("rag_chunk_overlap", 50)))
