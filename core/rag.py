"""Indexation et recherche RAG (documents locaux + FAISS)."""

from __future__ import annotations

import os
import shutil
from typing import List, Optional

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RagIndex:
    def __init__(self, documents_path: str = "documents/") -> None:
        self.documents_path = documents_path
        os.makedirs(self.documents_path, exist_ok=True)
        self.vector_store = None

    @property
    def ready(self) -> bool:
        return self.vector_store is not None

    def add_files(self, file_paths: List[str]) -> List[str]:
        copied = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            destination = os.path.join(self.documents_path, filename)
            shutil.copy2(file_path, destination)
            copied.append(destination)
        return copied

    def clear(self) -> None:
        self.vector_store = None
        if os.path.isdir(self.documents_path):
            for name in os.listdir(self.documents_path):
                path = os.path.join(self.documents_path, name)
                if os.path.isfile(path):
                    os.remove(path)

    def rebuild(self, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
        from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        overlap = min(max(0, int(chunk_overlap)), max(0, int(chunk_size) - 1))
        docs = []
        for filename in os.listdir(self.documents_path):
            file_path = os.path.join(self.documents_path, filename)
            if not os.path.isfile(file_path):
                continue
            lower = filename.lower()
            if lower.endswith(".pdf"):
                docs.extend(PyPDFLoader(file_path).load())
            elif lower.endswith(".docx"):
                docs.extend(Docx2txtLoader(file_path).load())
            elif lower.endswith(".txt"):
                docs.extend(TextLoader(file_path, encoding="utf-8").load())

        if not docs:
            self.vector_store = None
            raise ValueError("Aucun document valide trouvé à traiter.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=overlap,
        )
        splits = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
        self.vector_store = FAISS.from_documents(splits, embeddings)
        return len(splits)

    def search(self, query: str, k: int = 3) -> str:
        if not self.vector_store:
            return ""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n---\n".join(doc.page_content for doc in docs)
