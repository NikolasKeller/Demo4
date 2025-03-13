from typing import List, Dict, Optional, Union
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class ChunkMetadata:
    """Metadaten für einen Text-Chunk"""
    document_name: str
    chunk_id: str
    page_number: int
    chunk_number: int
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "document_name": self.document_name,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "chunk_number": self.chunk_number,
            "timestamp": self.timestamp
        }

class VectorStoreException(Exception):
    """Basisklasse für VectorStore-Ausnahmen"""
    pass

class VectorStore:
    def __init__(self,
                 persist_directory: str = "chroma_db",
                 collection_name: str = "pdf_chunks",
                 embedding_function_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialisiert die Vektordatenbank
        
        Args:
            persist_directory: Verzeichnis für persistente Speicherung
            collection_name: Name der Collection
            embedding_function_name: Name des Embedding-Modells
        """
        self.logger = logging.getLogger(__name__)
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        try:
            # Verzeichnis erstellen
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Client initialisieren
            self.client = chromadb.Client(Settings(
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False,
                is_persistent=True
            ))
            
            # Embedding-Funktion konfigurieren
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_function_name
            )
            
            # Collection erstellen oder laden
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.logger.info(f"Bestehende Collection '{collection_name}' geladen")
            except ValueError:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.logger.info(f"Neue Collection '{collection_name}' erstellt")
                
        except Exception as e:
            raise VectorStoreException(f"Fehler bei der Initialisierung: {str(e)}")
    
    def add_chunks(self,
                  chunks: List[str],
                  embeddings: np.ndarray,
                  document_name: str,
                  page_numbers: List[int]) -> bool:
        """
        Fügt Chunks und ihre Embeddings zur Datenbank hinzu
        
        Args:
            chunks: Liste von Textabschnitten
            embeddings: NumPy-Array mit Embeddings
            document_name: Name des Quelldokuments
            page_numbers: Liste der Seitenzahlen für jeden Chunk
        """
        try:
            # Eingabevalidierung
            if len(chunks) != len(embeddings):
                raise ValueError("Anzahl der Chunks und Embeddings stimmt nicht überein")
            
            if len(chunks) != len(page_numbers):
                raise ValueError("Anzahl der Chunks und Seitenzahlen stimmt nicht überein")
            
            # IDs und Metadaten vorbereiten
            chunk_ids = []
            metadatas = []
            
            for i, (chunk, page_num) in enumerate(zip(chunks, page_numbers)):
                chunk_id = f"{document_name}_chunk_{i}"
                metadata = ChunkMetadata(
                    document_name=document_name,
                    chunk_id=chunk_id,
                    page_number=page_num,
                    chunk_number=i
                )
                
                chunk_ids.append(chunk_id)
                metadatas.append(metadata.to_dict())
            
            # Chunks zur Collection hinzufügen
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),  # NumPy-Array in Liste konvertieren
                documents=chunks,
                metadatas=metadatas
            )
            
            self.logger.info(
                f"{len(chunks)} Chunks aus {document_name} zur Vektordatenbank hinzugefügt"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Hinzufügen der Chunks: {str(e)}")
            return False
    
    def search(self,
              query: str,
              n_results: int = 3,
              where: Optional[Dict] = None) -> List[Dict]:
        """
        Führt eine Ähnlichkeitssuche durch
        
        Args:
            query: Suchanfrage
            n_results: Anzahl der gewünschten Ergebnisse
            where: Optionaler Filter für Metadaten
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Ergebnisse formatieren
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Suche: {str(e)}")
            raise VectorStoreException(f"Suchfehler: {str(e)}")
    
    def get_document_chunks(self, document_name: str) -> List[Dict]:
        """Holt alle Chunks eines bestimmten Dokuments"""
        try:
            results = self.collection.query(
                query_texts=[""],  # Leere Abfrage
                where={"document_name": document_name},
                n_results=1000  # Hoher Wert, um alle Chunks zu bekommen
            )
            
            return [{
                'id': id_,
                'text': doc,
                'metadata': meta
            } for id_, doc, meta in zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0]
            )]
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Dokument-Chunks: {str(e)}")
            raise VectorStoreException(f"Fehler beim Abrufen der Chunks: {str(e)}")
    
    def list_documents(self) -> List[str]:
        """Listet alle verfügbaren Dokumente auf"""
        try:
            results = self.collection.get()
            if not results['metadatas']:
                return []
            
            # Eindeutige Dokumentnamen extrahieren
            document_names = {
                meta['document_name']
                for meta in results['metadatas']
            }
            
            return sorted(list(document_names))
            
        except Exception as e:
            self.logger.error(f"Fehler beim Auflisten der Dokumente: {str(e)}")
            raise VectorStoreException(f"Fehler beim Auflisten: {str(e)}")
    
    def has_documents(self) -> bool:
        """Prüft, ob Dokumente in der Collection vorhanden sind"""
        try:
            return len(self.list_documents()) > 0
        except:
            return False
    
    def delete_document(self, document_name: str) -> bool:
        """Löscht alle Chunks eines Dokuments"""
        try:
            self.collection.delete(
                where={"document_name": document_name}
            )
            self.logger.info(f"Dokument '{document_name}' gelöscht")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Löschen von {document_name}: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Löscht alle Daten aus der Collection"""
        try:
            self.collection.delete()
            self.logger.info("Collection geleert")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Leeren der Collection: {str(e)}")
            return False

# Beispielverwendung
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Vektordatenbank initialisieren
    store = VectorStore(
        persist_directory="./chroma_db",
        collection_name="pdf_chunks"
    )
    
    # Beispieldaten
    chunks = [
        "Dies ist der erste Testabschnitt.",
        "Dies ist der zweite Testabschnitt.",
        "Dies ist der dritte Testabschnitt."
    ]
    
    # Dummy-Embeddings (normalerweise von einem Embedding-Modell)
    embeddings = np.random.rand(len(chunks), 384)  # 384 ist die Embedding-Dimension
    
    # Chunks hinzufügen
    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_name="test.pdf",
        page_numbers=[1, 1, 2]
    )
    
    # Suche testen
    results = store.search(
        query="Testabschnitt",
        n_results=2
    )
    
    for result in results:
        print(f"\nGefunden (Score: {1 - result['distance']:.3f}):")
        print(f"Text: {result['text']}")
        print(f"Metadata: {result['metadata']}") 