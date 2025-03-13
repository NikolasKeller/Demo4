from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
from vector_store import VectorStore, VectorStoreException

@dataclass
class SearchResult:
    """Repräsentiert ein einzelnes Suchergebnis"""
    text: str
    score: float
    document: str
    page: int
    chunk_id: str
    context: Optional[str] = None

class SemanticSearcher:
    def __init__(self,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 persist_directory: str = "chroma_db",
                 device: Optional[str] = None):
        """
        Initialisiert den semantischen Sucher
        
        Args:
            model_name: Name des Embedding-Modells
            persist_directory: Verzeichnis der ChromaDB
            device: Optional - 'cuda' oder 'cpu'
        """
        self.logger = logging.getLogger(__name__)
        
        # Device für Inferenz bestimmen
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Embedding-Modell laden
            self.model = SentenceTransformer(model_name, device=device)
            self.logger.info(f"Modell '{model_name}' geladen auf {device}")
            
            # Vektordatenbank initialisieren
            self.vector_store = VectorStore(
                persist_directory=persist_directory,
                embedding_function_name=model_name
            )
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung: {str(e)}")
            raise
    
    def search(self,
              query: str,
              top_k: int = 3,
              min_score: float = 0.3,
              document_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Führt eine semantische Suche durch
        
        Args:
            query: Suchanfrage
            top_k: Anzahl der gewünschten Ergebnisse
            min_score: Minimaler Ähnlichkeitsscore (0-1)
            document_filter: Optional - Nur in diesem Dokument suchen
        
        Returns:
            Liste von SearchResult-Objekten
        """
        try:
            # Eingabevalidierung
            if not query.strip():
                raise ValueError("Suchanfrage darf nicht leer sein")
            
            if top_k < 1:
                raise ValueError("top_k muss mindestens 1 sein")
            
            # Filter vorbereiten
            where_filter = {"document_name": document_filter} if document_filter else None
            
            # Suche durchführen
            results = self.vector_store.search(
                query=query,
                n_results=top_k,
                where=where_filter
            )
            
            # Ergebnisse verarbeiten
            search_results = []
            for result in results:
                # Score aus Distanz berechnen (1 = beste Übereinstimmung)
                score = 1 - (result['distance'] or 0)
                
                # Nur Ergebnisse über dem Schwellenwert
                if score >= min_score:
                    search_results.append(SearchResult(
                        text=result['text'],
                        score=score,
                        document=result['metadata']['document_name'],
                        page=result['metadata']['page_number'],
                        chunk_id=result['metadata']['chunk_id']
                    ))
            
            return search_results
            
        except VectorStoreException as e:
            self.logger.error(f"Datenbankfehler bei der Suche: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Fehler bei der Suche: {str(e)}")
            raise
    
    def print_results(self, 
                     results: List[SearchResult],
                     query: str,
                     show_details: bool = True):
        """Gibt Suchergebnisse formatiert aus"""
        if not results:
            print("\nKeine relevanten Ergebnisse gefunden.")
            return
        
        print(f"\nSuchergebnisse für: '{query}'")
        print(f"Gefunden: {len(results)} relevante Stellen\n")
        
        for i, result in enumerate(results, 1):
            print(f"{'='*80}")
            print(f"Ergebnis {i}")
            print(f"Relevanz: {result.score:.1%}")
            if show_details:
                print(f"Dokument: {result.document}")
                print(f"Seite: {result.page}")
                print(f"Chunk-ID: {result.chunk_id}")
            print(f"{'-'*80}")
            print(f"Text: {result.text}")
            if result.context:
                print(f"\nKontext: ...{result.context}...")
            print()

def setup_logging(level: int = logging.INFO):
    """Konfiguriert das Logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Beispielverwendung
if __name__ == "__main__":
    setup_logging()
    
    try:
        # Searcher initialisieren
        searcher = SemanticSearcher()
        
        while True:
            # Benutzereingabe
            print("\nGeben Sie Ihre Suchanfrage ein (oder 'q' zum Beenden):")
            query = input("> ").strip()
            
            if query.lower() == 'q':
                break
            
            if not query:
                print("Bitte geben Sie eine Suchanfrage ein.")
                continue
            
            # Optionale Parameter
            try:
                top_k = int(input("Anzahl der Ergebnisse (Enter für 3): ") or 3)
                document = input("Dokumentfilter (Enter für alle): ").strip() or None
            except ValueError:
                print("Ungültige Eingabe, verwende Standardwerte.")
                top_k = 3
                document = None
            
            # Suche durchführen
            try:
                results = searcher.search(
                    query=query,
                    top_k=top_k,
                    document_filter=document
                )
                
                # Ergebnisse ausgeben
                searcher.print_results(results, query)
                
            except Exception as e:
                print(f"\nFehler bei der Suche: {str(e)}")
                continue
            
    except KeyboardInterrupt:
        print("\nProgramm beendet.")
    except Exception as e:
        print(f"\nUnerwarteter Fehler: {str(e)}") 