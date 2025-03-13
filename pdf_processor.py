from typing import List, Dict, Optional, Tuple, Union
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
import re
from nltk.tokenize import sent_tokenize
import nltk
import unicodedata
import torch
from tqdm import tqdm
from vector_store import VectorStore
from search_result import SearchResult, SearchResultFormatter
from input_validation import InputValidator, ValidationResult
import contextlib
from functools import wraps
import time

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class ExtractedText:
    content: List[str]
    page_count: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class TextChunk:
    text: str
    page_num: int
    chunk_num: int
    token_count: int
    embedding: Optional[np.ndarray] = None

# Benutzerdefinierte Ausnahmen
class PDFProcessingError(Exception):
    """Basisklasse für PDF-Verarbeitungsfehler"""
    pass

class PDFExtractionError(PDFProcessingError):
    """Fehler bei der Textextraktion"""
    pass

class EmbeddingError(PDFProcessingError):
    """Fehler bei der Einbettungsgenerierung"""
    pass

class DatabaseError(PDFProcessingError):
    """Fehler bei Datenbankoperationen"""
    pass

def retry_on_error(max_attempts: int = 3, delay: float = 1.0):
    """Decorator für automatische Wiederholungsversuche bei Fehlern"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))  # Exponentielles Backoff
                        continue
                    raise last_error
            
        return wrapper
    return decorator

class PDFSearchEngine:
    def __init__(self,
                 model_name: str = 'paraphrase-multilingual-mpnet-base-v2',  # Besseres Modell für mehrsprachige Dokumente
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100,
                 batch_size: int = 32,
                 persist_directory: str = "chroma_db"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.validator = InputValidator()
        
        # Initialisierung des Modells mit verbesserten Einstellungen
        try:
            self.model = SentenceTransformer(model_name)
            self.model.max_seq_length = 512  # Optimale Länge für die meisten Transformers
            
            # GPU-Beschleunigung wenn verfügbar
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            self.batch_size = batch_size
            self.logger.info(f"Modell '{model_name}' geladen auf {self.device}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells: {str(e)}")
            raise
            
        self.documents: Dict[str, List[TextChunk]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Vektordatenbank initialisieren
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            embedding_model_name=model_name
        )
        self.result_formatter = SearchResultFormatter()

    def estimate_tokens(self, text: str) -> int:
        """Schätzt die Anzahl der Tokens in einem Text"""
        return len(text.split())
    
    def clean_text(self, text: str) -> str:
        """Bereinigt den Text von unerwünschten Zeichen und Formatierungen"""
        # Mehrfache Leerzeichen entfernen
        text = re.sub(r'\s+', ' ', text)
        # Seitenumbrüche entfernen
        text = re.sub(r'\f', ' ', text)
        # Unicode-Kontrollzeichen entfernen
        text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
        return text.strip()
    
    def create_chunks(self, text: str, page_num: int) -> List[TextChunk]:
        """
        Teilt Text in semantisch sinnvolle Chunks unter Berücksichtigung von Satzgrenzen
        """
        # Text bereinigen
        text = self.clean_text(text)
        
        # Text in Sätze aufteilen
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        chunk_num = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_tokens = self.estimate_tokens(sentence)
            
            # Zu lange einzelne Sätze aufteilen
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    # Vorherigen Chunk speichern
                    chunks.append(TextChunk(
                        text=' '.join(current_chunk),
                        page_num=page_num,
                        chunk_num=chunk_num,
                        token_count=current_token_count
                    ))
                    chunk_num += 1
                    current_chunk = []
                    current_token_count = 0
                
                # Langen Satz in kleinere Teile aufteilen
                words = sentence.split()
                temp_chunk = []
                temp_token_count = 0
                
                for word in words:
                    temp_chunk.append(word)
                    temp_token_count += 1
                    
                    if temp_token_count >= self.chunk_size - 50:  # Puffer für Satzende
                        chunks.append(TextChunk(
                            text=' '.join(temp_chunk),
                            page_num=page_num,
                            chunk_num=chunk_num,
                            token_count=temp_token_count
                        ))
                        chunk_num += 1
                        temp_chunk = []
                        temp_token_count = 0
                
                if temp_chunk:  # Restliche Wörter
                    current_chunk = temp_chunk
                    current_token_count = temp_token_count
                continue
            
            # Prüfen, ob der aktuelle Satz in den Chunk passt
            if current_token_count + sentence_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_token_count += sentence_tokens
            else:
                # Chunk speichern und neuen beginnen
                if current_token_count >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        text=' '.join(current_chunk),
                        page_num=page_num,
                        chunk_num=chunk_num,
                        token_count=current_token_count
                    ))
                    chunk_num += 1
                
                # Überlappung für Kontext
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + [sentence]
                current_token_count = sum(self.estimate_tokens(s) for s in current_chunk)
        
        # Letzten Chunk speichern, wenn groß genug
        if current_chunk and current_token_count >= self.min_chunk_size:
            chunks.append(TextChunk(
                text=' '.join(current_chunk),
                page_num=page_num,
                chunk_num=chunk_num,
                token_count=current_token_count
            ))
        
        return chunks

    def extract_text_from_pdf(self, pdf_path: str) -> ExtractedText:
        """Extrahiert Text aus einer PDF-Datei mit verbessertem Chunking"""
        try:
            path = Path(pdf_path)
            if not path.exists():
                raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")
            
            if not path.suffix.lower() == '.pdf':
                raise PDFExtractionError(f"Datei ist keine PDF: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            all_chunks = []
            
            for page_num, page in enumerate(doc, 1):
                try:
                    text = page.get_text()
                    # Text in Chunks aufteilen
                    page_chunks = self.create_chunks(text, page_num)
                    all_chunks.extend(page_chunks)
                    
                    self.logger.debug(
                        f"Seite {page_num}: {len(page_chunks)} Chunks erstellt "
                        f"(durchschnittlich {sum(c.token_count for c in page_chunks)/len(page_chunks):.0f} Tokens)"
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Fehler beim Verarbeiten von Seite {page_num}: {str(e)}")
            
            if not all_chunks:
                return ExtractedText(
                    content=[],
                    page_count=len(doc),
                    success=False,
                    error_message="Keine verwertbaren Textinhalte gefunden"
                )
            
            return ExtractedText(
                content=[chunk.text for chunk in all_chunks],
                page_count=len(doc),
                success=True
            )
            
        except FileNotFoundError as e:
            self.logger.error(f"Datei nicht gefunden: {str(e)}")
            raise
        except fitz.FileDataError:
            return ExtractedText(
                content=[],
                page_count=0,
                success=False,
                error_message="Beschädigte oder ungültige PDF-Datei"
            )
        except Exception as e:
            self.logger.error(f"Unerwarteter Fehler bei der PDF-Verarbeitung: {str(e)}")
            return ExtractedText(
                content=[],
                page_count=0,
                success=False,
                error_message=f"Fehler bei der Textextraktion: {str(e)}"
            )

    def generate_embeddings(self, chunks: List[TextChunk]) -> np.ndarray:
        """
        Generiert Einbettungen für eine Liste von TextChunks mit Batch-Verarbeitung
        und Fortschrittsanzeige
        """
        texts = [chunk.text for chunk in chunks]
        self.logger.info(f"Generiere Einbettungen für {len(texts)} Chunks...")
        
        try:
            # Aktiviere den Evaluierungsmodus für bessere Performance
            self.model.eval()
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalisierung für effizientere Ähnlichkeitsberechnung
                )
            
            # Speichere Einbettungen in den Chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Einbettungsgenerierung: {str(e)}")
            raise

    @retry_on_error(max_attempts=3)
    def load_pdf(self, pdf_path: str) -> Tuple[bool, Optional[str]]:
        """PDF-Datei laden mit erweiterter Fehlerbehandlung"""
        try:
            # Eingabevalidierung
            validation = self.validator.validate_pdf_file(pdf_path)
            if not validation.is_valid:
                return False, validation.error_message
            
            # PDF verarbeiten mit Ressourcenverwaltung
            with contextlib.closing(fitz.open(pdf_path)) as doc:
                extracted = self.extract_text_from_pdf(doc)
                
                if not extracted.success:
                    return False, extracted.error_message
                
                path = Path(pdf_path)
                chunks = self.create_chunks(extracted.content, path.name)
                
                if not chunks:
                    return False, "Keine verwertbaren Textabschnitte gefunden"
                
                # Chunks zur Vektordatenbank hinzufügen
                try:
                    self._add_chunks_to_db(chunks, path.name)
                except DatabaseError as e:
                    return False, f"Datenbankfehler: {str(e)}"
                
                # Erfolgsprotokoll
                self._log_success(path.name, extracted.page_count, len(chunks))
                return True, None
                
        except PDFExtractionError as e:
            self.logger.error(f"Fehler bei der PDF-Extraktion: {str(e)}")
            return False, str(e)
        except Exception as e:
            self.logger.error(f"Unerwarteter Fehler: {str(e)}")
            return False, "Interner Fehler bei der PDF-Verarbeitung"
    
    def get_context(self, document: str, chunk_num: int) -> Optional[str]:
        """Holt Kontext aus benachbarten Chunks"""
        try:
            chunks = self.vector_store.get_document_chunks(document)
            chunks.sort(key=lambda x: x['metadata']['chunk_num'])
            
            for i, chunk in enumerate(chunks):
                if chunk['metadata']['chunk_num'] == chunk_num:
                    context = []
                    
                    # Vorheriger Chunk
                    if i > 0:
                        prev_text = chunks[i-1]['text']
                        context.append(prev_text[-100:])  # Letzten 100 Zeichen
                    
                    # Nächster Chunk
                    if i < len(chunks) - 1:
                        next_text = chunks[i+1]['text']
                        context.append(next_text[:100])  # Ersten 100 Zeichen
                    
                    return " ... ".join(context) if context else None
                    
            return None
            
        except Exception as e:
            self.logger.warning(f"Kontext konnte nicht geladen werden: {str(e)}")
            return None

    def search(self,
              query: str,
              top_k: int = 3,
              min_score: float = 0.3,
              filter_dict: Optional[Dict] = None,
              format_output: bool = True) -> Union[str, List[SearchResult], Tuple[bool, str]]:
        """Semantische Suche mit erweiterter Fehlerbehandlung"""
        try:
            # Eingabevalidierung
            query_validation = self.validator.validate_query(query)
            if not query_validation.is_valid:
                return False, query_validation.error_message
                
            params_validation = self.validator.validate_search_params(top_k, min_score)
            if not params_validation.is_valid:
                return False, params_validation.error_message
            
            # Sicherstellen, dass Dokumente geladen sind
            if not self.vector_store.has_documents():
                return False, "Keine Dokumente zum Durchsuchen verfügbar"
            
            # Suche durchführen
            try:
                raw_results = self.vector_store.similarity_search(
                    query=query,
                    n_results=top_k,
                    filter_dict=filter_dict
                )
            except DatabaseError as e:
                return False, f"Datenbankfehler bei der Suche: {str(e)}"
            
            # Ergebnisse verarbeiten
            results = self._process_search_results(raw_results, min_score)
            
            if not results:
                return [], "Keine relevanten Ergebnisse gefunden"
            
            # Ausgabe formatieren
            if format_output:
                return self.result_formatter.format_results(results, query)
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Suche: {str(e)}")
            return False, f"Interner Fehler bei der Suche: {str(e)}"
    
    def _add_chunks_to_db(self, chunks: List[Dict], document_name: str) -> None:
        """Fügt Chunks zur Datenbank hinzu mit Fehlerbehandlung"""
        try:
            self.vector_store.add_chunks(chunks, document_name)
        except Exception as e:
            raise DatabaseError(f"Fehler beim Hinzufügen zur Datenbank: {str(e)}")
    
    def _process_search_results(self, 
                              raw_results: List[Dict],
                              min_score: float) -> List[SearchResult]:
        """Verarbeitet Suchergebnisse mit Fehlerbehandlung"""
        results = []
        for r in raw_results:
            try:
                score = 1 - (r['distance'] or 0)
                if score >= min_score:
                    context = self.get_context(
                        r['metadata']['document'],
                        r['metadata']['chunk_num']
                    )
                    
                    results.append(SearchResult(
                        text=r['text'],
                        document=r['metadata']['document'],
                        page=r['metadata']['page'],
                        chunk=r['metadata']['chunk_num'],
                        score=score,
                        context=context
                    ))
            except Exception as e:
                self.logger.warning(f"Fehler bei der Verarbeitung eines Ergebnisses: {str(e)}")
                continue
                
        return results

    def _log_success(self, document_name: str, page_count: int, chunk_count: int) -> None:
        """Loggt Erfolgsmeldungen für die PDF-Verarbeitung"""
        avg_chunk_size = sum(len(chunk.text.split()) for chunk in self.documents[document_name]) / len(self.documents[document_name])
        self.logger.info(
            f"PDF erfolgreich geladen: {document_name}\n"
            f"- {page_count} Seiten\n"
            f"- {chunk_count} Chunks (Ø {avg_chunk_size:.0f} Wörter/Chunk)"
        ) 