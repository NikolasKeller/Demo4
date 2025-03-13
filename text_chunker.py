from typing import List, Optional, Tuple
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize
import re
import logging
from enum import Enum

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChunkingStrategy(Enum):
    CHARACTER = "character"
    SENTENCE = "sentence"
    AUTO = "auto"  # Wählt die beste Strategie basierend auf dem Text

@dataclass
class ChunkingConfig:
    """Konfiguration für das Text-Chunking"""
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    respect_paragraphs: bool = True
    max_sentence_length: int = 1000  # Für sehr lange Sätze

@dataclass
class Chunk:
    """Repräsentiert einen Textabschnitt"""
    text: str
    start_pos: int
    end_pos: int
    is_sentence_boundary: bool
    size: int = 0
    
    def __post_init__(self):
        self.size = len(self.text)

class TextChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.logger = logging.getLogger(__name__)
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Teilt Text in Chunks unter Berücksichtigung verschiedener Strategien
        """
        if not text:
            return []
        
        # Strategie bestimmen
        strategy = self._determine_strategy(text)
        self.logger.info(f"Verwende Chunking-Strategie: {strategy.value}")
        
        # Text vorverarbeiten
        cleaned_text = self._preprocess_text(text)
        
        if strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_based_chunking(cleaned_text)
        else:
            return self._character_based_chunking(cleaned_text)
    
    def _determine_strategy(self, text: str) -> ChunkingStrategy:
        """Bestimmt die beste Chunking-Strategie"""
        if self.config.strategy != ChunkingStrategy.AUTO:
            return self.config.strategy
            
        # Heuristik für die Strategiewahl
        avg_sentence_length = self._estimate_avg_sentence_length(text)
        has_clear_sentences = '.' in text and '?' in text or '!' in text
        
        if has_clear_sentences and avg_sentence_length < self.config.chunk_size:
            return ChunkingStrategy.SENTENCE
        return ChunkingStrategy.CHARACTER
    
    def _preprocess_text(self, text: str) -> str:
        """Bereinigt und normalisiert den Text"""
        # Mehrfache Leerzeichen entfernen
        text = re.sub(r'\s+', ' ', text)
        
        # Absätze erhalten wenn gewünscht
        if self.config.respect_paragraphs:
            text = re.sub(r'\n\s*\n', '\n\n', text)
        else:
            text = text.replace('\n', ' ')
        
        return text.strip()
    
    def _sentence_based_chunking(self, text: str) -> List[Chunk]:
        """
        Teilt Text in Chunks basierend auf Satzgrenzen
        """
        chunks = []
        current_chunk = []
        current_size = 0
        current_start = 0
        
        # Text in Sätze teilen
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_length = len(sentence)
            
            # Sehr lange Sätze aufteilen
            if sentence_length > self.config.max_sentence_length:
                if current_chunk:
                    # Aktuellen Chunk abschließen
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_pos=current_start,
                        end_pos=current_start + len(chunk_text),
                        is_sentence_boundary=True
                    ))
                    
                # Langen Satz separat verarbeiten
                sentence_chunks = self._split_long_sentence(
                    sentence, 
                    current_start + len(' '.join(current_chunk)) + 1
                )
                chunks.extend(sentence_chunks)
                
                current_chunk = []
                current_size = 0
                current_start = chunks[-1].end_pos if chunks else 0
                continue
            
            # Prüfen, ob der Satz in den aktuellen Chunk passt
            if current_size + sentence_length <= self.config.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_length + 1  # +1 für Leerzeichen
            else:
                # Aktuellen Chunk abschließen wenn nicht zu klein
                if current_size >= self.config.min_chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_pos=current_start,
                        end_pos=current_start + len(chunk_text),
                        is_sentence_boundary=True
                    ))
                    
                    # Neuen Chunk mit Überlappung beginnen
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk,
                        self.config.chunk_overlap
                    )
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                    current_start = chunks[-1].end_pos - sum(len(s) for s in overlap_sentences)
                else:
                    # Chunk ist zu klein, weiteren Satz hinzufügen
                    current_chunk.append(sentence)
                    current_size += sentence_length + 1
        
        # Letzten Chunk hinzufügen
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_pos=current_start,
                end_pos=current_start + len(chunk_text),
                is_sentence_boundary=True
            ))
        
        return chunks
    
    def _character_based_chunking(self, text: str) -> List[Chunk]:
        """
        Teilt Text in Chunks basierend auf Zeichenanzahl
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Ende des aktuellen Chunks bestimmen
            end = min(start + self.config.chunk_size, text_length)
            
            # Wenn nicht am Textende, nach Wortgrenze suchen
            if end < text_length:
                while end > start and not text[end].isspace():
                    end -= 1
                if end == start:  # Kein Leerzeichen gefunden
                    end = min(start + self.config.chunk_size, text_length)
            
            # Chunk extrahieren
            chunk_text = text[start:end].strip()
            
            # Nur hinzufügen wenn Mindestgröße erreicht
            if len(chunk_text) >= self.config.min_chunk_size or end == text_length:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    is_sentence_boundary=False
                ))
            
            # Nächste Startposition mit Überlappung
            start = max(end - self.config.chunk_overlap, start + 1)
        
        return chunks
    
    def _split_long_sentence(self, sentence: str, start_pos: int) -> List[Chunk]:
        """Teilt einen sehr langen Satz in kleinere Chunks"""
        chunks = []
        words = sentence.split()
        current_chunk = []
        current_size = 0
        chunk_start = start_pos
        
        for word in words:
            word_length = len(word)
            if current_size + word_length <= self.config.chunk_size:
                current_chunk.append(word)
                current_size += word_length + 1  # +1 für Leerzeichen
            else:
                # Aktuellen Chunk abschließen
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_pos=chunk_start,
                        end_pos=chunk_start + len(chunk_text),
                        is_sentence_boundary=False
                    ))
                    
                    # Neuen Chunk beginnen
                    current_chunk = [word]
                    current_size = word_length
                    chunk_start = chunks[-1].end_pos
                else:
                    # Wort ist länger als chunk_size
                    chunks.append(Chunk(
                        text=word,
                        start_pos=chunk_start,
                        end_pos=chunk_start + word_length,
                        is_sentence_boundary=False
                    ))
                    chunk_start += word_length
                    current_chunk = []
                    current_size = 0
        
        # Letzten Chunk hinzufügen
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_pos=chunk_start,
                end_pos=chunk_start + len(chunk_text),
                is_sentence_boundary=False
            ))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], target_overlap: int) -> List[str]:
        """Wählt Sätze für die Überlappung aus"""
        overlap_size = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if overlap_size >= target_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_size += len(sentence) + 1
        
        return overlap_sentences
    
    def _estimate_avg_sentence_length(self, text: str) -> float:
        """Schätzt die durchschnittliche Satzlänge"""
        sentences = sent_tokenize(text[:min(len(text), 1000)])  # Erste 1000 Zeichen als Sample
        if not sentences:
            return float('inf')
        return sum(len(s) for s in sentences) / len(sentences)

# Beispielverwendung
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Beispieltext
    text = """
    Dies ist ein Beispieltext mit mehreren Sätzen. Einige Sätze sind kurz.
    Andere Sätze sind deutlich länger und enthalten mehr Informationen über
    das Thema, das wir hier behandeln. Dies ist ein weiterer Satz.
    
    Dies ist ein neuer Absatz. Er enthält auch mehrere Sätze mit
    unterschiedlichen Längen und Strukturen.
    """
    
    # Chunker mit verschiedenen Konfigurationen testen
    configs = [
        ChunkingConfig(strategy=ChunkingStrategy.SENTENCE),
        ChunkingConfig(strategy=ChunkingStrategy.CHARACTER),
        ChunkingConfig(strategy=ChunkingStrategy.AUTO)
    ]
    
    chunker = TextChunker()
    for config in configs:
        chunker.config = config
        print(f"\nTeste {config.strategy.value}-basiertes Chunking:")
        chunks = chunker.chunk_text(text)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i} ({chunk.size} Zeichen):")
            print(f"Text: {chunk.text}")
            print(f"Position: {chunk.start_pos}-{chunk.end_pos}")
            print(f"Satzgrenze: {chunk.is_sentence_boundary}") 