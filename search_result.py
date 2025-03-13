from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json

@dataclass
class SearchResult:
    text: str
    document: str
    page: int
    score: float
    chunk: int
    context: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'document': self.document,
            'page': self.page,
            'score': self.score,
            'chunk': self.chunk,
            'context': self.context
        }

class SearchResultFormatter:
    def __init__(self, 
                 show_scores: bool = True,
                 max_context_length: int = 100,
                 min_score_threshold: float = 0.3):
        self.show_scores = show_scores
        self.max_context_length = max_context_length
        self.min_score_threshold = min_score_threshold
    
    def format_single_result(self, result: SearchResult, index: int) -> str:
        """Formatiert ein einzelnes Suchergebnis"""
        output = []
        
        # Überschrift mit Dokumentinformationen
        output.append(f"\n{'='*80}")
        output.append(f"Ergebnis {index + 1}")
        output.append(f"Dokument: {result.document} (Seite {result.page})")
        
        if self.show_scores:
            score_percent = result.score * 100
            output.append(f"Relevanz: {score_percent:.1f}%")
        
        output.append(f"{'-'*80}")
        
        # Haupttext
        output.append(f"{result.text}")
        
        # Kontext (falls vorhanden)
        if result.context:
            output.append(f"\nKontext:")
            output.append(f"...{result.context}...")
            
        return "\n".join(output)
    
    def format_results(self, results: List[SearchResult], query: str) -> str:
        """Formatiert eine Liste von Suchergebnissen"""
        if not results:
            return "Keine relevanten Ergebnisse gefunden."
        
        # Filtern nach Relevanz-Schwellenwert
        filtered_results = [r for r in results if r.score >= self.min_score_threshold]
        
        if not filtered_results:
            return "Keine ausreichend relevanten Ergebnisse gefunden."
        
        output = []
        
        # Zusammenfassung
        output.append(f"Suchergebnisse für: '{query}'")
        output.append(f"Gefunden: {len(filtered_results)} relevante Stellen\n")
        
        # Einzelne Ergebnisse
        for i, result in enumerate(filtered_results):
            output.append(self.format_single_result(result, i))
        
        return "\n".join(output)
    
    def to_json(self, results: List[SearchResult], query: str) -> str:
        """Konvertiert Ergebnisse in JSON-Format"""
        output = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'results': [result.to_dict() for result in results]
        }
        return json.dumps(output, indent=2, ensure_ascii=False) 