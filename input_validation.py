from typing import Optional, List
from pathlib import Path
import magic
import re
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    error_message: Optional[str] = None

class InputValidator:
    def __init__(self):
        self.mime = magic.Magic(mime=True)
        
    def validate_pdf_file(self, file_path: str) -> ValidationResult:
        """Überprüft, ob eine Datei ein gültiges PDF ist"""
        try:
            path = Path(file_path)
            
            # Grundlegende Dateiprüfungen
            if not path.exists():
                return ValidationResult(False, f"Datei nicht gefunden: {file_path}")
                
            if not path.is_file():
                return ValidationResult(False, f"Pfad ist keine Datei: {file_path}")
                
            if path.suffix.lower() != '.pdf':
                return ValidationResult(False, f"Keine PDF-Datei: {file_path}")
                
            # MIME-Typ prüfen
            mime_type = self.mime.from_file(str(path))
            if mime_type != 'application/pdf':
                return ValidationResult(False, f"Ungültiger PDF-Dateityp: {mime_type}")
                
            # Dateigröße prüfen (max. 100MB)
            max_size = 100 * 1024 * 1024  # 100MB in Bytes
            if path.stat().st_size > max_size:
                return ValidationResult(False, f"PDF-Datei zu groß (max. 100MB)")
            
            return ValidationResult(True)
            
        except Exception as e:
            return ValidationResult(False, f"Fehler bei der PDF-Validierung: {str(e)}")
    
    def validate_query(self, query: str) -> ValidationResult:
        """Überprüft eine Suchanfrage auf Gültigkeit"""
        if not query or not query.strip():
            return ValidationResult(False, "Suchanfrage darf nicht leer sein")
            
        # Mindest- und Maximallänge prüfen
        if len(query) < 3:
            return ValidationResult(False, "Suchanfrage zu kurz (min. 3 Zeichen)")
            
        if len(query) > 500:
            return ValidationResult(False, "Suchanfrage zu lang (max. 500 Zeichen)")
            
        # Unerwünschte Zeichen entfernen
        cleaned_query = re.sub(r'[^\w\s\-.,?!]', '', query)
        if not cleaned_query:
            return ValidationResult(False, "Suchanfrage enthält keine gültigen Zeichen")
            
        return ValidationResult(True)
    
    def validate_search_params(self, 
                             top_k: int,
                             min_score: float) -> ValidationResult:
        """Überprüft Suchparameter"""
        if not isinstance(top_k, int):
            return ValidationResult(False, "top_k muss eine ganze Zahl sein")
            
        if top_k < 1:
            return ValidationResult(False, "top_k muss mindestens 1 sein")
            
        if top_k > 100:
            return ValidationResult(False, "top_k darf maximal 100 sein")
            
        if not 0 <= min_score <= 1:
            return ValidationResult(False, "min_score muss zwischen 0 und 1 liegen")
            
        return ValidationResult(True) 