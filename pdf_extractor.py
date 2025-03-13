from PyPDF2 import PdfReader, PdfReadError
import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path

@dataclass
class ExtractionResult:
    """Ergebnis der PDF-Extraktion"""
    success: bool
    text: Optional[str] = None
    error_message: Optional[str] = None
    page_count: int = 0
    empty_pages: List[int] = None
    
    def __post_init__(self):
        if self.empty_pages is None:
            self.empty_pages = []

class PDFExtractor:
    def __init__(self, log_file: str = "pdf_extraction.log"):
        # Logger konfigurieren
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # File Handler für detaillierte Logs
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Statistiken initialisieren
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_pages': 0,
            'empty_pages': 0
        }
        
    def extract_text_from_pdf(self, file_path: str) -> ExtractionResult:
        """
        Extrahiert Text aus einer PDF-Datei mit verbesserter Fehlerbehandlung
        """
        path = Path(file_path)
        
        try:
            # Überprüfen, ob die Datei existiert und lesbar ist
            if not path.exists():
                return ExtractionResult(
                    success=False,
                    error_message=f"Datei nicht gefunden: {file_path}"
                )
            
            if not os.access(path, os.R_OK):
                return ExtractionResult(
                    success=False,
                    error_message=f"Keine Leserechte für: {file_path}"
                )
            
            # PDF öffnen und verarbeiten
            reader = PdfReader(file_path)
            extracted_text = []
            empty_pages = []
            
            # Seitenzahl protokollieren
            total_pages = len(reader.pages)
            self.stats['total_pages'] += total_pages
            
            # Text von jeder Seite extrahieren
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text() or ""
                    page_text = page_text.strip()
                    
                    if page_text:
                        extracted_text.append(page_text)
                    else:
                        empty_pages.append(page_num)
                        self.stats['empty_pages'] += 1
                        self.logger.warning(
                            f"Leere oder nicht extrahierbare Seite {page_num} in {path.name}"
                        )
                        
                except Exception as e:
                    self.logger.error(
                        f"Fehler bei der Extraktion von Seite {page_num} in {path.name}: {str(e)}"
                    )
                    empty_pages.append(page_num)
            
            # Ergebnis zusammenstellen
            if not extracted_text:
                return ExtractionResult(
                    success=False,
                    error_message="Keine verwertbaren Textinhalte gefunden",
                    page_count=total_pages,
                    empty_pages=empty_pages
                )
            
            return ExtractionResult(
                success=True,
                text="\n\n".join(extracted_text),
                page_count=total_pages,
                empty_pages=empty_pages
            )
            
        except PdfReadError as e:
            self.logger.error(f"PDF-Lesefehler in {path.name}: {str(e)}")
            return ExtractionResult(
                success=False,
                error_message=f"Ungültiges oder beschädigtes PDF: {str(e)}"
            )
            
        except Exception as e:
            self.logger.error(f"Unerwarteter Fehler bei {path.name}: {str(e)}")
            return ExtractionResult(
                success=False,
                error_message=f"Extraktionsfehler: {str(e)}"
            )
    
    def process_directory(self, directory: str = ".") -> Dict[str, ExtractionResult]:
        """
        Verarbeitet alle PDF-Dateien in einem Verzeichnis
        """
        results = {}
        directory = Path(directory)
        
        # Alle PDF-Dateien im Verzeichnis finden
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"Keine PDF-Dateien gefunden in: {directory}")
            return results
        
        self.logger.info(f"Starte Verarbeitung von {len(pdf_files)} PDF-Dateien")
        
        # Jede PDF-Datei verarbeiten
        for pdf_file in pdf_files:
            self.stats['processed'] += 1
            self.logger.info(f"Verarbeite {pdf_file.name} ({self.stats['processed']}/{len(pdf_files)})")
            
            result = self.extract_text_from_pdf(str(pdf_file))
            results[pdf_file.name] = result
            
            if result.success:
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
        
        # Zusammenfassung ausgeben
        self._print_summary()
        
        return results
    
    def _print_summary(self):
        """Gibt eine Zusammenfassung der Verarbeitung aus"""
        success_rate = (self.stats['successful'] / self.stats['processed'] * 100) if self.stats['processed'] > 0 else 0
        empty_page_rate = (self.stats['empty_pages'] / self.stats['total_pages'] * 100) if self.stats['total_pages'] > 0 else 0
        
        summary = [
            "\n=== Verarbeitungszusammenfassung ===",
            f"Verarbeitete Dateien: {self.stats['processed']}",
            f"Erfolgreich: {self.stats['successful']} ({success_rate:.1f}%)",
            f"Fehlgeschlagen: {self.stats['failed']}",
            f"Gesamtseitenzahl: {self.stats['total_pages']}",
            f"Leere/Nicht extrahierbare Seiten: {self.stats['empty_pages']} ({empty_page_rate:.1f}%)",
            "=================================="
        ]
        
        summary_text = "\n".join(summary)
        print(summary_text)
        self.logger.info(summary_text)

# Beispielverwendung
if __name__ == "__main__":
    # Logger für Konsolenausgabe konfigurieren
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    
    extractor = PDFExtractor()
    results = extractor.process_directory("./pdfs")
    
    # Detaillierte Ergebnisse für jede Datei
    for filename, result in results.items():
        if not result.success:
            print(f"\nFehler bei {filename}: {result.error_message}")
        else:
            print(f"\nErfolgreich verarbeitet: {filename}")
            print(f"- Seitenzahl: {result.page_count}")
            if result.empty_pages:
                print(f"- Leere Seiten: {result.empty_pages}") 