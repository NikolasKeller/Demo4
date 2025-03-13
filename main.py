from pdf_processor import PDFSearchEngine
import logging
import json

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Suchmaschine initialisieren
    search_engine = PDFSearchEngine(persist_directory="./chroma_db")
    
    # PDFs laden
    pdf_files = ['dokument1.pdf', 'dokument2.pdf']
    
    for pdf_file in pdf_files:
        if search_engine.load_pdf(pdf_file):
            logger.info(f"PDF erfolgreich geladen: {pdf_file}")
        else:
            logger.error(f"PDF konnte nicht geladen werden: {pdf_file}")
    
    # Beispielsuche
    query = "Was sind die Hauptvorteile von erneuerbaren Energien?"
    
    try:
        # Formatierte Konsolenausgabe
        results_text = search_engine.search(
            query=query,
            top_k=3,
            format_output=True
        )
        print(results_text)
        
        # Optional: JSON-Ausgabe für API-Nutzung
        results = search_engine.search(
            query=query,
            top_k=3,
            format_output=False
        )
        json_output = search_engine.result_formatter.to_json(results, query)
        
        # In Datei speichern
        with open('search_results.json', 'w', encoding='utf-8') as f:
            f.write(json_output)
            
    except Exception as e:
        logger.error(f"Fehler bei der Suche: {str(e)}")

if __name__ == "__main__":
    main() 