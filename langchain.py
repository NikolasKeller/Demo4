import os
import PyPDF2
import re

print("Einfaches PDF-Suchskript wird gestartet...")

# Funktion zum Extrahieren von Text aus einer PDF
def extract_text_from_pdf(pdf_path):
    print(f"Extrahiere Text aus: {os.path.basename(pdf_path)}")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Fehler beim Lesen der PDF: {e}")
    
    return text

# Funktion zum Suchen nach einer direkten Antwort
def get_direct_answer(text, query):
    # Entferne Fragewörter und Artikel
    clean_words = ["ist", "die", "der", "das", "was", "wie", "wo", "wann", "warum", "welche", "welcher", "welches"]
    clean_query = query.lower()
    for word in clean_words:
        clean_query = clean_query.replace(word, "").strip()
    
    # Suche nach dem Muster "Suchbegriff: Wert"
    pattern = rf'{clean_query}[^:]*?:\s*([^\n,]+)'
    match = re.search(pattern, text.lower())
    
    if match:
        return match.group(1).strip()
    else:
        return None

# Hauptprogramm
try:
    # PDF-Verzeichnis
    pdf_directory = r"C:\Users\niko_\OneDrive\Dokumente\PDF DRYFT"
    
    if not os.path.exists(pdf_directory):
        print(f"FEHLER: Das Verzeichnis {pdf_directory} existiert nicht!")
        exit(1)
    
    # PDF-Dateien finden
    pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
    
    if not pdf_files:
        print("WARNUNG: Keine PDF-Dateien gefunden!")
        exit(1)
    
    print(f"Gefundene PDF-Dateien: {len(pdf_files)}")
    
    # Alle PDFs verarbeiten und Text extrahieren
    all_text = ""
    for pdf_file in pdf_files[:5]:  # Maximal 5 PDFs
        print(f"Verarbeite PDF: {os.path.basename(pdf_file)}")
        pdf_text = extract_text_from_pdf(pdf_file)
        all_text += pdf_text + "\n\n"
    
    # Interaktive Suche
    print("\nAlle PDFs wurden verarbeitet. Sie können jetzt Fragen stellen.")
    print("Beispiel: 'Was ist die Warenart?' oder 'Umgebungstemperatur Maschine'")
    
    while True:
        user_query = input("\nGeben Sie Ihre Suchanfrage ein (oder 'exit' zum Beenden): ")
        
        if user_query.lower() == 'exit':
            break
        
        answer = get_direct_answer(all_text, user_query)
        
        if answer:
            print(f"\nAntwort: {answer}")
        else:
            print("\nKeine Antwort gefunden.")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
    import traceback
    traceback.print_exc()

print("Suche beendet.")