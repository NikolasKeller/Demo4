import PyPDF2
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import gc
import uuid
import time

print("Skript wird gestartet...")

def process_pdf(pdf_file, model, collection, chunk_size=1000, chunk_overlap=100):
    """
    Verarbeitet eine PDF-Datei und fügt die Chunks zur Datenbank hinzu.
    """
    print(f"Verarbeite PDF: {os.path.basename(pdf_file)}")
    
    try:
        # PDF öffnen und Text extrahieren
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"  PDF hat {total_pages} Seiten")
            
            # Maximal 5 Seiten verarbeiten für schnelle Tests
            max_pages = min(5, total_pages)
            print(f"  Verarbeite die ersten {max_pages} Seiten")
            
            # Jede Seite einzeln verarbeiten
            for page_num in range(max_pages):
                start_time = time.time()
                print(f"  Verarbeite Seite {page_num+1}/{max_pages}")
                
                # Text extrahieren
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                print(f"    Extrahierter Text: {len(page_text)} Zeichen")
                
                # Speicher freigeben
                page = None
                gc.collect()
                
                if not page_text.strip():
                    print(f"    Seite {page_num+1} enthält keinen Text, überspringe...")
                    continue
                
                # Text in Chunks aufteilen
                chunks = []
                ids = []
                metadatas = []
                
                for i in range(0, len(page_text), chunk_size - chunk_overlap):
                    chunk = page_text[i:i + chunk_size]
                    if len(chunk.strip()) < 10:  # Überspringe sehr kurze Chunks
                        continue
                    
                    chunk_id = f"{os.path.basename(pdf_file)}_p{page_num+1}_c{i}_{uuid.uuid4().hex[:6]}"
                    
                    chunks.append(chunk)
                    ids.append(chunk_id)
                    metadatas.append({
                        "source": os.path.basename(pdf_file),
                        "page": page_num + 1,
                        "position": i
                    })
                
                # Embeddings erstellen und zur Datenbank hinzufügen
                if chunks:
                    print(f"    Erstelle Embeddings für {len(chunks)} Chunks...")
                    embeddings = model.encode(chunks)
                    
                    print(f"    Füge Embeddings zur Datenbank hinzu...")
                    collection.add(
                        documents=chunks,
                        embeddings=embeddings.tolist(),
                        metadatas=metadatas,
                        ids=ids
                    )
                    
                    print(f"    {len(chunks)} Chunks zur Datenbank hinzugefügt")
                else:
                    print("    Keine gültigen Chunks gefunden, überspringe...")
                
                elapsed_time = time.time() - start_time
                print(f"    Seite {page_num+1} in {elapsed_time:.2f} Sekunden verarbeitet")
        
        print(f"  PDF erfolgreich verarbeitet")
        return True
    
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {pdf_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Hauptskript ---
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
    
    # Nur die erste PDF verarbeiten
    pdf_files = pdf_files[:1]
    print(f"Verarbeite PDF: {os.path.basename(pdf_files[0])}")
    
    # Mehrsprachiges Modell laden
    print("Lade mehrsprachiges SentenceTransformer-Modell...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # ChromaDB initialisieren
    print("Initialisiere ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
    
    # Collection erstellen oder leeren
    collection_name = "pdf_collection"
    try:
        client.delete_collection(collection_name)
        print(f"Bestehende Collection '{collection_name}' gelöscht")
    except:
        print(f"Keine bestehende Collection '{collection_name}' gefunden")
    
    collection = client.create_collection(collection_name)
    print(f"Neue Collection '{collection_name}' erstellt")
    
    # PDF verarbeiten
    success = process_pdf(
        pdf_files[0], 
        model, 
        collection, 
        chunk_size=1000,  # Kleinere Chunks für bessere Suche
        chunk_overlap=100  # Überlappung für Kontext
    )
    
    if success:
        print("\nPDF erfolgreich verarbeitet und zur Datenbank hinzugefügt")
        print(f"Die Datenbank enthält jetzt {collection.count()} Chunks")
        
        print("\nSie können jetzt das search_pdf.py-Skript ausführen, um Suchanfragen zu stellen")
        print("Beispiel: 'Was ist die Warenart?' oder 'Wie hoch ist die Betriebsspannung?'")
    else:
        print("\nFehler beim Verarbeiten der PDF")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
    import traceback
    traceback.print_exc()

print("Skript beendet.")