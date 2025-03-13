from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
import logging
import PyPDF2
import anthropic

# Konfigurieren Sie das Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Flask-App initialisieren
app = Flask(__name__)
CORS(app)  # CORS für alle Routen aktivieren

# Anthropic-Client initialisieren (falls API-Key vorhanden)
anthropic_client = None
if os.environ.get('ANTHROPIC_API_KEY'):
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

# Hilfsfunktionen
def extract_text_from_pdf(pdf_path):
    """Extrahiert Text aus einer PDF-Datei."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren des Textes: {str(e)}")
        raise

def get_direct_answer(query, text):
    """
    Versucht, eine direkte Antwort auf eine Frage im Text zu finden.
    
    Args:
        query (str): Die Suchanfrage/Frage
        text (str): Der Text, in dem gesucht werden soll
    
    Returns:
        dict: Ein Dictionary mit der gefundenen Antwort oder einer Fehlermeldung
    """
    logger.debug(f"get_direct_answer aufgerufen mit Query: '{query}'")
    logger.debug(f"Text-Länge: {len(text)} Zeichen")
    
    if not query or not text:
        logger.warning("Query oder Text ist leer")
        return {"direct_answer": None, "error": "Query oder Text ist leer"}
    
    # Bereinigen der Query für die Regex-Suche
    cleaned_query = query.lower().strip()
    cleaned_query = re.sub(r'[?.,!]', '', cleaned_query)
    logger.debug(f"Bereinigte Query: '{cleaned_query}'")
    
    # Extrahiere wichtige Schlüsselwörter (Wörter mit mehr als 3 Buchstaben)
    keywords = [word for word in cleaned_query.split() if len(word) > 3]
    logger.debug(f"Extrahierte Schlüsselwörter: {keywords}")
    
    if not keywords:
        # Wenn keine langen Schlüsselwörter gefunden wurden, verwende alle Wörter
        keywords = cleaned_query.split()
        logger.debug(f"Keine langen Schlüsselwörter gefunden, verwende alle Wörter: {keywords}")
    
    # Verschiedene Suchmuster ausprobieren
    patterns = [
        # Muster 1: Sätze, die alle Schlüsselwörter enthalten
        rf"(?i)[^.!?]*{' '.join([rf'.*\b{re.escape(word)}\b' for word in keywords])}.*?[.!?]",
        
        # Muster 2: Sätze, die mindestens ein Schlüsselwort enthalten
        rf"(?i)[^.!?]*({'|'.join([rf'\b{re.escape(word)}\b' for word in keywords])}).*?[.!?]"
    ]
    
    for i, pattern in enumerate(patterns):
        logger.debug(f"Versuche Muster {i+1}: {pattern}")
        try:
            matches = re.findall(pattern, text)
            logger.debug(f"Gefundene Übereinstimmungen für Muster {i+1}: {len(matches) if isinstance(matches, list) else 'Nicht-Liste-Typ'}")
            
            # Stelle sicher, dass matches eine Liste ist
            if isinstance(matches, list) and matches:
                if isinstance(matches[0], tuple):  # Bei Gruppen in der Regex
                    logger.debug("Matches sind Tupel, extrahiere vollständige Übereinstimmungen")
                    # Extrahiere den vollständigen Text aus dem Text
                    full_matches = []
                    for match in matches:
                        # Suche nach dem ersten Vorkommen des Schlüsselworts im Text
                        keyword = match[0] if match else keywords[0]
                        start_idx = text.lower().find(keyword.lower())
                        if start_idx >= 0:
                            # Finde den Satz, der dieses Schlüsselwort enthält
                            sentence_start = text.rfind('.', 0, start_idx) + 1
                            sentence_end = text.find('.', start_idx)
                            if sentence_end == -1:
                                sentence_end = len(text)
                            full_matches.append(text[sentence_start:sentence_end].strip())
                    matches = full_matches
                
                # Sortiere Übereinstimmungen nach Relevanz (Anzahl der enthaltenen Schlüsselwörter)
                matches = sorted(matches, key=lambda x: sum(word.lower() in x.lower() for word in keywords), reverse=True)
                
                # Logge die ersten 3 Übereinstimmungen
                for j, match in enumerate(matches[:3]):
                    logger.debug(f"Match {j+1}: {match[:100]}...")
                
                return {
                    "direct_answer": matches[0].strip(),
                    "matches": [m.strip() for m in matches[:3]],
                    "pattern_used": i+1
                }
        except Exception as e:
            logger.error(f"Fehler bei Muster {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Fallback: Einfache Satzsuche mit Schlüsselwörtern
    logger.debug("Keine Übereinstimmungen gefunden, versuche Fallback-Methode")
    try:
        # Teile den Text in Sätze auf
        sentences = re.split(r'[.!?]', text)
        relevant_sentences = []
        
        logger.debug(f"Anzahl der Sätze im Text: {len(sentences)}")
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            relevance_score = sum(keyword.lower() in sentence.lower() for keyword in keywords)
            if relevance_score > 0:
                relevant_sentences.append((sentence, relevance_score))
        
        logger.debug(f"Gefundene relevante Sätze: {len(relevant_sentences)}")
        
        if relevant_sentences:
            # Sortiere nach Relevanz
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Logge die ersten 3 relevanten Sätze
            for i, (sentence, score) in enumerate(relevant_sentences[:3]):
                logger.debug(f"Relevanter Satz {i+1} (Score {score}): {sentence[:100]}...")
            
            return {
                "direct_answer": relevant_sentences[0][0],
                "matches": [s[0] for s in relevant_sentences[:3]],
                "pattern_used": "fallback"
            }
    except Exception as e:
        logger.error(f"Fehler bei Fallback-Methode: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Letzte Fallback-Methode: Einfache Wortsuche
    logger.debug("Versuche letzte Fallback-Methode: Einfache Wortsuche")
    try:
        # Finde Absätze, die Schlüsselwörter enthalten
        paragraphs = text.split('\n')
        relevant_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            relevance_score = sum(keyword.lower() in paragraph.lower() for keyword in keywords)
            if relevance_score > 0:
                relevant_paragraphs.append((paragraph, relevance_score))
        
        logger.debug(f"Gefundene relevante Absätze: {len(relevant_paragraphs)}")
        
        if relevant_paragraphs:
            # Sortiere nach Relevanz
            relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
            
            # Logge die ersten 3 relevanten Absätze
            for i, (paragraph, score) in enumerate(relevant_paragraphs[:3]):
                logger.debug(f"Relevanter Absatz {i+1} (Score {score}): {paragraph[:100]}...")
            
            return {
                "direct_answer": relevant_paragraphs[0][0],
                "matches": [p[0] for p in relevant_paragraphs[:3]],
                "pattern_used": "word_search"
            }
    except Exception as e:
        logger.error(f"Fehler bei letzter Fallback-Methode: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.warning("Keine Übereinstimmungen gefunden")
    return {"direct_answer": None, "error": "Keine Übereinstimmungen gefunden"}

# API-Routen
@app.route('/', methods=['GET'])
def index():
    """Startseite der API."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF-Verarbeitung API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { background-color: #f5f5f5; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>PDF-Verarbeitung API</h1>
        <p>Willkommen bei der PDF-Verarbeitung API.</p>
        
        <div class="endpoint">
            <h3>API-Test</h3>
            <p><a href="/api/hello">Testen Sie den API-Endpunkt</a></p>
        </div>
        
        <div class="endpoint">
            <h3>PDF-Verarbeitung</h3>
            <p>Endpunkt: <code>/process_pdf</code> (POST)</p>
        </div>
        
        <div class="endpoint">
            <h3>Suche</h3>
            <p>Endpunkt: <code>/search</code> (POST)</p>
        </div>
        
        <div class="endpoint">
            <h3>Anthropic-Anfrage</h3>
            <p>Endpunkt: <code>/ask_anthropic</code> (POST)</p>
        </div>
        
        <div class="endpoint">
            <h3>Test-Suche</h3>
            <p><a href="/test_search?query=Was%20ist%20ein%20PDF">Testen Sie die Suchfunktion</a></p>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/api/hello', methods=['GET'])
def api_hello():
    """Einfacher Test-Endpunkt."""
    return jsonify({"message": "Hello world"})

@app.route('/say_hello', methods=['GET'])
def say_hello():
    """Einfacher Test-Endpunkt."""
    return jsonify({"text": "hello world"})

@app.route('/test', methods=['GET'])
def test():
    """Test-Endpunkt."""
    return jsonify({"message": "Test erfolgreich"})

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    """Verarbeitet eine PDF-Datei und gibt den extrahierten Text zurück."""
    # Debug-Ausgabe: Prüfen, welche Dateien im Request enthalten sind
    logger.debug(f"Verfügbare Dateien im Request: {list(request.files.keys())}")
    
    if 'file' not in request.files:
        logger.warning("Keine Datei mit dem Namen 'file' im Request gefunden")
        return jsonify({'error': 'Keine Datei im Request. Bitte senden Sie eine Datei mit dem Feldnamen "file"'}), 400
    
    file = request.files['file']
    logger.debug(f"Dateiname: {file.filename}, Typ: {type(file)}")
    
    if file.filename == '':
        logger.warning("Leerer Dateiname")
        return jsonify({'error': 'Kein Dateiname angegeben'}), 400
    
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Ungültiges Dateiformat: {file.filename}")
        return jsonify({'error': f'Ungültiges Dateiformat. Nur PDF-Dateien sind erlaubt. Erhalten: {file.filename}'}), 400
    
    try:
        # Speichern der hochgeladenen Datei
        upload_folder = '/tmp/uploads' if os.environ.get('VERCEL_ENV') else 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        logger.debug(f"Speichere Datei unter: {file_path}")
        file.save(file_path)
        
        # Text aus PDF extrahieren
        logger.debug("Extrahiere Text aus PDF...")
        text = extract_text_from_pdf(file_path)
        logger.debug(f"Extrahierter Text (erste 100 Zeichen): {text[:100]}...")
        
        return jsonify({'text': text})
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Fehler bei der Verarbeitung: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search():
    """Sucht nach einer Antwort im extrahierten Text."""
    data = request.json
    logger.debug(f"Suchanfrage erhalten: {data}")
    
    if not data or 'query' not in data or 'text' not in data:
        logger.warning("Ungültige Anfrage: query oder text fehlt")
        return jsonify({'error': 'Ungültige Anfrage. "query" und "text" sind erforderlich.'}), 400
    
    query = data['query']
    text = data['text']
    
    # Direkte Antwort suchen
    result = get_direct_answer(query, text)
    logger.debug(f"Suchergebnis: {result}")
    
    return jsonify({
        'query': query,
        'results': {
            'direct_answer': result.get('direct_answer'),
            'matches': result.get('matches', []),
            'pattern_used': result.get('pattern_used'),
            'error': result.get('error')
        }
    })
