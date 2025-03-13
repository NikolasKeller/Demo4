from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import re
import os
from anthropic import Anthropic

app = Flask(__name__)
CORS(app)

# Sichere Handhabung des API-Schlüssels
# 1. Laden aus Umgebungsvariablen (für Produktionsumgebungen)
# 2. Alternativ aus .env-Datei laden (für Entwicklungsumgebungen)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Lädt Variablen aus .env-Datei
except ImportError:
    # Falls python-dotenv nicht installiert ist
    pass

# Anthropic-Client initialisieren
# Der API-Schlüssel wird automatisch aus Umgebungsvariablen geladen
anthropic_client = Anthropic()

def extract_text_from_pdf(pdf_path):
    """Extrahiert Text aus einer PDF-Datei."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def search_with_regex(text, pattern):
    """Sucht nach einem Regex-Muster im Text."""
    matches = re.finditer(pattern, text, re.MULTILINE)
    results = []
    for match in matches:
        results.append({
            'text': match.group(),
            'start': match.start(),
            'end': match.end()
        })
    return results

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    """Verarbeitet eine PDF-Datei und gibt den extrahierten Text zurück."""
    if 'file' not in request.files:
        return jsonify({'error': 'Keine Datei im Request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Kein Dateiname angegeben'}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Speichern der hochgeladenen Datei
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        
        # Text aus PDF extrahieren
        text = extract_text_from_pdf(file_path)
        
        return jsonify({'text': text})
    
    return jsonify({'error': 'Ungültiges Dateiformat'}), 400

@app.route('/search', methods=['POST'])
def search():
    """Sucht nach einem Regex-Muster im Text."""
    data = request.json
    if not data or 'text' not in data or 'pattern' not in data:
        return jsonify({'error': 'Text und Suchmuster erforderlich'}), 400
    
    text = data['text']
    pattern = data['pattern']
    
    try:
        results = search_with_regex(text, pattern)
        return jsonify({'results': results})
    except re.error as e:
        return jsonify({'error': f'Ungültiges Regex-Muster: {str(e)}'}), 400

@app.route('/ask_anthropic', methods=['POST'])
def ask_anthropic():
    """Sendet eine Anfrage an die Anthropic API."""
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt erforderlich'}), 400
    
    prompt = data['prompt']
    
    try:
        # Anfrage an Anthropic senden
        message = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Antwort zurückgeben
        return jsonify({'response': message.content[0].text})
    except Exception as e:
        return jsonify({'error': f'Fehler bei der Anfrage an Anthropic: {str(e)}'}), 500
    
@app.route('/say_hello', methods=['GET'])
def say_hello():
    return jsonify({'text': 'hello world'}), 200
    

if __name__ == '__main__':
    app.run(debug=True)