from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import re
import os
from anthropic import Anthropic
import tempfile  # Import tempfile for temporary file handling

app = Flask(__name__)
CORS(app)

# Sichere Handhabung des API-Schlüssels
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

anthropic_client = Anthropic()

def extract_text_from_pdf(pdf_path):
    """Extrahiert Text aus einer PDF-Datei."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or "" #Handle cases where extract_text returns None
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

def search_with_regex(text, pattern):
    """Sucht nach einem Regex-Muster im Text."""
    try:
        matches = re.finditer(pattern, text, re.MULTILINE)
        results = []
        for match in matches:
            results.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        return results
    except re.error as e:
        return f'Ungültiges Regex-Muster: {str(e)}'

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    """Verarbeitet eine PDF-Datei und gibt den extrahierten Text zurück."""
    if 'file' not in request.files:
        return jsonify({'error': 'Keine Datei im Request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Kein Dateiname angegeben'}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            # Use tempfile to create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                file.save(temp_pdf)
                temp_pdf_path = temp_pdf.name  # Get the path to the temporary file

            # Extract text from PDF
            text = extract_text_from_pdf(temp_pdf_path)

            # Clean up temporary file
            os.unlink(temp_pdf_path)

            return jsonify({'text': text})
        except Exception as e:
            return jsonify({'error': f"Error processing PDF: {e}"}), 500

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
    except Exception as e:
        return jsonify({'error': f'Error during search: {str(e)}'}), 500

@app.route('/ask_anthropic', methods=['POST'])
def ask_anthropic():
    """Sendet eine Anfrage an die Anthropic API."""
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt erforderlich'}), 400

    prompt = data['prompt']

    try:
        message = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return jsonify({'response': message.content[0].text})
    except Exception as e:
        return jsonify({'error': f'Fehler bei der Anfrage an Anthropic: {str(e)}'}), 500

@app.route('/say_hello', methods=['GET'])
def say_hello():
    return jsonify({'text': 'hello world'}), 200

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Test erfolgreich'})

