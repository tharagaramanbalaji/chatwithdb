from flask import Flask, jsonify, render_template, send_from_directory
from flask_session import Session 
from flask_cors import CORS
from dotenv import load_dotenv
import sys
import os

from routes import api_bp

load_dotenv()

# Define absolute paths for React assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.normpath(os.path.join(BASE_DIR, '..', 'frontend', 'dist'))

app = Flask(__name__, 
            static_folder=FRONTEND_DIST, 
            template_folder=FRONTEND_DIST,
            static_url_path='')
# Enable CORS for frontend development
CORS(app)

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'
Session(app)

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "service": "chatwithdb-api"})

# Register all API endpoints
app.register_blueprint(api_bp, url_prefix='/api')

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)