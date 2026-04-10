from flask import Flask, jsonify
from flask_session import Session 
from flask_cors import CORS
from dotenv import load_dotenv
import sys
import os

from routes import api_bp

load_dotenv()

app = Flask(__name__)
# Enable CORS for frontend development
CORS(app)

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'  
Session(app)

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "service": "chatwithdb-api"})

# Register all API endpoints and UI routes
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)