from flask import Flask
from flask_session import Session 
from dotenv import load_dotenv
import sys
import os

from routes import api_bp

load_dotenv()

# Configure paths for PyInstaller bundling
if getattr(sys, 'frozen', False):
    # If running as a bundled executable
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    # If running from source
    app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'  
Session(app)

# Register all API endpoints and UI routes
app.register_blueprint(api_bp)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)