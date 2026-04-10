import sys
import os

# Add the backend directory to the path so we can import the app
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app

# Vercel needs the app object to be named 'app'
# Since we imported 'app' from backend.app, we are good.
