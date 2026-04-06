# SQL Chat - AI-Powered Natural Language to SQL 🚀

The modern desktop companion for database querying. **SQL Chat** allows you to connect to your databases—Oracle, MySQL, or PostgreSQL—and interact with them using natural language. Built with a stunning dark-mode desktop interface, SQL Chat uses state-of-the-art LLMs (Google Gemini or Ollama) to translate your questions into perfect SQL queries instantly.

---

## ✨ Key Features

- **Multi-Database Support**: Connect to **Oracle**, **MySQL**, and **PostgreSQL** seamlessly.
- **Dual AI Engine**:
  - **Google Gemini**: Leverage the latest Gemini 2.5 models (Pro, Flash, Flash-Lite) via direct API.
  - **Ollama**: Connect to local or remote Ollama instances for private, server-side-free querying.
- **Interactive Chat Interface**:
  - Remembers your table context within the conversation.
  - Generates SQL from your natural language (e.g., *"Show me all high-value orders from last month"*).
  - Execute SQL directly and view results in a professional-grade table.
- **Smart Context Management**: Automatically fetches table schemas and sample data to provide high-quality context to the AI for more accurate results.
- **Data Export**: Export your query results to CSV/JSON with ease.
- **Desktop Power**: Built as a native Windows Electron desktop application for a smooth and isolated workspace.

---

## 🛠️ Architecture & Tech Stack

- **Frontend**: Electron, HTML5, Vanilla CSS (Modern Dark UI), Bootstrap 5.
- **Backend**: Python (Flask) with `oracledb`, `pymysql`, and `psycopg2` connectors.
- **AI Integration**: `google-generativeai` and Ollama REST APIs.
- **Packaging**: Bundled with **PyInstaller** for the backend and **electron-builder** for the desktop app.

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+** (for the backend)
- **Node.js 18+** (for Electron and building)
- **Database Client Drivers**: Ensure your database is accessible and the relevant client libraries (like `oracledb`) are installed if necessary.

### 2. Installation & Development

1. **Clone the repository**:
   ```bash
   git clone [YOUR_REPO_URL]
   cd chatwithdb
   ```
2. **Setup Python Virtual Environment**:
   ```bash
   # Create a virtual environment named 'vebnv' (expected by main.js)
   python -m venv vebnv
   source vebnv/bin/activate  # Windows: vebnv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Install Node Dependencies**:
   ```bash
   npm install
   ```
4. **Run in Development Mode**:
   ```bash
   npm start
   ```

---

## 🏗️ Building for Production

To create a standalone `.exe` for Windows:

1. **Build the Backend**:
   Use PyInstaller to bundle the Flask server into an executable.
   ```bash
   pyinstaller --workpath ./build --distpath ./dist/backend app.spec
   ```
2. **Build the Electron App**:
   Use `electron-builder` to package the app. It will automatically include the backend from `dist/backend/app.exe`.
   ```bash
   npm run dist
   ```

---

## ⚙️ Configuration & Security

### AI Configuration
- **Google Gemini**: Enter your API key in the configuration panel. Keys are processed locally and only sent to Google's API during generation.
- **Ollama**: Connect by providing your Ollama base URL (e.g., `http://localhost:11434`) and selecting your model (Llama-3, CodeLlama, etc.).

### Security
- **Privacy First**: Database credentials and AI API keys are **never stored on any server**. They are handled locally in your session.
- **Local Context**: Table schemas are used purely for LLM prompting and never persisted.

---

## 🖥️ User Interface Preview

The application features a modern, high-contrast dark theme designed for developer focus. It includes a collapsible sidebar for table management, a persistent session status bar for both AI and DB, and a fluid chat experience with built-in SQL execution modules.

---

## 📄 License

This project is licensed under the **MIT License**.

---