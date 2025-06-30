# Natural Language to Oracle SQL App

This web application allows you to connect to your Oracle database and use AI (Google Gemini or Ollama) to generate and execute SQL queries from natural language.

## Features

- Connect to your own Oracle database
- Use Google Gemini (1.5 Pro, 2.5 Flash, 2.5 Flash-Lite Preview) or Ollama LLMs
- Generate Oracle SQL from natural language
- View and download query results
- No API keys or credentials are stored on the server

## Getting Started

### 1. Requirements

- Python 3.10+
- Oracle database (accessible from the server)
- (Optional) Ollama server for local LLMs

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
pip install -r requirements.txt
```

### 3. Running Locally

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### 4. Deploying on Render

1. Push your code to GitHub.
2. Create a new Web Service on [Render](https://render.com/).
3. Set the build command:
   ```
   pip install -r requirements.txt
   ```
4. Set the start command:
   ```
   python app.py
   ```
5. (Recommended) Add a `Procfile` with:
   ```
   web: python app.py
   ```
6. Set environment variables:
   - `FLASK_ENV=production`
   - `SESSION_TYPE=filesystem`
   - `SECRET_KEY=your-strong-secret`

### 5. Usage

- Open the app in your browser.
- Enter your Oracle DB credentials and connect.
- Enter your Gemini API key or Ollama details in the AI configuration.
- Select tables, enter your question, and generate SQL.
- Review and execute the SQL, and download results if needed.

### 6. Security

- **API keys are never stored on the server.** They are entered by users at runtime.
- Change the Flask `SECRET_KEY` for production.

### 7. License

MIT License

---

## 4. **Summary Table**

| File      | Purpose                                |
| --------- | -------------------------------------- |
| Procfile  | Tells Render how to start your app     |
| README.md | Explains how to use and deploy the app |

---

**Let me know if you want the README customized further or want to see a sample `.gitignore` or Dockerfile!**
