# SQL Chat - AI-Powered Natural Language to SQL 🚀

The modern, full-stack web companion for database querying. **SQL Chat** allows you to connect to your databases—Oracle, MySQL, or PostgreSQL—and interact with them using natural language. 

Built with a premium React interface and a robust Flask backend, SQL Chat uses state-of-the-art LLMs (Google Gemini) to translate your questions into perfect SQL queries instantly. Now optimized for **Vercel** deployment with a "Bring Your Own Key" (BYOK) model.

---

## ✨ Key Features

- **Web & Desktop Optimized**: Now a fully responsive web application, deployable to Vercel, Render, or as a local Electron app.
- **Multi-Database Support**: Connect to **Oracle**, **MySQL**, and **PostgreSQL** seamlessly.
- **Bring Your Own Key (BYOK)**: Users provide their own Gemini API keys, which are stored securely in their browser's `localStorage`.
- **Session Isolation**: Multi-user safe architecture ensures every user has their own private database connection and AI session.
- **Interactive Chat Interface**:
  - Remembers your table context within the conversation.
  - Generates SQL from your natural language (e.g., *"Show me all high-value orders from last month"*).
  - Execute SQL directly and view results in a premium data table.
- **Smart Context Management**: Automatically fetches table schemas and sample data to provide high-quality context to the AI for more accurate results.
- **Data Export**: Export your query results to CSV with a single click.

---

## 🛠️ Architecture & Tech Stack

- **Frontend**: React 19 (Vite), Tailwind CSS 4, Framer Motion (for smooth animations).
- **Backend**: Python 3 (Flask) with `oracledb`, `PyMySQL`, and `psycopg2-binary` connectors.
- **AI Integration**: Google Generative AI (Gemini).
- **Deployment**: Optimized for **Vercel** (Static Hosting + Serverless Functions).

---

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+**

### 2. Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/tharagaramanbalaji/chatwithdb.git
   cd chatwithdb
   ```
2. **Setup Backend**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Setup Frontend**:
   ```bash
   cd ../frontend
   npm install
   ```

### 3. Running Locally
1. **Start Backend**:
   ```bash
   cd backend
   python app.py
   ```
2. **Start Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

---

## ☁️ Deployment (Vercel)

This project is configured as a monorepo for Vercel.

1. **Push to GitHub**: Push your changes to your repository.
2. **Import to Vercel**: 
   - Import the root of the project.
   - **Build Command**: `npm run build` (configured in root `package.json`).
   - **Output Directory**: `frontend/dist`.
3. **Environment Variables**:
   - Add a `SECRET_KEY`: A long random string to secure user sessions.
   - *Note*: Gemini API keys are provided by users in the UI, so no server-side key is required.

---

## ⚙️ Configuration & Security

- **Privacy First**: Database credentials and AI API keys are **never stored on the server**. They are handled locally in your session or browser storage.
- **Local Context**: Table schemas are used purely for LLM prompting and are never persisted beyond the session.

---

## 📄 License

This project is licensed under the **MIT License**.