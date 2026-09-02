# Vincent - AI Chat Application

An AI chat interface powered by Ollama with user authentication and real-time messaging.

## Features

- Clean black & white minimalist design
- User authentication (signup/login)
- Real-time WebSocket communication
- Multi-language support (auto-detect)
- AI powered by Ollama (llama3.1)
- Mobile responsive

## Local Setup

### Prerequisites

- Python 3.8+
- Ollama installed and running locally

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure Ollama is running with llama3.1:
   ```bash
   ollama pull llama3.1
   ollama serve
   ```

3. Start the server:
   ```bash
   python server.py
   ```

4. Open http://localhost:8080 in your browser

### Environment Variables

- `OLLAMA_URL` - Ollama API URL (default: http://localhost:11434)
- `MODEL_NAME` - Model to use (default: llama3.1:latest)

## Vercel Deployment

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Deploy:
   ```bash
   vercel
   ```

3. Set environment variables in Vercel dashboard:
   - `OLLAMA_URL` - Your Ollama server URL
   - `MODEL_NAME` - Your model name

**Note:** Vercel serverless functions cannot run Ollama directly. You'll need to expose your Ollama server via a public URL (e.g., using ngrok) and set `OLLAMA_URL` accordingly.

## API Endpoints

- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Sign in
- `GET /api/history/{username}` - Get message history
- `GET /ws` - WebSocket connection
- `GET /health` - Health check

## File Structure

- `index.html` - Frontend UI
- `server.py` - Main server (local development)
- `vercel_app.py` - Vercel serverless entry point
- `vercel.json` - Vercel configuration
- `requirements.txt` - Python dependencies
