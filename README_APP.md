# Memgraph + Gemini RAG Web App

This is a web application for Knowledge Graph RAG using Memgraph and Google Gemini.

## Features
* **File Management**: Upload PDF/TXT/MD files.
* **ETL Pipeline**: Semantic chunking, embedding, and graph extraction.
* **Knowledge Chat**: RAG-based chat interface.
* **Graph Visualization**: Visual exploration of the Knowledge Graph.

## Architecture
* **Backend**: FastAPI (`src/backend`)
* **Frontend**: Streamlit (`src/frontend`)
* **Database**: Memgraph (Docker)

## Prerequisites
* Docker & Docker Compose
* Python 3.10+
* Google Gemini API Key

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory:
   ```ini
   GEMINI_API_KEY=your_gemini_key
   MEMGRAPH_URI=bolt://localhost:7687
   MEMGRAPH_USER=
   MEMGRAPH_PASSWORD=
   ```

3. **Start Database**
   ```bash
   docker-compose up -d
   ```

4. **Start Backend**
   ```bash
   uvicorn src.backend.api:app --host 0.0.0.0 --port 8000
   ```

5. **Start Frontend**
   In a new terminal:
   ```bash
   streamlit run src/frontend/app.py
   ```

## Usage
1. Go to `http://localhost:8501`.
2. Upload a document in the "File Management" tab.
3. Click "Process" to ingest the document into the Graph.
4. Go to "Chat" to ask questions about your documents.
5. Go to "Graph Visualization" to see the extracted entities and relations.
