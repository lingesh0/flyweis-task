# AI Text Intelligence API

Production-ready FastAPI service for sentiment analysis, keyword extraction, summarization, and semantic search powered by Hugging Face, spaCy, Sentence Transformers, and FAISS. **Includes real-time support via WebSocket and dynamic corpus updates.**

## Tech Stack
- Python 3.11
- FastAPI + Pydantic
- Transformers (`distilbert-base-uncased-finetuned-sst-2-english`, `t5-small`)
- spaCy (`en_core_web_sm`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (in-memory vector index)
- WebSocket for real-time analysis
- Docker

## Quick Start
### Local
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker
```bash
docker build -t ai-text-api .
docker run -p 8000:8000 ai-text-api
```

Swagger UI: http://localhost:8000/docs

## API Endpoints

### REST Endpoints

#### POST /analyze
Real-time sentiment analysis and keyword extraction.

**Request:**
```json
{ "text": "FastAPI makes building APIs delightful." }
```

**Response:**
```json
{
  "sentiment": "positive",
  "keywords": ["api", "fastapi", "building"]
}
```

---

#### POST /summarize
Summarize the provided text using T5 model.

**Request:**
```json
{ "text": "Long article text..." }
```

**Response:**
```json
{ "summary": "Concise summary text." }
```

---

#### POST /semantic-search
Semantic search over the in-memory corpus.

**Request:**
```json
{
  "query": "How do I deploy apps in containers?",
  "top_k": 3
}
```

**Response:**
```json
{
  "results": [
    "Docker simplifies packaging and deploying applications.",
    "..."
  ]
}
```

---

#### POST /corpus/add (NEW - Real-time Updates)
Dynamically add texts to the semantic search corpus.

**Request:**
```json
{
  "texts": [
    "New text to index",
    "Another text for semantic search"
  ]
}
```

**Response:**
```json
{
  "count": 2,
  "total_corpus_size": 7
}
```

---

#### GET /corpus/size (NEW - Corpus Info)
Get current corpus size.

**Response:**
```json
{ "corpus_size": 5 }
```

---

### WebSocket Endpoint

#### WS /ws/analyze (NEW - Real-time Streaming)
Real-time sentiment and keyword analysis via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analyze');

ws.onopen = () => {
  ws.send(JSON.stringify({ text: "Your text here" }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.type, msg.data); // "sentiment", "keywords", or "error"
};
```

**Server Response (streaming):**
```json
{
  "type": "sentiment",
  "data": { "sentiment": "positive" },
  "timestamp": "2026-01-07T12:34:56.789Z"
}
```

Then:

```json
{
  "type": "keywords",
  "data": { "keywords": ["api", "fastapi"] },
  "timestamp": "2026-01-07T12:34:56.850Z"
}
```

## Sample cURL Requests

### Sentiment Analysis
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Transformers make NLP easier."}'
```

### Summarization
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern, fast web framework for building APIs with Python."}'
```

### Semantic Search (with top_k)
```bash
curl -X POST http://localhost:8000/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is semantic search?", "top_k": 5}'
```

### Add Texts to Corpus (Real-time)
```bash
curl -X POST http://localhost:8000/corpus/add \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Machine learning models", "Deep learning frameworks"]}'
```

### Check Corpus Size
```bash
curl http://localhost:8000/corpus/size
```

---

## WebSocket Testing

### Using wscat
```bash
npm install -g wscat
wscat -c ws://localhost:8000/ws/analyze
# Then type: {"text": "Your message here"}
```

### Using Python
```python
import asyncio
import json
import websockets

async def test():
    async with websockets.connect('ws://localhost:8000/ws/analyze') as ws:
        await ws.send(json.dumps({"text": "FastAPI is amazing!"}))
        while True:
            msg = await ws.recv()
            print(json.loads(msg))

asyncio.run(test())
```

---

## Production Considerations

- Models are warmed on startup for reduced latency.
- FAISS index is in-memory and preloaded with default corpus.
- Adjust `DEFAULT_CORPUS` in `app/embeddings.py` for your data.
- For GPU support, use GPU base image and adjust Dockerfile.
- For persistent vector stores, integrate a database (PostgreSQL + pgvector, Weaviate, etc.).
- For scaling, deploy behind Gunicorn/Uvicorn with multiple workers.

---

## Swagger UI

Interactive API testing available at: **http://localhost:8000/docs**
