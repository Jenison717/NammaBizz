# NAmma BiZZ — AI Wholesale Business Platform

A Python + FastAPI version of the NAmma BiZZ prototype. It keeps the existing single-page UI and adds a real server-side LLM matching endpoint.

## Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python + FastAPI
- AI: Groq chat completions API
- Data: JSON demo catalog (40 Madurai wholesale businesses, 200 products)
- Editor: VS Code

## Project structure
```text
NAmma_BiZZ_Project/
├── backend/main.py
├── data/businesses.json
├── frontend/index.html
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py
└── README.md
```

## Run in VS Code
1. Open this folder in VS Code.
2. Create a virtual environment:
   `python -m venv .venv`
3. Activate it.
   - Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
4. Install packages:
   `pip install -r requirements.txt`
5. Copy `.env.example` to `.env`.
6. Create a Groq API key at `https://console.groq.com/keys`.
7. Put it in `.env` as `GROQ_API_KEY=your_key_here`.
8. Optionally set `GROQ_MODEL` to another model available to your Groq account. The default is `openai/gpt-oss-20b`.
9. Start:
   `python run.py`
10. Open `http://127.0.0.1:8000`

## AI flow
Customer requirement -> Groq LLM -> structured requirement + business IDs -> validated catalog lookup -> frontend results.

The model is instructed never to invent businesses; recommendations are restricted to the supplied catalog.

## API
- `GET /api/health`
- `POST /api/ai-match` with JSON `{ "requirement": "I need 500 cotton school uniforms" }`
- `POST /api/ask` for catalog questions
- `POST /api/estimate` for catalog price estimates
- `POST /api/feedback` for server-side customer feedback storage

If Groq is unavailable, `/api/ai-match` falls back to deterministic local catalog matching so the search remains usable. GPS coordinates are rounded before being sent to Groq. Product cards use domain/product-specific generated illustrations, while business cards use category-specific photography.

Never commit `.env` or expose your API key in browser JavaScript. Feedback is stored locally in `data/feedback.json` for this prototype and is excluded from Git.
