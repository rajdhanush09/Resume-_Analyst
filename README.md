# ResumeAI Studio

Local resume reader with RAG chat, structured profile extraction, ATS job matching, skill-gap suggestions, resume audit, and interview questions.

## Run locally

```bash
py -3 -m pip install -r requirements.txt
py -3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

Upload a PDF/DOCX/TXT resume or load a sample. Optional: add a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) for fuller LLM answers. Offline hybrid RAG still works without a key.

Do not commit API keys.
