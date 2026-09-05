# ResumeAI Studio (Resume Analyst)

Local web app that reads a resume (PDF, DOCX, or TXT), indexes it with RAG, and lets you chat about that file. It also extracts a profile, scores the resume against a job description, lists missing skills with suggestions, audits writing quality, and generates interview questions.

Repository: [https://github.com/rajdhanush09/Resume-_Analyst](https://github.com/rajdhanush09/Resume-_Analyst)

---

## Features

- **Chat with bot** — Ask questions about the active resume. Answers are grounded in retrieved chunks.
- **Structured profile** — Name, contact, skills, experience, and education extracted from the file.
- **ATS job matcher** — Paste a job description to see match %, skills already on the resume, and skills the JD requires that are missing.
- **Skill-gap suggestions** — For each missing skill: why the JD needs it, where to add it, and a sample bullet (do not list skills you cannot discuss in an interview).
- **Resume audit** — Overall score plus action verbs, metrics, and structure.
- **Interview coach** — Questions tailored to the resume.
- **Samples** — Built-in demo resumes (Alex Chen, Maya Patel).

Retrieval uses **TF-IDF / BM25** locally. If `sentence-transformers` installs successfully, it also uses the free Hugging Face model **`all-MiniLM-L6-v2`** for denser embeddings (downloaded once, then offline). Optional **Google Gemini** improves natural-language answers when you add an API key.

---

## Requirements

### Software

| Requirement | Notes |
| --- | --- |
| **Python 3.10+** (3.11–3.13 recommended) | Windows: `py -3`. Linux/macOS: `python3`. |
| **pip** | Comes with Python. |
| **Internet (first run)** | To install packages and download MiniLM (~80 MB). After that, the app can run offline except Gemini. |
| **Modern browser** | Chrome, Edge, or Firefox. |

### Python packages (`requirements.txt`)

| Package | Purpose |
| --- | --- |
| `fastapi`, `uvicorn` | Web API and local server |
| `python-multipart` | Resume file uploads |
| `pypdf` | PDF text extraction |
| `python-docx` | Word (.docx) extraction |
| `numpy` | Local vector math |
| `sentence-transformers` | Free MiniLM embeddings (Hugging Face) |
| `google-generativeai` | Optional Gemini chat / critiques |
| `pydantic`, `python-dotenv`, `certifi` | Config, validation, TLS |

PyTorch is pulled in by `sentence-transformers`. First install can take several minutes and needs a few GB of disk.

### Optional

- **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey) — keys usually start with `AIza`. Not required. Do not commit keys.

---

## Setup and run

From the project root (`Resume_Reader` or a clone of this repo):

```bash
py -3 -m pip install -r requirements.txt
py -3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

If `py` is not available:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** in your browser.

Stop the server with `Ctrl+C` in the terminal.

### First-time MiniLM download

The first chat or index after install may download `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face. If that fails (network or SSL), the app still runs using local TF-IDF retrieval.

### Optional Gemini

1. Create a key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. In the app header, open **Gemini Key** / **Resume bot**.
3. Paste the key and choose **Save & Validate Key**.
4. If validation fails, use **Use Offline Fallback**. Chat, ATS, and skill gaps still work.

---

## How to use

1. **Upload resume** (PDF / DOCX / TXT, up to 10 MB) or pick **Samples**.
2. Select the active resume in the header dropdown.
3. **Chat with bot** — e.g. skills, projects, education.
4. **ATS Job Matcher** — paste a JD (or a sample JD) → **Analyze match**. Review missing skills and suggested bullets.
5. Open **Structured Profile**, **Resume Audit**, or **Interview Coach** as needed.

---

## Tests

```bash
py -3 backend/test_rag.py
```

---

## Project layout

```
├── backend/
│   ├── main.py              # FastAPI app and routes
│   ├── pdf_parser.py        # PDF / DOCX / TXT parsing
│   ├── rag_engine.py        # Chunking, MiniLM + TF-IDF RAG, Gemini optional
│   ├── resume_analyzer.py   # Profile, ATS match, skill suggestions, audit
│   ├── sample_data.py       # Demo resumes
│   └── test_rag.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── requirements.txt
└── README.md
```

---

## License / academic use

Use this as a local demo or course project. Uploaded resumes stay in memory on your machine while the server is running; they are not sent to GitHub. Gemini is only used if you paste a key in the UI.
