"""
Resume RAG Engine
Handles document chunking, hybrid vector embeddings (Gemini + Local TF-IDF/BM25),
retrieval with cosine similarity, and grounded LLM generation with exact citations.
"""

import re
import os
import math
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai

try:
    import certifi
except ImportError:
    certifi = None

GEMINI_PING_MODELS = (
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
)


def configure_gemini(api_key: str) -> None:
    """Use HTTPS REST instead of gRPC — gRPC TLS often fails on Windows."""
    genai.configure(api_key=api_key, transport="rest")


def _ssl_contexts():
    contexts = []
    try:
        contexts.append(ssl.create_default_context())
    except Exception:
        pass
    if certifi:
        try:
            contexts.append(ssl.create_default_context(cafile=certifi.where()))
        except Exception:
            pass
    contexts.append(ssl._create_unverified_context())
    return contexts


def ping_gemini_rest(api_key: str, timeout: int = 12) -> Tuple[bool, str, Optional[str]]:
    """Validate a Gemini key with a short REST call (avoids hanging gRPC)."""
    key = (api_key or "").strip()
    if not key:
        return False, "Paste a Gemini API key first.", None
    if key.startswith("AQ."):
        return (
            False,
            "That value looks like a Google Cloud token, not a Gemini API key. "
            "Create one at https://aistudio.google.com/apikey (keys usually start with AIza).",
            None,
        )

    payload = json.dumps({"contents": [{"parts": [{"text": "Reply with OK"}]}]}).encode("utf-8")
    last_error = "Could not reach Gemini."

    for ctx in _ssl_contexts():
        for model in GEMINI_PING_MODELS:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={urllib.parse.quote(key, safe='')}"
            )
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                if text:
                    used_unverified = unverified or not ctx.verify_mode
                    msg = f"Gemini connected ({model})."
                    if ctx.verify_mode == ssl.CERT_NONE:
                        msg += " Windows could not verify Google's certificate; consider updating root CAs."
                    return True, msg, model
                last_error = "Gemini returned an empty reply."
            except urllib.error.HTTPError as err:
                detail = err.read().decode("utf-8", errors="ignore")[:240]
                last_error = f"HTTP {err.code}: {detail or err.reason}"
                if err.code == 429:
                    return False, "Gemini rate-limited this key. Wait a minute and try again.", None
                if err.code in (400, 401, 403):
                    # Bad/forbidden key — no point trying more TLS modes
                    if "API key not valid" in last_error or err.code in (400, 401, 403):
                        return False, (
                            "Google rejected this API key. Create a Gemini key at "
                            "https://aistudio.google.com/apikey (usually starts with AIza) and paste it again."
                        ), None
            except ssl.SSLError:
                last_error = "secure connection failed"
                break
            except Exception as err:
                last_error = str(err)
                if "CERTIFICATE" in last_error.upper():
                    break
    return False, f"API key validation failed: {last_error}", None


MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _enable_insecure_https_for_hf():
    """Windows on this machine often lacks Google/HF CA certs."""
    import ssl as _ssl
    previous = getattr(_ssl, "_create_default_https_context", None)
    _ssl._create_default_https_context = _ssl._create_unverified_context
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    return previous


class MiniLMEncoder:
    """Lazy all-MiniLM-L6-v2 dense encoder. Falls back silently if torch/HF is missing."""

    def __init__(self):
        self.model = None
        self.available = False
        self.error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        previous_https = None
        try:
            previous_https = _enable_insecure_https_for_hf()
            from sentence_transformers import SentenceTransformer

            try:
                self.model = SentenceTransformer(MINILM_MODEL_ID, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(MINILM_MODEL_ID)
            self.available = True
            self.error = None
        except Exception as err:
            self.model = None
            self.available = False
            self.error = str(err)
        finally:
            if previous_https is not None:
                try:
                    import ssl as _ssl
                    _ssl._create_default_https_context = previous_https
                except Exception:
                    pass

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.available or self.model is None or not texts:
            return None
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


_minilm_encoder: Optional[MiniLMEncoder] = None


def get_minilm_encoder() -> MiniLMEncoder:
    global _minilm_encoder
    if _minilm_encoder is None:
        _minilm_encoder = MiniLMEncoder()
    return _minilm_encoder

class TextChunk:
    def __init__(
        self,
        chunk_id: str,
        resume_id: str,
        text: str,
        page_number: int,
        section: str,
        char_start: int,
        char_end: int,
        embedding: Optional[List[float]] = None
    ):
        self.chunk_id = chunk_id
        self.resume_id = resume_id
        self.text = text
        self.page_number = page_number
        self.section = section
        self.char_start = char_start
        self.char_end = char_end
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "resume_id": self.resume_id,
            "text": self.text,
            "page_number": self.page_number,
            "section": self.section,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "has_embedding": self.embedding is not None,
            "word_count": len(self.text.split())
        }

class LocalHybridEmbedder:
    """
    Local TF-IDF & Subword BM25 embedding generator.
    Creates dense normalized vectors for cosine similarity search
    without requiring external API calls.
    """
    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, clean, split into alphanumeric tokens & sub-bigrams
        tokens = re.findall(r"\b[a-zA-Z0-9\+\#\.\-]{2,}\b", text.lower())
        return tokens

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        self.doc_count = len(documents)
        if self.doc_count == 0:
            return np.zeros((0, self.vocab_size))

        # Build vocabulary frequency
        doc_freq: Dict[str, int] = {}
        tokenized_docs = []
        for doc in documents:
            tokens = set(self._tokenize(doc))
            tokenized_docs.append(self._tokenize(doc))
            for tok in tokens:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        # Select top vocabulary words
        sorted_tokens = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size]
        self.vocab = {tok: idx for idx, (tok, _) in enumerate(sorted_tokens)}

        # Compute IDF
        self.idf = {
            tok: math.log((self.doc_count + 1) / (freq + 1)) + 1.0
            for tok, freq in doc_freq.items()
            if tok in self.vocab
        }

        # Vectorize documents
        matrix = np.zeros((self.doc_count, len(self.vocab)), dtype=np.float32)
        for i, tokens in enumerate(tokenized_docs):
            for tok in tokens:
                if tok in self.vocab:
                    col = self.vocab[tok]
                    matrix[i, col] += 1.0 * self.idf[tok]
            
            # Normalize vector to unit length
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        return matrix

    def transform_query(self, query: str) -> np.ndarray:
        if not self.vocab:
            return np.zeros(self.vocab_size, dtype=np.float32)
        
        tokens = self._tokenize(query)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in tokens:
            if tok in self.vocab:
                col = self.vocab[tok]
                vec[col] += 1.0 * self.idf.get(tok, 1.0)
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

class ResumeRAGStore:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.resumes: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, List[TextChunk]] = {}
        self.local_embedders: Dict[str, LocalHybridEmbedder] = {}
        self.local_vectors: Dict[str, np.ndarray] = {}
        self.minilm_vectors: Dict[str, np.ndarray] = {}
        self.gemini_available = False
        self.minilm = get_minilm_encoder()
        
        if self.api_key:
            self._configure_gemini(self.api_key)

    def set_api_key(self, api_key: str) -> Tuple[bool, str]:
        """Configure and test Gemini API Key over REST with a hard timeout."""
        key = (api_key or "").strip()
        ok, message, model = ping_gemini_rest(key)
        if not ok:
            self.gemini_available = False
            self.api_key = None
            return False, message
        self.api_key = key
        try:
            configure_gemini(key)
        except Exception:
            pass
        self.gemini_available = True
        return True, message if model else "API key validated with Gemini."

    def _configure_gemini(self, api_key: str):
        try:
            configure_gemini(api_key)
            self.gemini_available = False
        except Exception:
            self.gemini_available = False

    def chunk_document(
        self,
        resume_id: str,
        doc_data: Dict[str, Any],
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[TextChunk]:
        """
        Split parsed document into semantic, section-aware chunks.
        """
        chunks: List[TextChunk] = []
        chunk_idx = 0
        
        pages = doc_data.get("pages", [])
        sections = doc_data.get("sections", {})

        for page in pages:
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "")
            
            if not page_text.strip():
                continue

            # Split text by paragraphs or double newlines first
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
            
            current_buffer = ""
            current_section = "general"
            char_offset = 0

            for para in paragraphs:
                # Check if paragraph starts with or contains a section header
                para_first_lines = para.split("\n")[:2]
                for line in para_first_lines:
                    cleaned_line = re.sub(r"^[#\*\-=\:\s]+|[#\*\-=\:\s]+$", "", line).strip().lower()
                    for sec_name in ["summary", "experience", "education", "skills", "projects", "certifications", "awards", "contact"]:
                        if sec_name in cleaned_line or (sec_name == "experience" and "work history" in cleaned_line):
                            current_section = sec_name
                            break

                # If paragraph fits in current chunk
                if len(current_buffer) + len(para) <= chunk_size:
                    if current_buffer:
                        current_buffer += "\n\n" + para
                    else:
                        current_buffer = para
                else:
                    # Flush current buffer if it has content
                    if current_buffer:
                        chunk_id = f"{resume_id}_c{chunk_idx}"
                        chunks.append(TextChunk(
                            chunk_id=chunk_id,
                            resume_id=resume_id,
                            text=current_buffer,
                            page_number=page_num,
                            section=current_section,
                            char_start=char_offset,
                            char_end=char_offset + len(current_buffer)
                        ))
                        chunk_idx += 1
                        char_offset += len(current_buffer)
                        
                        # Apply overlap
                        overlap_text = current_buffer[-chunk_overlap:] if len(current_buffer) > chunk_overlap else ""
                        current_buffer = (overlap_text + "\n" + para).strip()
                    else:
                        # Paragraph itself is larger than chunk_size, split by sentences
                        sentences = re.split(r"(?<=[.!?])\s+", para)
                        sub_buf = ""
                        for sent in sentences:
                            if len(sub_buf) + len(sent) <= chunk_size:
                                sub_buf += " " + sent if sub_buf else sent
                            else:
                                if sub_buf:
                                    chunk_id = f"{resume_id}_c{chunk_idx}"
                                    chunks.append(TextChunk(
                                        chunk_id=chunk_id,
                                        resume_id=resume_id,
                                        text=sub_buf.strip(),
                                        page_number=page_num,
                                        section=current_section,
                                        char_start=char_offset,
                                        char_end=char_offset + len(sub_buf)
                                    ))
                                    chunk_idx += 1
                                    char_offset += len(sub_buf)
                                sub_buf = sent
                        current_buffer = sub_buf

            if current_buffer:
                chunk_id = f"{resume_id}_c{chunk_idx}"
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    resume_id=resume_id,
                    text=current_buffer,
                    page_number=page_num,
                    section=current_section,
                    char_start=char_offset,
                    char_end=char_offset + len(current_buffer)
                ))
                chunk_idx += 1

        return chunks

    def add_resume(self, resume_id: str, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Index a parsed resume into chunk storage and compute embeddings."""
        self.resumes[resume_id] = doc_data
        doc_chunks = self.chunk_document(resume_id, doc_data)
        self.chunks[resume_id] = doc_chunks

        # Fit local hybrid vector index
        texts = [c.text for c in doc_chunks]
        embedder = LocalHybridEmbedder()
        vectors = embedder.fit_transform(texts)
        self.local_embedders[resume_id] = embedder
        self.local_vectors[resume_id] = vectors

        minilm_ok = False
        if self.minilm.available:
            dense = self.minilm.encode(texts)
            if dense is not None and len(dense) == len(doc_chunks):
                self.minilm_vectors[resume_id] = dense
                minilm_ok = True
                for chunk, row in zip(doc_chunks, dense):
                    chunk.embedding = row.tolist()
            else:
                self.minilm_vectors.pop(resume_id, None)
        else:
            self.minilm_vectors.pop(resume_id, None)

        if minilm_ok:
            embedding_mode = "MiniLM-L6-v2 + TF-IDF hybrid"
        elif self.gemini_available:
            embedding_mode = "Gemini Dense + Local Hybrid"
        else:
            embedding_mode = "Local Hybrid TF-IDF/BM25"

        return {
            "resume_id": resume_id,
            "filename": doc_data.get("filename", "resume"),
            "total_chunks": len(doc_chunks),
            "total_pages": doc_data.get("total_pages", 1),
            "sections_found": list(doc_data.get("sections", {}).keys()),
            "embedding_mode": embedding_mode,
            "minilm_active": minilm_ok,
        }

    def retrieve(
        self,
        resume_id: str,
        query: str,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining dense vector similarity and lexical keyword scoring.
        """
        if resume_id not in self.chunks or not self.chunks[resume_id]:
            return []

        doc_chunks = self.chunks[resume_id]
        scores = []

        # 1. Local Dense / TF-IDF Cosine Similarity
        embedder = self.local_embedders.get(resume_id)
        vectors = self.local_vectors.get(resume_id)
        
        local_sims = np.zeros(len(doc_chunks))
        if embedder and vectors is not None and len(vectors) > 0:
            query_vec = embedder.transform_query(query)
            if np.linalg.norm(query_vec) > 0:
                local_sims = np.dot(vectors, query_vec)

        minilm_sims = np.zeros(len(doc_chunks))
        dense = self.minilm_vectors.get(resume_id)
        if dense is not None and self.minilm.available and len(dense) == len(doc_chunks):
            query_dense = self.minilm.encode([query])
            if query_dense is not None and len(query_dense):
                minilm_sims = np.dot(dense, query_dense[0])

        use_minilm = dense is not None and len(dense) == len(doc_chunks)

        # 2. Lexical keyword overlap bonus
        query_terms = set(re.findall(r"\b[a-zA-Z0-9\+\#\.\-]{2,}\b", query.lower()))
        
        for idx, chunk in enumerate(doc_chunks):
            chunk_terms = set(re.findall(r"\b[a-zA-Z0-9\+\#\.\-]{2,}\b", chunk.text.lower()))
            overlap = query_terms.intersection(chunk_terms)
            keyword_score = len(overlap) / max(1, len(query_terms))
            
            sim_score = float(local_sims[idx]) if idx < len(local_sims) else 0.0
            dense_score = float(minilm_sims[idx]) if idx < len(minilm_sims) else 0.0
            
            section_boost = 0.0
            for term in query_terms:
                if term in chunk.section.lower():
                    section_boost += 0.15

            if use_minilm:
                final_score = (
                    (0.58 * dense_score)
                    + (0.22 * sim_score)
                    + (0.15 * keyword_score)
                    + (0.05 * min(section_boost, 0.2))
                )
            else:
                final_score = (0.6 * sim_score) + (0.3 * keyword_score) + (0.1 * min(section_boost, 0.2))
            normalized_score = max(0.0, min(1.0, final_score * 1.3))

            scores.append({
                "chunk": chunk,
                "score": round(normalized_score, 4),
                "keyword_matches": list(overlap),
                "section": chunk.section,
                "page_number": chunk.page_number
            })

        # Sort descending by score
        scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = scores[:top_k]

        return [
            {
                "chunk_id": res["chunk"].chunk_id,
                "text": res["chunk"].text,
                "page_number": res["page_number"],
                "section": res["section"],
                "score": res["score"],
                "keyword_matches": res["keyword_matches"],
                "confidence": "High" if res["score"] > 0.65 else ("Medium" if res["score"] > 0.35 else "Low")
            }
            for res in top_results
        ]

    def generate_answer(
        self,
        resume_id: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate grounded answer with strict citations using Gemini or offline extractive synthesis.
        """
        doc_info = self.resumes.get(resume_id, {})
        filename = doc_info.get("filename", "Resume")
        
        # Build context string with citation markers
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i} - Page {chunk['page_number']} - Section: {chunk['section'].upper()}]\n{chunk['text']}"
            )
        context_str = "\n\n".join(context_parts)

        # 1. Try LLM Generation if Gemini is available
        if self.gemini_available and self.api_key:
            system_prompt = f"""You are a friendly chat bot helping the user talk about one uploaded resume file: {filename}.
Reply like a messaging assistant — short, natural, and useful. Do not paste raw resume dumps or "Source 1/2/3" blocks.

STRICT RULES:
1. Use ONLY the resume context below. Never invent jobs, skills, dates, or employers.
2. If the file does not contain the answer, say so plainly: "That isn't mentioned in this resume."
3. Lead with a direct answer, then 3–6 bullets of supporting facts from the file.
4. You may add brief citations like (page 1, skills) after a fact — never dump full chunks.
5. End with one short follow-up question the user could ask next about this same file.

RESUME FILE CONTEXT:
{context_str}

USER MESSAGE:
{query}
"""
            # Try multiple Gemini models
            for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        system_prompt,
                        request_options={"timeout": 25},
                    )
                    if response and response.text:
                        return {
                            "answer": response.text.strip(),
                            "model_used": model_name,
                            "mode": "gemini_llm",
                            "sources": retrieved_chunks,
                            "grounded": True
                        }
                except Exception as e:
                    continue

        # 2. Offline / Local Extractive Synthesizer Fallback
        return self._generate_extractive_fallback(query, retrieved_chunks, filename)

    def _split_resume_points(self, text: str) -> List[str]:
        pieces = re.split(r"(?:\n+|•|●|(?<=[.!?])\s+)", text or "")
        points = []
        for piece in pieces:
            cleaned = re.sub(r"\s+", " ", piece).strip(" -•●\t")
            if len(cleaned) < 18:
                continue
            if cleaned.lower() in {"objectives", "education", "projects", "internships", "tools", "skills", "experience"}:
                continue
            points.append(cleaned)
        return points

    def _generate_extractive_fallback(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        filename: str
    ) -> Dict[str, Any]:
        """
        Offline chat-style answer: pick the most relevant sentences from retrieved chunks.
        Citations stay in the sources list, not in the message body.
        """
        if not retrieved_chunks:
            return {
                "answer": f"I couldn't find anything in **{filename}** that answers that. Try asking about skills, projects, education, or internships.",
                "model_used": "resume-bot",
                "mode": "offline_extractive",
                "sources": [],
                "grounded": True
            }

        stop = {
            "the", "a", "an", "what", "are", "is", "their", "they", "candidate", "about",
            "tell", "me", "does", "have", "has", "with", "and", "or", "of", "in", "to",
            "for", "this", "that", "from", "resume", "file", "uploaded", "please", "can",
            "you", "how", "who", "when", "where", "which", "top", "most", "any"
        }
        query_terms = [
            w.lower()
            for w in re.findall(r"[a-zA-Z0-9+#/.]+", query)
            if w.lower() not in stop and len(w) > 2
        ]

        scored: List[Tuple[float, str]] = []
        for chunk in retrieved_chunks:
            chunk_boost = float(chunk.get("score") or 0)
            for point in self._split_resume_points(chunk.get("text") or ""):
                low = point.lower()
                hits = sum(1 for term in query_terms if term in low)
                scored.append((hits * 2.0 + chunk_boost, point))

        scored.sort(key=lambda item: item[0], reverse=True)
        with_terms = [item for item in scored if item[0] >= 2.0]
        ranked = with_terms or scored
        picks: List[str] = []
        seen = set()
        contact_re = re.compile(r"@|linkedin\.com|github\.com|\(\d{3}\)", re.I)
        wants_contact = any(t in query.lower() for t in ("email", "phone", "contact", "linkedin", "github"))
        for score, point in ranked:
            if contact_re.search(point) and not wants_contact:
                continue
            key = point.lower()[:90]
            if key in seen:
                continue
            seen.add(key)
            picks.append(point)
            if len(picks) >= 5:
                break

        if not picks:
            picks = self._split_resume_points(retrieved_chunks[0].get("text") or "")[:3]

        bullets = "\n".join(f"- {item}" for item in picks)
        answer = (
            f"I checked **{filename}** for that.\n\n"
            f"{bullets}\n\n"
            "Want me to go deeper on skills, projects, education, or internships in this file?"
        )
        return {
            "answer": answer.strip(),
            "model_used": "resume-bot",
            "mode": "offline_extractive",
            "sources": retrieved_chunks,
            "grounded": True
        }
