"""
Resume Reader - Document Parser & Section Extractor
Supports PDF (via pypdf), DOCX (via python-docx), and plain text.
Extracts raw text, pages, and identifies resume sections.
"""

import io
import re
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
import docx

SECTION_PATTERNS = {
    "summary": [r"(?:professional\s+)?summary", r"objective", r"profile", r"about\s+me", r"overview"],
    "experience": [r"(?:work\s+)?experience", r"employment\s+history", r"work\s+history", r"professional\s+experience", r"career\s+history"],
    "education": [r"education", r"academic\s+background", r"qualifications", r"degrees", r"academic\s+history"],
    "skills": [r"(?:technical\s+)?skills", r"core\s+competencies", r"technologies", r"tools\s+&\s+technologies", r"expertise", r"skillset"],
    "projects": [r"projects", r"personal\s+projects", r"key\s+projects", r"portfolio", r"technical\s+projects"],
    "certifications": [r"certifications", r"licenses", r"courses\s+&\s+certifications", r"credentials", r"accreditations"],
    "awards": [r"awards", r"honors", r"achievements", r"publications", r"recognitions"],
    "contact": [r"contact(?:\s+information)?", r"personal\s+details"]
}

def clean_text(text: str) -> str:
    """Normalize whitespace, remove excess blank lines and unusual unicode chars."""
    if not text:
        return ""
    # Replace non-breaking spaces and special bullet points with standard characters
    text = text.replace("\u00a0", " ").replace("\u2022", "•").replace("\u2013", "-").replace("\u2014", "-")
    # Replace multiple empty lines with double newline
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()

def detect_section(line: str) -> Optional[str]:
    """Check if a single line matches a known resume section header."""
    normalized = line.strip().lower()
    # Remove leading/trailing punctuation and markdown headers like ### or ---
    normalized = re.sub(r"^[#\*\-=\:\s]+|[#\*\-=\:\s]+$", "", normalized)
    
    if len(normalized) > 50 or len(normalized) < 3:
        return None
        
    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.fullmatch(pattern, normalized, re.IGNORECASE):
                return section_name
    return None

def parse_pdf(file_bytes: bytes, filename: str = "resume.pdf") -> Dict[str, Any]:
    """
    Extract text, page by page, and identify structural sections from a PDF file.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_data: List[Dict[str, Any]] = []
    full_text_list: List[str] = []
    
    current_section = "general"
    sections_map: Dict[str, List[str]] = {}

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = page.extract_text() or ""
        page_text = clean_text(page_text)
        
        full_text_list.append(page_text)
        
        # Analyze lines for sections on this page
        page_lines = page_text.split("\n")
        annotated_lines = []
        for line in page_lines:
            detected = detect_section(line)
            if detected:
                current_section = detected
            annotated_lines.append({
                "text": line,
                "section": current_section
            })
            if current_section not in sections_map:
                sections_map[current_section] = []
            sections_map[current_section].append(line)

        pages_data.append({
            "page_number": page_num,
            "text": page_text,
            "char_count": len(page_text),
            "word_count": len(page_text.split())
        })

    full_text = "\n\n".join(full_text_list)
    
    # Clean sections map text
    parsed_sections = {
        sec: "\n".join(lines).strip()
        for sec, lines in sections_map.items()
        if lines
    }

    return {
        "filename": filename,
        "format": "pdf",
        "total_pages": len(pages_data),
        "total_characters": len(full_text),
        "total_words": len(full_text.split()),
        "full_text": full_text,
        "pages": pages_data,
        "sections": parsed_sections
    }

def parse_docx(file_bytes: bytes, filename: str = "resume.docx") -> Dict[str, Any]:
    """Extract text and sections from a DOCX file."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = clean_text("\n".join(paragraphs))
    
    current_section = "general"
    sections_map: Dict[str, List[str]] = {}
    
    for line in full_text.split("\n"):
        detected = detect_section(line)
        if detected:
            current_section = detected
        if current_section not in sections_map:
            sections_map[current_section] = []
        sections_map[current_section].append(line)

    parsed_sections = {
        sec: "\n".join(lines).strip()
        for sec, lines in sections_map.items()
        if lines
    }

    return {
        "filename": filename,
        "format": "docx",
        "total_pages": max(1, len(full_text) // 2500 + 1),
        "total_characters": len(full_text),
        "total_words": len(full_text.split()),
        "full_text": full_text,
        "pages": [{"page_number": 1, "text": full_text, "char_count": len(full_text), "word_count": len(full_text.split())}],
        "sections": parsed_sections
    }

def parse_text(file_bytes: bytes, filename: str = "resume.txt") -> Dict[str, Any]:
    """Extract text from plain text or markdown file."""
    text = file_bytes.decode("utf-8", errors="ignore")
    full_text = clean_text(text)
    
    current_section = "general"
    sections_map: Dict[str, List[str]] = {}
    
    for line in full_text.split("\n"):
        detected = detect_section(line)
        if detected:
            current_section = detected
        if current_section not in sections_map:
            sections_map[current_section] = []
        sections_map[current_section].append(line)

    parsed_sections = {
        sec: "\n".join(lines).strip()
        for sec, lines in sections_map.items()
        if lines
    }

    return {
        "filename": filename,
        "format": "text",
        "total_pages": max(1, len(full_text) // 2500 + 1),
        "total_characters": len(full_text),
        "total_words": len(full_text.split()),
        "full_text": full_text,
        "pages": [{"page_number": 1, "text": full_text, "char_count": len(full_text), "word_count": len(full_text.split())}],
        "sections": parsed_sections
    }

def parse_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Universal dispatcher based on file extension."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return parse_pdf(file_bytes, filename)
    elif lower_name.endswith(".docx"):
        return parse_docx(file_bytes, filename)
    elif lower_name.endswith(".txt") or lower_name.endswith(".md"):
        return parse_text(file_bytes, filename)
    else:
        # Attempt PDF first, fallback to text
        try:
            return parse_pdf(file_bytes, filename)
        except Exception:
            return parse_text(file_bytes, filename)
