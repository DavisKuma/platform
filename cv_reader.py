import os

from config import log


def read_cv(filepath: str) -> str:
    """Read a PDF or DOCX CV and return the full text content."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CV file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _read_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return _read_docx(filepath)
    else:
        raise ValueError(f"Unsupported CV format: {ext}. Use .pdf or .docx")


def _read_pdf(filepath: str) -> str:
    """Extract text from a PDF file."""
    from PyPDF2 import PdfReader

    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    full_text = "\n".join(pages)
    log.info("Read PDF CV: %d pages, %d characters", len(reader.pages), len(full_text))
    return full_text


def _read_docx(filepath: str) -> str:
    """Extract text from a DOCX file."""
    from docx import Document

    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    full_text = "\n".join(paragraphs)
    log.info("Read DOCX CV: %d paragraphs, %d characters", len(paragraphs), len(full_text))
    return full_text
