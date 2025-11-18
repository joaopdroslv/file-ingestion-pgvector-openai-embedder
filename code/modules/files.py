from typing import Optional

import docx
import pdfplumber


def read_pdf_file(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            texts.append(p.extract_text() or "")
    return "\n".join(texts)


def read_docx_file(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def load_file(path: str) -> Optional[str]:
    if path.lower().endswith(".pdf"):
        return read_pdf_file(path)
    elif path.lower().endswith(".docx") or path.lower().endswith(".doc"):
        return read_docx_file(path)
    else:
        raise RuntimeError("File format not supported.")
