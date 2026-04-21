from pathlib import Path
from pypdf import PdfReader


def load_pdf_texts(pdf_dir: str) -> dict[str, str]:
    docs = {}
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        docs[pdf_path.stem] = "\n".join(page.extract_text() or "" for page in reader.pages)
    return docs
