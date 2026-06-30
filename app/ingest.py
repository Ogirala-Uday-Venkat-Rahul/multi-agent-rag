"""Document loading and chunking, then writing chunks into the vector store."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import settings
from .schema import Chunk
from .vectorstore import get_store


def load_text(path: Path) -> str:
    """Read a .txt, .md, or .pdf file into plain text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Word-based sliding window. Overlap keeps a sentence from being split across
    a boundary so that retrieval doesn't lose the context around the seam."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}-{index}-{digest}"


def ingest_path(path: str | Path) -> int:
    """Ingest a single file or a directory of files. Returns chunk count added."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}
    )

    all_chunks: list[Chunk] = []
    for file in files:
        text = load_text(file)
        pieces = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    id=_chunk_id(file.name, i, piece),
                    text=piece,
                    metadata={"source": file.name, "chunk": i},
                )
            )

    get_store().add(all_chunks)
    return len(all_chunks)
