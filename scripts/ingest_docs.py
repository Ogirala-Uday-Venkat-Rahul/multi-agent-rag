"""CLI to index documents into the configured vector store.

    python -m scripts.ingest_docs data/sample
    python -m scripts.ingest_docs path/to/file.pdf
"""

from __future__ import annotations

import sys

from app.ingest import ingest_path
from app.vectorstore import get_store


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.ingest_docs <file-or-directory>")
        raise SystemExit(1)
    added = ingest_path(sys.argv[1])
    print(f"Ingested {added} chunks. Store now holds {get_store().count()} total.")


if __name__ == "__main__":
    main()
