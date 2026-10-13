"""Step 2: split the label text into chunks, embed them, and store them in ChromaDB."""
import argparse
import json
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag
from config import COLLECTION, DEFAULT_DB_DIR, LABELS_PATH


def build(chunk_size=800, overlap=100, out_dir=DEFAULT_DB_DIR, labels_path=LABELS_PATH):
    docs = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs.append(Document(
                page_content=r["text"],
                metadata={"drug": r["drug"], "section": r["section"], "set_id": r.get("set_id", "")},
            ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    for c in chunks:  # make every chunk self-describing so the LLM can cite it
        c.page_content = f"[{c.metadata['drug']} | {c.metadata['section']}] {c.page_content}"

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)  # rebuild from scratch, avoid duplicate chunks

    Chroma.from_documents(
        chunks, rag.get_embeddings(), persist_directory=out_dir, collection_name=COLLECTION
    )
    print(f"Indexed {len(chunks)} chunks from {len(docs)} sections into '{out_dir}'")
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--out", default=DEFAULT_DB_DIR)
    args = ap.parse_args()
    build(args.chunk_size, args.overlap, args.out)
