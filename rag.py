"""Core RAG logic: retrieve label chunks from ChromaDB and answer with citations."""
from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import COLLECTION, DEFAULT_DB_DIR, EMBED_MODEL, GROQ_MODEL, LABELS_PATH, REFUSAL
from drugs import detect_drugs

load_dotenv()  # reads GROQ_API_KEY from a local .env file

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about medicines using ONLY the label excerpts provided. "
     "Each excerpt starts with [drug | section]. Cite every claim as [drug | section]. "
     f"If the excerpts do not contain the answer, reply exactly: {REFUSAL} "
     "Keep answers short and factual. You are not giving medical advice."),
    ("human", "Excerpts:\n{context}\n\nQuestion: {question}"),
])


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


@lru_cache(maxsize=4)
def get_db(persist_dir=DEFAULT_DB_DIR):
    return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings(),
                  collection_name=COLLECTION)


@lru_cache(maxsize=1)
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=0)  # reads GROQ_API_KEY from the environment


@lru_cache(maxsize=1)
def get_bm25_index(labels_path=LABELS_PATH):
    """Build or load an in-memory BM25 index over the chunked corpus for keyword search."""
    import re
    from rank_bm25 import BM25Okapi
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            import json
            r = json.loads(line)
            docs.append(Document(
                page_content=r["text"],
                metadata={"drug": r["drug"], "section": r["section"], "set_id": r.get("set_id", "")},
            ))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.page_content = f"[{c.metadata['drug']} | {c.metadata['section']}] {c.page_content}"

    tokenized_corpus = [re.findall(r"\w+", c.page_content.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


@lru_cache(maxsize=1)
def get_reranker():
    """Load cross-encoder reranker model (CPU-friendly). Returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    except Exception:
        return None


def rrf_fuse(rankings_list, k_rrf=60):
    """Reciprocal Rank Fusion over multiple ranked Document lists."""
    scores = {}
    doc_map = {}
    for ranking in rankings_list:
        for rank, doc in enumerate(ranking):
            key = (doc.metadata.get("drug", ""), doc.metadata.get("section", ""), doc.page_content[:100])
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + (1.0 / (k_rrf + rank + 1))

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def bm25_search(question, k=10, drug=None, labels_path=LABELS_PATH):
    import re
    bm25, chunks = get_bm25_index(labels_path)
    tokens = re.findall(r"\w+", question.lower())
    if not tokens:
        return chunks[:k]
    scores = bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_indices:
        c = chunks[idx]
        if drug and c.metadata.get("drug") != drug:
            continue
        results.append(c)
        if len(results) >= k:
            break
    return results


def _search(db, question, k, drug=None):
    kwargs = {"k": k}
    if drug:
        kwargs["filter"] = {"drug": drug}
    return db.as_retriever(search_kwargs=kwargs).invoke(question)


def retrieve_dense(question, drug=None, k=4, persist_dir=DEFAULT_DB_DIR, auto_filter=True):
    db = get_db(persist_dir)
    drugs = [drug] if drug else (detect_drugs(question) if auto_filter else [])
    if not drugs:
        return _search(db, question, k)
    per_drug = max(2, k // len(drugs))
    docs = []
    for d in drugs:
        docs.extend(_search(db, question, per_drug, d))
    return docs or _search(db, question, k)


def retrieve_hybrid(question, drug=None, k=4, persist_dir=DEFAULT_DB_DIR, auto_filter=True, pool_k=15):
    """Combine Dense vector search and BM25 lexical search using Reciprocal Rank Fusion."""
    dense_candidates = retrieve_dense(question, drug=drug, k=pool_k, persist_dir=persist_dir, auto_filter=auto_filter)
    drugs = [drug] if drug else (detect_drugs(question) if auto_filter else [])
    if drugs:
        bm25_candidates = []
        for d in drugs:
            bm25_candidates.extend(bm25_search(question, k=pool_k // len(drugs), drug=d))
    else:
        bm25_candidates = bm25_search(question, k=pool_k)

    fused = rrf_fuse([dense_candidates, bm25_candidates])
    return fused[:k]


def rerank(question, docs, top_k=4):
    """Rerank candidates using a Cross-Encoder."""
    if not docs:
        return []
    reranker = get_reranker()
    if not reranker:
        return docs[:top_k]
    pairs = [[question, d.page_content] for d in docs]
    try:
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked][:top_k]
    except Exception:
        return docs[:top_k]


def retrieve(question, drug=None, k=4, persist_dir=DEFAULT_DB_DIR, auto_filter=True, mode="hybrid_rerank"):
    """Retrieve top chunks.

    Modes:
      - 'dense': Dense Chroma vector search only
      - 'bm25': BM25 keyword search only
      - 'hybrid': Dense + BM25 via Reciprocal Rank Fusion
      - 'hybrid_rerank': Hybrid RRF retrieval followed by Cross-Encoder reranking
    """
    if mode == "dense":
        return retrieve_dense(question, drug=drug, k=k, persist_dir=persist_dir, auto_filter=auto_filter)
    elif mode == "bm25":
        drugs = [drug] if drug else (detect_drugs(question) if auto_filter else [])
        if drugs:
            res = []
            for d in drugs:
                res.extend(bm25_search(question, k=k // len(drugs), drug=d))
            return res or bm25_search(question, k=k)
        return bm25_search(question, k=k)
    elif mode == "hybrid":
        return retrieve_hybrid(question, drug=drug, k=k, persist_dir=persist_dir, auto_filter=auto_filter)
    else:  # hybrid_rerank default
        candidates = retrieve_hybrid(question, drug=drug, k=max(k * 3, 10), persist_dir=persist_dir, auto_filter=auto_filter)
        return rerank(question, candidates, top_k=k)



def answer(question, drug=None, k=4, persist_dir=DEFAULT_DB_DIR):
    docs = retrieve(question, drug, k, persist_dir)
    context = "\n\n".join(d.page_content for d in docs)
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"context": context, "question": question}), docs


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Can I take ibuprofen with warfarin?"
    text, docs = answer(q)
    print(text)
    print("\nSources:")
    for d in docs:
        print("-", d.metadata["drug"], "|", d.metadata["section"])
