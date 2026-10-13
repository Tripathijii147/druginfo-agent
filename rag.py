"""Core RAG logic: retrieve label chunks from ChromaDB and answer with citations."""
from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import COLLECTION, DEFAULT_DB_DIR, EMBED_MODEL, GROQ_MODEL, REFUSAL
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


def _search(db, question, k, drug=None):
    kwargs = {"k": k}
    if drug:
        kwargs["filter"] = {"drug": drug}
    return db.as_retriever(search_kwargs=kwargs).invoke(question)


def retrieve(question, drug=None, k=4, persist_dir=DEFAULT_DB_DIR, auto_filter=True):
    """Return the top chunks. If drug names are found in the question, search only their labels."""
    db = get_db(persist_dir)
    drugs = [drug] if drug else (detect_drugs(question) if auto_filter else [])
    if not drugs:
        return _search(db, question, k)

    per_drug = max(2, k // len(drugs))   # interaction questions get chunks from each drug
    docs = []
    for d in drugs:
        docs.extend(_search(db, question, per_drug, d))
    return docs or _search(db, question, k)  # drug not in the index -> fall back to no filter


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
