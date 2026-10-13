import os

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # small, runs fine on CPU
COLLECTION = "labels"
DEFAULT_DB_DIR = "chroma_db"
LABELS_PATH = "data/labels.jsonl"

# Groq model names get retired from time to time. If you see a "model not found"
# error, pick a current model at https://console.groq.com/docs/models and put it
# in your .env file as GROQ_MODEL=... (no code change needed).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# The exact sentence the model must use when the excerpts do not contain the answer.
REFUSAL = "The retrieved label text does not answer this."
