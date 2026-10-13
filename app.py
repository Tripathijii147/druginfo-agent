import os

import streamlit as st

import rag  # noqa: F401  (loads .env)
from agent import ask, make_graph
from config import DEFAULT_DB_DIR, LABELS_PATH
from drugs import available_drugs

st.set_page_config(page_title="DrugInfo Agent", page_icon="💊", layout="centered")
st.title("💊 DrugInfo Agent")
st.caption("An agentic RAG assistant over FDA drug labels. Every claim cites the label section. "
           "Not medical advice; ask a doctor or pharmacist.")

if not os.path.isdir(DEFAULT_DB_DIR):
    if os.path.exists(LABELS_PATH):
        with st.spinner("Building the index (first run only, about a minute)..."):
            from build_index import build
            build()
    else:
        st.error("No index found. Run fetch_labels.py and build_index.py first.")
        st.stop()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to .env (local) or to the Space secrets (deployed).")
    st.stop()


@st.cache_resource
def get_graph():
    return make_graph()


EXAMPLES = ["Can I take ibuprofen with warfarin?",
            "What are the contraindications of lisinopril?",
            "Which side effects are most reported for atorvastatin?"]

with st.sidebar:
    st.subheader("How it works")
    st.markdown("A LangGraph agent decides which tool to call:\n"
                "- **search_labels**: vector search (ChromaDB + BGE embeddings)\n"
                "- **check_interaction**: scans both labels' interaction sections\n"
                "- **adverse_event_stats**: live FDA FAERS data\n\n"
                "An emergency guardrail runs before the LLM.")
    st.caption("Indexed drugs: " + ", ".join(available_drugs()))
    st.subheader("Try")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat = []

if "chat" not in st.session_state:
    st.session_state.chat = []   # [{"role", "text", "steps"}]

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("steps"):
            with st.expander(f"Agent steps ({len(m['steps'])} tool call(s))"):
                for s in m["steps"]:
                    st.markdown(f"**{s['tool']}** `{s['args']}`")
                    st.caption(s["output"][:900])

question = st.chat_input("Ask about a medicine...") or st.session_state.pop("pending", None)
if question:
    history = [(m["role"], m["text"]) for m in st.session_state.chat][-8:]
    st.session_state.chat.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent is working..."):
                text, steps = ask(get_graph(), question, history)
        except Exception as e:  # noqa: BLE001
            text, steps = f"Something went wrong: {type(e).__name__}: {e}", []
        st.markdown(text)
        if steps:
            with st.expander(f"Agent steps ({len(steps)} tool call(s))"):
                for s in steps:
                    st.markdown(f"**{s['tool']}** `{s['args']}`")
                    st.caption(s["output"][:900])
    st.session_state.chat.append({"role": "assistant", "text": text, "steps": steps})
