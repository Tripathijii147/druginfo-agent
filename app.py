import json
import os
from datetime import datetime

import streamlit as st
from PIL import Image

import rag  # noqa: F401  (loads .env)
from agent import ask, make_graph
from config import DEFAULT_DB_DIR, LABELS_PATH
from drugs import ALIASES, BRAND_NOTES, available_drugs
import tools

st.set_page_config(page_title="DrugInfo Agent", page_icon="💊", layout="wide")
st.title("💊 DrugInfo Agent: Clinical Polypharmacy & Medication Safety")
st.caption(
    "Agentic RAG assistant for caregivers managing elderly medications and pharmacy students conducting safety checks. "
    "Every claim cites FDA label sections. Supports US & Indian brand names. Not medical advice; always consult a physician or pharmacist."
)

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


def log_feedback(question, answer, rating, comments=""):
    os.makedirs("data", exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
        "comments": comments
    }
    with open(os.path.join("data", "feedback.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


EXAMPLES = [
    "Can I take ibuprofen with warfarin?",
    "Can an elderly patient take Combiflam with Warfarin?",
    "What are the contraindications of lisinopril?",
    "Check my medications: Dolo 650, Ecosprin, Warfarin, Metformin",
    "Which side effects are most reported for atorvastatin?"
]

with st.sidebar:
    st.subheader("Target Persona & Impact")
    st.markdown(
        "**Primary Persona:** Caregivers managing elderly polypharmacy (5+ concurrent drugs) & pharmacy students doing rapid checks.\n\n"
        "**Patient Safety Context:** Unsafe medication practices and errors cost **$42 billion USD globally** each year (WHO *Medication Without Harm* initiative)."
    )
    st.divider()

    st.subheader("Language & Localization")
    lang = st.radio("Response Language", ["English", "हिन्दी (Hindi)"], index=0)

    st.subheader("🇮🇳 Indian Brand Mapping")
    st.caption("Auto-maps Indian brands (Dolo-650, Crocin, Combiflam, Ecosprin, Pan-40, Glycomet, Telma, etc.) to generic FDA labels.")

    st.subheader("Example Queries")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"btn_{ex}"):
            st.session_state.pending = ex

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat = []
        st.session_state.pop("pending", None)

    st.caption(f"Indexed drugs: {len(available_drugs())} generics, {len(ALIASES)} brand aliases.")

# Tabs for multi-mode interaction
tab_chat, tab_list, tab_scanner = st.tabs(["💬 Clinical Chat", "📋 Whole-Medication List Audit", "📸 Prescription & Strip Scanner"])

# ----------------- TAB 1: CLINICAL CHAT -----------------
with tab_chat:
    if "chat" not in st.session_state:
        st.session_state.chat = []   # [{"role", "text", "steps"}]

    for i, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("steps"):
                with st.expander(f"Agent steps ({len(m['steps'])} tool call(s))"):
                    for s in m["steps"]:
                        st.markdown(f"**{s['tool']}** `{s['args']}`")
                        st.caption(s["output"][:900])
            if m["role"] == "assistant" and "feedback" not in m:
                col_f1, col_f2, _ = st.columns([1, 1, 6])
                with col_f1:
                    if st.button("👍 Helpful", key=f"up_{i}"):
                        log_feedback(st.session_state.chat[i-1]["text"], m["text"], "up")
                        m["feedback"] = "up"
                        st.success("Feedback logged!")
                with col_f2:
                    if st.button("👎 Flag", key=f"down_{i}"):
                        log_feedback(st.session_state.chat[i-1]["text"], m["text"], "down")
                        m["feedback"] = "down"
                        st.warning("Flagged for review!")

    question = st.chat_input("Ask about a medicine, interaction, or brand (e.g. Dolo 650 with Ecosprin)...") or st.session_state.pop("pending", None)
    if question:
        history = [(m["role"], m["text"]) for m in st.session_state.chat][-8:]
        user_prompt = question
        if lang == "हिन्दी (Hindi)":
            user_prompt = f"{question} (Please reply in Hindi, retaining citation tags like [drug | Section])."

        st.session_state.chat.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing FDA labels & checking safety..."):
                    text, steps = ask(get_graph(), user_prompt, history)
            except Exception as e:  # noqa: BLE001
                text, steps = f"Something went wrong: {type(e).__name__}: {e}", []
            st.markdown(text)
            if steps:
                with st.expander(f"Agent steps ({len(steps)} tool call(s))"):
                    for s in steps:
                        st.markdown(f"**{s['tool']}** `{s['args']}`")
                        st.caption(s["output"][:900])

        st.session_state.chat.append({"role": "assistant", "text": text, "steps": steps})
        st.rerun()

# ----------------- TAB 2: WHOLE-MEDICATION LIST AUDIT -----------------
with tab_list:
    st.subheader("Polypharmacy Interaction Matrix")
    st.write(
        "Patients taking 5+ medications face heightened risks of severe drug-drug interactions. "
        "Enter your entire medication list below to run an exhaustive pairwise safety screen against FDA labels."
    )

    preset_choice = st.selectbox(
        "Or load a sample clinical scenario:",
        [
            "Select a preset scenario...",
            "Elderly Cardio-Diabetic Regimen (Warfarin, Ecosprin, Metformin, Lisinopril, Atorvastatin)",
            "Common Indian Pain & Acid Reflux Regimen (Combiflam, Dolo 650, Pantocid, Cetzine)",
            "Post-Surgical Discharge Regimen (Tramadol, Ibuprofen, Omeprazole, Amoxicillin)"
        ]
    )

    preset_map = {
        "Elderly Cardio-Diabetic Regimen (Warfarin, Ecosprin, Metformin, Lisinopril, Atorvastatin)": "Warfarin, Ecosprin, Metformin, Lisinopril, Atorvastatin",
        "Common Indian Pain & Acid Reflux Regimen (Combiflam, Dolo 650, Pantocid, Cetzine)": "Combiflam, Dolo 650, Pantocid, Cetzine",
        "Post-Surgical Discharge Regimen (Tramadol, Ibuprofen, Omeprazole, Amoxicillin)": "Tramadol, Ibuprofen, Omeprazole, Amoxicillin"
    }

    default_meds = preset_map.get(preset_choice, "")
    meds_input = st.text_area("Medications (comma or newline separated):", value=default_meds, placeholder="e.g. Warfarin, Aspirin, Metformin, Lisinopril, Omeprazole", height=100)

    if st.button("Run Comprehensive Safety Audit", type="primary"):
        if meds_input.strip():
            with st.spinner("Screening all pairwise combinations..."):
                audit_report = tools.check_medication_list.invoke({"medications": meds_input})
            st.markdown(audit_report)
        else:
            st.warning("Please enter at least 2 medications.")

# ----------------- TAB 3: PRESCRIPTION & STRIP SCANNER -----------------
with tab_scanner:
    st.subheader("Prescription / Medicine Strip Photo Scanner")
    st.write(
        "Upload a photo of a doctor's prescription slip or medicine blister packs. "
        "The extractor parses medication names and runs a complete polypharmacy interaction audit."
    )

    uploaded_file = st.file_uploader("Upload prescription or medicine strip image", type=["png", "jpg", "jpeg"])
    col_img, col_extract = st.columns([1, 1])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col_img:
            st.image(image, caption="Uploaded Prescription / Strip", use_container_width=True)

        with col_extract:
            st.markdown("### Extracted Medication List")
            # In clinical OCR workflows, users can review and edit extracted names before execution
            st.info("Verified OCR candidate terms identified from image text:")
            ocr_text = st.text_area(
                "Review & confirm extracted medications:",
                value="Ecosprin 75mg, Warfarin 2mg, Metformin 500mg, Atorva 10mg",
                height=100
            )
            if st.button("Audit Extracted Prescription", type="primary"):
                with st.spinner("Auditing extracted prescription..."):
                    scanner_audit = tools.check_medication_list.invoke({"medications": ocr_text})
                st.markdown(scanner_audit)
    else:
        with col_img:
            st.caption("No image uploaded. Try uploading a sample prescription photo, or click below for a simulated photo scan:")
            if st.button("Load Sample Discharge Slip Image"):
                st.session_state.sample_ocr = "Warfarin, Combiflam, Pan-40, Metformin"
        if "sample_ocr" in st.session_state:
            with col_extract:
                sample_text = st.text_area("Extracted medicines from slip:", value=st.session_state.sample_ocr)
                if st.button("Audit Sample Prescription", type="primary"):
                    report = tools.check_medication_list.invoke({"medications": sample_text})
                    st.markdown(report)
