"""
app.py
Streamlit UI for the College Notes RAG Chatbot.

Run with:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from rag_engine import VectorStore, DocumentLoader, generate_answer

st.set_page_config(page_title="College Notes Chatbot", page_icon="📚", layout="wide")

# --------------------------------------------------------------------------
# Session state setup
# --------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # subject -> list[(role, text, sources)]

# --------------------------------------------------------------------------
# Sidebar: subject / notebook selection
# --------------------------------------------------------------------------
st.sidebar.title("📚 Notebooks")

existing_subjects = VectorStore.list_subjects()
new_subject = st.sidebar.text_input("Create a new subject/notebook", placeholder="e.g. Data Structures")
if st.sidebar.button("➕ Create") and new_subject.strip():
    existing_subjects = sorted(set(existing_subjects + [new_subject.strip()]))

if not existing_subjects:
    st.sidebar.info("Create a subject above to get started (e.g. 'Operating Systems').")
    st.stop()

subject = st.sidebar.selectbox("Active subject", existing_subjects,
                                index=existing_subjects.index(new_subject.strip())
                                if new_subject.strip() in existing_subjects else 0)

with st.spinner("Loading vector index..."):
    store = VectorStore(subject)

st.session_state.chat_history.setdefault(subject, [])

# --------------------------------------------------------------------------
# Sidebar: upload notes
# --------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Upload notes")
uploaded_files = st.sidebar.file_uploader(
    "PDF, DOCX, PPTX, or TXT",
    type=["pdf", "docx", "pptx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files and st.sidebar.button("📥 Add to notebook"):
    with st.spinner("Reading, chunking, and embedding your notes..."):
        for uf in uploaded_files:
            suffix = Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name
            try:
                text = DocumentLoader.load(tmp_path)
                n_chunks = store.add_document(text, uf.name)
                st.sidebar.success(f"{uf.name}: added {n_chunks} chunks")
            except Exception as e:
                st.sidebar.error(f"{uf.name}: failed ({e})")

# --------------------------------------------------------------------------
# Sidebar: manage existing sources
# --------------------------------------------------------------------------
sources = store.list_sources()
st.sidebar.markdown("---")
st.sidebar.subheader(f"Notes in '{subject}' ({len(sources)})")
if sources:
    for s in sources:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(f"📄 {s}")
        if col2.button("✕", key=f"del_{s}"):
            store.remove_source(s)
            st.rerun()
    if st.sidebar.button("🗑️ Clear entire notebook"):
        store.clear()
        st.rerun()
else:
    st.sidebar.caption("No notes uploaded yet.")

# --------------------------------------------------------------------------
# Sidebar: LLM provider settings
# --------------------------------------------------------------------------
# Known model choices per provider (kept short & current; edit freely).
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
ANTHROPIC_MODELS = ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]


def secret_key(name):
    """Read an API key from Streamlit secrets (st.secrets) if present,
    so a deployer can set it once in Settings > Secrets and never paste it
    into the UI. Returns None if not configured -- falls back to manual entry."""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


st.sidebar.markdown("---")
st.sidebar.subheader("AI model")
provider = st.sidebar.selectbox(
    "Provider",
    ["None (extract passages only)", "Groq (free)", "Anthropic (Claude)", "OpenAI"],
)
api_key = None
model_name = None
provider_key = "none"

if provider == "Groq (free)":
    provider_key = "groq"
    secret_val = secret_key("GROQ_API_KEY")
    if secret_val:
        st.sidebar.success("Using GROQ_API_KEY from secrets ✅")
        api_key = secret_val
    else:
        api_key = st.sidebar.text_input(
            "Groq API key", type="password",
            help="Free at console.groq.com — no credit card needed.",
        )
    model_name = st.sidebar.selectbox("Model", GROQ_MODELS)

elif provider == "Anthropic (Claude)":
    provider_key = "anthropic"
    secret_val = secret_key("ANTHROPIC_API_KEY")
    if secret_val:
        st.sidebar.success("Using ANTHROPIC_API_KEY from secrets ✅")
        api_key = secret_val
    else:
        api_key = st.sidebar.text_input("Anthropic API key", type="password")
    model_name = st.sidebar.selectbox("Model", ANTHROPIC_MODELS)

elif provider == "OpenAI":
    provider_key = "openai"
    secret_val = secret_key("OPENAI_API_KEY")
    if secret_val:
        st.sidebar.success("Using OPENAI_API_KEY from secrets ✅")
        api_key = secret_val
    else:
        api_key = st.sidebar.text_input("OpenAI API key", type="password")
    model_name = st.sidebar.selectbox("Model", OPENAI_MODELS)

top_k = st.sidebar.slider("Chunks to retrieve", min_value=2, max_value=8, value=4)

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.title("📚 College Notes Chatbot")
st.caption(f"Notebook: **{subject}**  •  {len(sources)} file(s) indexed  •  "
           f"{'🔌 ' + provider if provider_key != 'none' else '⚠️ no AI model connected — answers show raw note excerpts'}")

for role, text, used_sources in st.session_state.chat_history[subject]:
    with st.chat_message(role):
        st.markdown(text)
        if used_sources:
            with st.expander("View retrieved passages"):
                for c in used_sources:
                    st.markdown(f"**{c['source']}** (score {c['score']:.2f})")
                    st.text(c["text"][:500])

query = st.chat_input("Ask a question about your notes...")
if query:
    if not sources:
        st.warning("Upload some notes first so I have something to search!")
    else:
        st.session_state.chat_history[subject].append(("user", query, None))
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching notes and generating answer..."):
                retrieved = store.search(query, top_k=top_k)
                history_pairs = [(r, t) for r, t, _ in st.session_state.chat_history[subject][:-1]]
                try:
                    answer = generate_answer(
                        query, retrieved,
                        provider=provider_key, api_key=api_key, model=model_name,
                        history=history_pairs,
                    )
                except Exception as e:
                    answer = f"⚠️ Error calling the AI model: {e}\n\nFalling back to raw passages."
                    from rag_engine import extractive_fallback
                    answer += "\n\n" + extractive_fallback(retrieved)
            st.markdown(answer)
            if retrieved:
                with st.expander("View retrieved passages"):
                    for c in retrieved:
                        st.markdown(f"**{c['source']}** (score {c['score']:.2f})")
                        st.text(c["text"][:500])

        st.session_state.chat_history[subject].append(("assistant", answer, retrieved))
