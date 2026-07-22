# College Notes RAG Chatbot

A chatbot that answers questions using your own uploaded college notes (PDF, Word,
PowerPoint, or plain text) instead of relying only on general knowledge.

## How it works

1. You upload notes → they're split into overlapping text chunks.
2. Each chunk is turned into a vector embedding **locally** (no API needed for this step)
   using `sentence-transformers`.
3. Embeddings are stored in a **FAISS** index on disk, one per "subject/notebook", so your
   notes persist between sessions.
4. When you ask a question, the app embeds your question, retrieves the most relevant
   chunks, and sends them + your question to an LLM (Claude or GPT) to generate a grounded
   answer with sources cited.
5. If you don't add an API key, the app still works — it just shows you the raw relevant
   passages from your notes instead of a synthesized answer.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Using it

1. In the sidebar, create a subject (e.g. "Operating Systems", "Thermodynamics").
2. Upload your notes (PDF / DOCX / PPTX / TXT) — multiple files at once is fine.
3. (Optional but recommended) Add an API key so you get real generated answers instead
   of just raw excerpts. Three providers are supported — pick whichever you like:
   - **Groq (free)** — no credit card required, generous free tier, several open models
     (Llama 3.3, Llama 3.1, GPT-OSS). Get a key at console.groq.com → API Keys.
   - **Anthropic (Claude)** — console.anthropic.com → API Keys (small free credit, then paid).
   - **OpenAI** — platform.openai.com → API Keys (paid).

   You can either paste the key directly into the sidebar each session, **or** set it once
   via Streamlit secrets so it's picked up automatically (see below).
4. Ask questions in the chat box at the bottom.

## Setting your API key via Secrets (recommended — no re-typing it every time)

Instead of pasting a key into the sidebar each session, you can store it once:

**Running locally:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Then edit `.streamlit/secrets.toml` and fill in whichever key(s) you use, e.g.:
```toml
GROQ_API_KEY = "gsk_..."
```
This file is already in `.gitignore` so it will never get committed to GitHub. Restart
the app, and the sidebar will show "Using GROQ_API_KEY from secrets ✅" instead of asking
for a key.

**Deployed on Streamlit Community Cloud:**
Go to your app → **Manage app** → **Settings** → **Secrets**, and paste the same TOML
content there. It's stored securely server-side and never shown in the app's UI.

## Features

- **Multi-format ingestion**: PDF, Word (.docx), PowerPoint (.pptx), and plain text notes.
- **Local, free embeddings**: uses `sentence-transformers` (`all-MiniLM-L6-v2`) — no API
  cost or key needed just to index your notes.
- **Persistent vector storage**: FAISS index is saved to disk per subject, so you don't
  need to re-upload notes every session.
- **Multiple subjects/notebooks**: keep different courses' notes in separate, isolated
  indexes so retrieval doesn't mix unrelated subjects.
- **Source-cited answers**: every answer names which file(s) it drew from, and you can
  expand a "View retrieved passages" panel to see the exact chunks used.
- **Three supported LLM providers**: Groq (free tier, multiple open models to pick from),
  Anthropic (Claude), or OpenAI — switch anytime from the sidebar.
- **Secrets-based key storage**: set your API key once via Streamlit secrets instead of
  pasting it into the UI every session; falls back to a manual field if no secret is set.
- **Works with zero API key**: falls back to showing the top-matching raw passages from
  your notes so the tool is still useful offline.
- **Chat memory**: recent conversation turns are included in the prompt so you can ask
  natural follow-up questions ("what about part 2?").
- **Manage your notes**: remove individual files or clear a whole notebook from the sidebar.
- **Adjustable retrieval depth**: slider to control how many note chunks are pulled in per
  question (more context vs. more precision).

## Notes / things you may want to extend

- Currently uses character-based chunking (800 chars, 150 overlap) — good enough for most
  notes, but you could swap in a smarter splitter (e.g. by heading/paragraph) for very
  structured notes.
- FAISS index here uses exact search (`IndexFlatIP`), which is fine up to tens of
  thousands of chunks; for huge note collections you'd want an approximate index (e.g.
  `IndexIVFFlat`).
- No authentication — this is built for personal/local use. Add login if you plan to
  deploy it for multiple users sharing one server.
