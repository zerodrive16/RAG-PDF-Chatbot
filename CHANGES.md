# RAG PDF Chatbot — Änderungen & fehlende Implementierungen

## Übersicht der Dateien

| Datei | Aktion |
|-------|--------|
| `main.py` | Bestehende Datei — Bugs fixen + Funktionen ergänzen |
| `app.py` | Neue Datei erstellen — Streamlit UI |

---

## `main.py` — Fixes & Ergänzungen

### 1. Imports & Konstanten (Zeile 1–11)

**Keine Änderung an den Imports nötig.** Aber Variablennamen anpassen:

```python
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072          # umbenannt von EMBED → EMBED_DIM (klarer)
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500          # neu — für Chunking
CHUNK_OVERLAP = 50        # neu — für Chunking
```

---

### 2. Qdrant Client & Collection (Zeile 14–19)

**Bug:** `create_collection` wird bei jedem Start aufgerufen und crasht, wenn die Collection schon existiert.

**Vorher:**
```python
client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=EMBED, distance=Distance.EUCLID)
)
```

**Nachher:**
```python
qdrant = QdrantClient(host="localhost", port=6333)   # umbenannt: client → qdrant

if not qdrant.collection_exists(COLLECTION_NAME):    # nur erstellen wenn nötig
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),  # EUCLID → COSINE
    )
```

**Warum `Distance.COSINE` statt `Distance.EUCLID`?**
OpenAI Embeddings sind für Cosine Similarity optimiert. Euclid misst absolute Abstände — das führt zu schlechteren Suchergebnissen.

---

### 3. `embed_texts` (Zeile 34–39) — 2 Bugs

**Bug 1:** `client` ist der Qdrant-Client, nicht der OpenAI-Client.
**Bug 2:** Gibt das ganze Response-Objekt zurück statt die Vektoren.

**Vorher:**
```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(   # ❌ falscher client
        model=EMBED_MODEL,
        input=texts,
    )
    return response                        # ❌ gibt Response-Objekt zurück
```

**Nachher:**
```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(   # ✅ richtiger client
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]  # ✅ gibt Vektoren zurück
```

---

### 4. Neue Funktion: `chunk_text` — nach `extract_text_from_pdf` einfügen

**Warum?** Eine PDF-Seite kann tausende Zeichen lang sein — zu groß für eine sinnvolle Suche. Chunks sind kleinere, überlappende Textblöcke.

```python
def chunk_text(pages: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    full_text = "\n".join(pages)
    chunks = []
    start = 0
    while start < len(full_text):
        chunks.append(full_text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks
```

---

### 5. Neue Funktion: `store_chunks` — nach `embed_texts` einfügen

**Warum?** Ohne diese Funktion werden die Embeddings zwar erstellt, aber nie in Qdrant gespeichert.

```python
def store_chunks(chunks: list[str], embeddings: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk},
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
```

---

### 6. Neue Funktion: `search` — nach `store_chunks` einfügen

**Warum?** Das ist das Herzstück des RAG — sucht die relevantesten Chunks zu einer Frage.

```python
def search(query: str, top_k: int = 5) -> list[str]:
    query_vector = embed_texts([query])[0]
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    return [hit.payload["text"] for hit in results]
```

---

### 7. Neue Funktion: `chat` — nach `search` einfügen

**Warum?** Schickt die gefundenen Chunks als Kontext an GPT-4o und generiert eine Antwort.

```python
def chat(query: str, history: list[dict]) -> str:
    context = "\n\n".join(search(query))
    system_prompt = (
        "Du bist ein hilfreicher Assistent. Beantworte Fragen basierend auf dem folgenden "
        "Kontext aus den hochgeladenen Dokumenten. "
        "Wenn die Antwort nicht im Kontext zu finden ist, sage das ehrlich.\n\n"
        f"Kontext:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]
    response = openai_client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content
```

---

### 8. Neue Funktion: `ingest_pdf` — ans Ende von `main.py`

**Warum?** Fasst die gesamte Ingestion-Pipeline in einer Funktion zusammen — wird von der Streamlit-App aufgerufen.

```python
def ingest_pdf(path: str) -> int:
    pages = extract_text_from_pdf(path)
    chunks = chunk_text(pages)
    embeddings = embed_texts(chunks)
    store_chunks(chunks, embeddings)
    return len(chunks)   # gibt Anzahl gespeicherter Chunks zurück
```

---

## `app.py` — Neue Datei erstellen

Diese Datei komplett neu anlegen. Sie enthält die Streamlit-Benutzeroberfläche.

```python
import os
import tempfile
import streamlit as st
from main import ingest_pdf, chat

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("PDF Chatbot")

# Gesprächsverlauf im Session State speichern
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: PDF hochladen
with st.sidebar:
    st.header("PDF hochladen")
    uploaded_file = st.file_uploader("PDF auswählen", type="pdf")
    if uploaded_file and st.button("Verarbeiten"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with st.spinner("PDF wird verarbeitet..."):
            num_chunks = ingest_pdf(tmp_path)
        os.unlink(tmp_path)
        st.success(f"{num_chunks} Chunks gespeichert!")

# Gesprächsverlauf anzeigen
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Eingabe & Antwort
if prompt := st.chat_input("Stelle eine Frage zum Dokument..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Denke nach..."):
            answer = chat(prompt, st.session_state.history[:-1])
        st.write(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
```

**Starten mit:**
```bash
streamlit run app.py
```

---

## Vollständige Pipeline nach allen Änderungen

```
PDF Upload (Streamlit)
    ↓
ingest_pdf(path)
    ↓
extract_text_from_pdf()  →  Seiten als Liste
    ↓
chunk_text()             →  Kleine Textblöcke (500 Zeichen, 50 Overlap)
    ↓
embed_texts()            →  Vektoren via OpenAI text-embedding-3-large
    ↓
store_chunks()           →  Speichern in Qdrant (Cosine Distance)
    
User stellt Frage (Streamlit Chat)
    ↓
chat(query, history)
    ↓
search(query)            →  Query embedden → Qdrant Suche → Top 5 Chunks
    ↓
openai chat.completions  →  GPT-4o mit Kontext + Gesprächsverlauf
    ↓
Antwort anzeigen
```

---

## Zusammenfassung aller Bugs

| Zeile | Bug | Fix |
|-------|-----|-----|
| 14 | `client` als Name für Qdrant | Umbenennen zu `qdrant` |
| 16–19 | `create_collection` crasht bei 2. Start | `if not qdrant.collection_exists(...)` |
| 18 | `Distance.EUCLID` | `Distance.COSINE` |
| 35 | `client.embeddings.create` (falscher Client) | `openai_client.embeddings.create` |
| 39 | `return response` (Response-Objekt) | `return [item.embedding for item in response.data]` |
