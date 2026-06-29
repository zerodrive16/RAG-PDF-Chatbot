import uuid 
import fitz # pymupdf
from qdrant_client import QdrantClient 
from qdrant_client.models import VectorParams, Distance, PointStruct, Query
from openai import OpenAI 
from dotenv import load_dotenv


load_dotenv()

EMBED_MODEL = "text-embedding-3-large" # Embedding the model 
EMBED_DIM = 3072 # dimension space size for vector points
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500 # Chunks text 
CHUNK_OVERLAP = 50 # prevents text from getting cut out (important information)

# Start with Docker 
qdrant = QdrantClient(host="localhost", port=6333)

openai_client = OpenAI()


# Creates a collection 
if not qdrant.collection_exists(COLLECTION_NAME): 
    qdrant.create_collection(
        collection_name=COLLECTION_NAME, 
        vectors_config = VectorParams(size=EMBED_DIM, distance=Distance.COSINE) # Using Cosine cuz its optimized by OpenAIs embeddings
    )


# Send the request to openAI 
def embed_texts(texts: list[str]) -> list[list[float]]: 
    response  = openai_client.embeddings.create(
        model=EMBED_MODEL, 
        input=texts, 
    )
    return [item.embedding for item in response.data] # Returns a vector 


# Read the PDF and chunk it 
def chunk_text(pages: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    full_text = "\n".join(pages)
    chunks = []
    start = 0 
    while start < len(full_text): 
        chunks.append(full_text[start: start + chunk_size])
        start += chunk_size - overlap 
    return chunks 


# Insert and Update (upsert) in dimensional space 
def store_chunks(chunks: list[str], embeddings: list[list[float]]) -> None: 
    points = [
        PointStruct(
            id = str(uuid.uuid4()), 
            vector = embedding, 
            payload = {"text": chunk}, 
        ) for chunk, embedding in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


# Search the OpenAI prompt with the vector database 
def search(query: str, top_k: int = 5) -> list[str]: 
    query_vector = embed_texts([query])[0]
    results = qdrant.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
    ).points
    return [found.payload["text"] for found in results]


# Finds the chunks and sends it to OpenAI 
def chat(query: str, history: list[dict]) -> str: 
    context = "\n\n".join(search(query))
    system_prompt = (
        "You are a helpful assistant. Answer questions based on the following "
        "context from the uploaded documents. "
        "If the answer cannot be found in the context, say so honestly.\n\n"
        f"Context:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]
    response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


# Read the PDF and extract the text per page 
def extract_text_from_pdf(path: str) -> list[str]:
    doc = fitz.open(path)
    return [page.get_text() for page in doc]


# Preprocess the text to chunks to embeddings and then store in Vector DB
def ingest_pdf(path: str) -> int:
    pages = extract_text_from_pdf(path)
    chunks = chunk_text(pages)
    embeddings = embed_texts(chunks)
    store_chunks(chunks, embeddings)
    return len(chunks)
