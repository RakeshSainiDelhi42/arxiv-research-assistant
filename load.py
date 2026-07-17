import os
from dotenv import load_dotenv
import psycopg
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

text_dir = "data/text"
BATCH_SIZE = 64

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

all_chunks = []

text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]

for filename in text_files:
    paper_id = filename.replace(".txt", "")
    with open(os.path.join(text_dir, filename), encoding="utf-8") as f:
        text = f.read()

    for i, chunk in enumerate(splitter.split_text(text)):
        all_chunks.append({
            "paper_id": paper_id,
            "chunk_index": i,
            "content": chunk,
        })

print(f"{len(all_chunks)} chunks ready")

texts = [c["content"] for c in all_chunks]

print(f"Embedding {len(texts)} chunks...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
)
print(f"Embeddings shape: {embeddings.shape}")

rows = [
    (c["paper_id"], c["chunk_index"], c["content"], emb.tolist())
    for c, emb in zip(all_chunks, embeddings)
]

with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE chunks;")
        cur.executemany(
            "INSERT INTO chunks (paper_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
            rows,
        )

print(f"Inserted {len(rows)} chunks")