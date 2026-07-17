import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_dir = "data/text"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

all_chunks = []

text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]

for filename in text_files:
    paper_id = filename.replace(".txt", "")
    with open(os.path.join(text_dir, filename), encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "paper_id": paper_id,
            "chunk_index": i,
            "content": chunk,
        })

print(f"{len(all_chunks)} chunks from {len(text_files)} papers")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

vector = model.encode(all_chunks[0]["content"])
print(f"Vector dimension: {len(vector)}")
print(f"First 5 values: {vector[:5]}")