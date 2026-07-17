import os
from dotenv import load_dotenv
import psycopg
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "How does relative position representation work in transformers?"

query_vector = model.encode(query).tolist()

with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT paper_id, chunk_index, content,
                   embedding <=> %s::vector AS distance
            FROM chunks
            ORDER BY distance
            LIMIT 5;
            """,
            (query_vector,),
        )
        for paper_id, chunk_index, content, distance in cur.fetchall():
            print(f"\n[{distance:.4f}] {paper_id} chunk {chunk_index}")
            print(content[:300])