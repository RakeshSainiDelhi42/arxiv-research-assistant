import os
from dotenv import load_dotenv
import psycopg
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

TOP_K = 5
# Calibrated on eval set: answerable questions max 0.42, unanswerable min 0.74.
# 0.60 sits mid-gap with margin on both sides.
DISTANCE_THRESHOLD = 0.60

model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(query, top_k=TOP_K):
    query_vector = model.encode(query).tolist()

    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT paper_id, chunk_index, content,
                       embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY distance
                LIMIT %s;
                """,
                (query_vector, top_k),
            )
            return cur.fetchall()
        
PROMPT_TEMPLATE = """You are a research assistant answering questions about machine learning papers.

Answer the question using ONLY the context below. Each context block is labelled with its source paper id.

Rules:
- Use only information present in the context. Do not use prior knowledge.
- Cite the source paper id inline, in square brackets, after each claim. Example: [2310.10375v3]
- If the context does not contain enough information to answer, reply exactly: "The retrieved context does not contain enough information to answer this question."
- Do not speculate or fill gaps.

Context:
{context}

Question: {question}

Answer:"""

REFUSAL = "The retrieved context does not contain enough information to answer this question."


def answer(question):
    results = retrieve(question)

    if not results or results[0][3] > DISTANCE_THRESHOLD:
        return REFUSAL, []

    context = "\n\n".join(
        f"[{paper_id}]\n{content}" for paper_id, _, content, _ in results
    )

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content, results

if __name__ == "__main__":
    q = "How does relative position representation work in transformers?"
    result, sources = answer(q)
    print(f"Q: {q}\n")
    print(result)
    print(f"\nSources: {[s[0] for s in sources]}")
    print(f"Top distance: {sources[0][3]:.4f}" if sources else "No sources")