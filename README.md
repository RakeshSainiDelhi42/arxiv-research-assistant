# arXiv Research Assistant

A retrieval-augmented generation (RAG) system that answers questions about machine learning research papers, grounded in a corpus of arXiv papers, with inline source citations.

## Live demo

https://arxiv-research-assistant-by-rakesh.streamlit.app



## What it does

Ask a question about transformer and attention mechanism research. The system embeds the question, runs a semantic search over 2,249 text chunks drawn from 29 arXiv papers, and passes the closest matches to a language model with instructions to answer only from that context.

Every claim in the answer carries the arXiv id of the paper it came from. The retrieved chunks are displayed alongside the answer with their similarity scores, so any claim can be checked against its source. When the retrieved context does not support an answer, the system declines rather than guessing.

## Results

Retrieval was evaluated against a hand-labelled question set, where each question is tagged with the paper that should answer it.

| Metric | Value |
|---|---|
| Retrieval hit rate (top-5) | 85.7% (6/7) |
| Answerable questions — worst distance | 0.4179 |
| Unanswerable questions — best distance | 0.7403 |
| Separation gap | 0.3225 |

The refusal threshold is set to 0.60, roughly midway between the two distributions. An earlier value of 0.75 was chosen by intuition and later shown by measurement to be too permissive: all three unanswerable test questions passed the gate and were caught only by the prompt.

The single retrieval miss was the question "What is the generalized attention mechanism?" The expected paper uses that phrase in its title, but the wording is generic enough to match many papers in the corpus weakly rather than one strongly. Questions containing distinctive vocabulary retrieved the correct paper at rank 1.

## Architecture

```
arXiv API → PDF download → text extraction → chunking → embedding → pgvector
                                                                        ↓
                                    question → embedding → similarity search
                                                                        ↓
                                              grounded prompt → LLM → cited answer
```

| Component | Choice |
|---|---|
| Vector store | pgvector on PostgreSQL 18 (Neon) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2`, 384 dimensions |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, 1000 chars, 150 overlap |
| Index | HNSW with cosine distance |
| LLM | Llama 3.3 70B via Groq |
| UI | Streamlit |

## Setup

```bash
git clone https://github.com/RakeshSainiDelhi42/arxiv-research-assistant.git
cd arxiv-research-assistant

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt

cp .env.example .env              # then fill in real values
```

`.env` needs two values: a PostgreSQL connection string for a database with the pgvector extension available, and a Groq API key.

Build the corpus and populate the database:

```bash
python db_setup.py                # create table and HNSW index
python -m src.ingest              # download PDFs from arXiv
python -m src.extract             # extract text from PDFs
python -m src.load                # chunk, embed, insert
```

Run the evaluation:

```bash
python -m src.evaluate
```

Run the app:

```bash
streamlit run app.py
```

All commands are run from the project root.

## Design decisions

**pgvector rather than a dedicated vector database.** The alternative was Pinecone, Chroma, or similar. Postgres was already the datastore for three prior projects, so co-locating vectors alongside relational data avoids running a second system, keeps one backup and access story, and means the vector search is ordinary SQL. For a corpus of this size the performance difference is not the deciding factor; the operational simplicity is.

**Managed Postgres for development, not just deployment.** Streamlit Community Cloud cannot reach a database running on a laptop, so a hosted database was needed for the live demo regardless. Developing against the same database that production uses removes a class of environment-drift bugs, and Neon ships pgvector preinstalled, which also avoids compiling the extension on Windows.

**Chunk size of 1000 characters with 150 overlap.** 1000 characters is roughly 250 tokens, which matches the embedding model's input limit, so no text is silently truncated. It is also approximately one dense paragraph of a research paper, which tends to be one coherent idea. Smaller chunks lose the context that makes a passage interpretable; larger chunks average several ideas into one vector, which matches many queries weakly instead of one strongly. The 150-character overlap means a sentence spanning a chunk boundary still appears complete in at least one chunk.

**Recursive character splitting.** Chunks break at paragraph boundaries first, then line breaks, then sentence ends, and only cut mid-word as a last resort. Chunks therefore end at natural boundaries rather than arbitrary offsets.

**HNSW index with cosine distance.** The index metric matches the metric the embedding model was trained on. HNSW is an approximate nearest-neighbour method: it trades a small amount of recall for a large reduction in search time by avoiding a full scan of every vector.

**Refusal threshold calibrated on measured data.** The gate compares the nearest chunk's distance against a threshold and refuses before calling the LLM if it is exceeded. The value was chosen from the observed separation between answerable and unanswerable questions rather than picked by intuition.

**Two independent refusal layers.** The distance gate is mechanical and does not depend on the model's cooperation. The prompt instruction is a second layer that catches cases where retrieval returns close-but-unhelpful chunks. Both were tested independently.

**PDF text extraction with PyMuPDF.** Extraction quality was inspected on sample papers before committing to the approach, including an equation-heavy paper where symbol handling was most at risk. Prose and reading order survived intact. Ligatures and null bytes are stripped at extraction rather than downstream, so the text files on disk are clean for any consumer.

**Downloads fetched by URL rather than through the arxiv library's downloader.** The library's download method changed across major versions. The `pdf_url` attribute is stable data, so fetching it with an HTTP request decouples ingestion from a changing third-party API.

**The LLM is swappable.** Retrieval, chunking, and grounding do not depend on which model generates the answer. This was not theoretical: the project began on Gemini and moved to Groq when the Gemini free tier returned a zero quota allocation, with no changes to the retrieval layer.

## Known limitations

**Retrieval hit rate measures paper-level retrieval, not chunk-level usefulness.** A question can retrieve the correct paper and still fail to be answered, because the specific chunks returned may not contain the relevant passage. This happened on a real test question: "Which datasets were used to evaluate the Music Transformer?" retrieved five chunks from the correct paper, all of them from the listening-study section, none naming the datasets. The system correctly refused rather than inventing an answer, but the refusal was caused by a retrieval shortfall, not an absence in the corpus. The word "evaluate" matched the paper's evaluation methodology section rather than its experiments section.

**Generic queries retrieve poorly.** Questions phrased with vocabulary common across the corpus match many chunks weakly rather than one strongly.

**The evaluation set is small.** Seven answerable and three unanswerable questions, written by hand. It is enough to expose real behaviour and calibrate the threshold, but the hit rate carries wide error bars at this sample size.

**The embedding model is small.** `all-MiniLM-L6-v2` was chosen because it runs locally on CPU with no API dependency. Larger embedding models would likely improve retrieval. The evaluation harness exists to measure any such change rather than assume it.

**Corpus scope.** 29 papers on transformers and attention. One paper was skipped during ingestion because its PDF returned a 404 from arXiv.
