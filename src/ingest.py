import os
import arxiv
import requests

search = arxiv.Search(
    query='"attention mechanism" AND transformer',
    max_results=30,
    sort_by=arxiv.SortCriterion.Relevance,
)

client = arxiv.Client()

os.makedirs("data/pdfs", exist_ok=True)

for result in client.results(search):
    paper_id = result.entry_id.split("/")[-1]
    filepath = f"data/pdfs/{paper_id}.pdf"
    try:
        print(f"Downloading {paper_id} | {result.title[:60]}")
        response = requests.get(result.pdf_url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
    except requests.exceptions.HTTPError as e:
        print(f"  SKIPPED {paper_id}: {e}")

print("Done.")