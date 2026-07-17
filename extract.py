import os
import re
import fitz

os.makedirs("data/text", exist_ok=True)

pdf_dir = "data/pdfs"
text_dir = "data/text"

pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDFs")

for filename in pdf_files:
    paper_id = filename.replace(".pdf", "")
    doc = fitz.open(os.path.join(pdf_dir, filename))

    text = ""
    for page in doc:
        text += page.get_text()
    page_count = doc.page_count
    doc.close()

    # Ligatures extract as single glyphs, which embed as non-words
    text = text.replace("\x00", "")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"\n{3,}", "\n\n", text)

    out_path = os.path.join(text_dir, f"{paper_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"{paper_id}: {page_count} pages, {len(text)} chars")