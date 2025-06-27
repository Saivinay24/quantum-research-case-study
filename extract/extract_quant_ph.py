from datasets import load_dataset
import pandas as pd


dataset = load_dataset("librarian-bots/arxiv-metadata-snapshot", split="train", streaming=True)

quant_ph_papers = []
limit = 10000  

for i, record in enumerate(dataset):
    if "quant-ph" in record["categories"]:
        quant_ph_papers.append({
    "id": record.get("id", ""),
    "title": record.get("title", ""),
    "authors": record.get("authors", ""),
    "abstract": record.get("abstract", ""),
    "categories": record.get("categories", ""),
    "submitted": record.get("submitted", ""),  
})

    if len(quant_ph_papers) >= limit:
        break


df = pd.DataFrame(quant_ph_papers)
df.to_csv("quant_ph_arxiv.csv", index=False)
print(f"Saved {len(df)} quantum computing papers to quant_ph_arxiv.csv")

