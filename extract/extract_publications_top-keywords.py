import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("quant_ph_arxiv.csv")


def extract_year(arxiv_id):
    match = re.match(r'^(\d{2})(\d{2})\.', arxiv_id)
    if match:
        year = int(match.group(1))
        return year + 2000 if year < 50 else year + 1900
    return None

df["year"] = df["id"].apply(extract_year)


df_years = df["year"].value_counts().sort_index()
df_years.to_csv("publications_by_year.csv", header=["count"])


df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")


vectorizer = CountVectorizer(stop_words="english", max_features=100)
X = vectorizer.fit_transform(df["text"])
words = vectorizer.get_feature_names_out()
frequencies = X.sum(axis=0).A1

df_keywords = pd.DataFrame({
    "word": words,
    "frequency": frequencies
}).sort_values(by="frequency", ascending=False)

df_keywords.to_csv("top_keywords.csv", index=False)
