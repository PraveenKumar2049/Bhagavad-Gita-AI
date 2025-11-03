import json
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
DATA_PATH= "data/bhagavad_gita_clean.json"
OUT_DIR= "embeddings"
os.makedirs(OUT_DIR,exist_ok=True)
df=pd.read_json(DATA_PATH)
verses= df["english"].tolist()
model= SentenceTransformer("all-MiniLM-L6-v2")
embeddings= model.encode(verses, show_progress_bar=True)
embedding_dim= embeddings.shape[1]
index= faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, os.path.join(OUT_DIR,"faiss_index.bin"))
df.to_json(os.path.join(OUT_DIR,"verse_map.json"),orient="records",indent=2,force_ascii=False)
print(f"Created Faiss index with{len(verses)} verses")