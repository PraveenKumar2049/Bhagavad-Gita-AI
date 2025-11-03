import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
INDEX_PATH= "embeddings/faiss_index.bin"
VERSE_MAP_PATH= "embeddings/verse_map.json"
index= faiss.read_index(INDEX_PATH)

with open(VERSE_MAP_PATH,"r", encoding="utf-8") as f:
    verses=json.load(f)

model= SentenceTransformer("all-MiniLM-L6-v2")

def search_gita(query,top_k=3):
    query_vec= model.encode([query])
    distances,indices= index.search(np.array(query_vec, dtype=np.float32), top_k)
    for i,idx in enumerate(indices[0]):
        verse= verses[int(idx)]
        print(f"Rank{i+1}: Chapter {verse['chapter']} verse {verse['verse']}")
        print(f"{verse['english']}\n")

if __name__== "__main__":
    while True:
        query= input("What's on your mind? (or type 'exit') ")
        if query.lower()=='exit':
            break
        search_gita(query)

