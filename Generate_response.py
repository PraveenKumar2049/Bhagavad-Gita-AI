import  json
import numpy as np
import faiss
import os
import requests
from sentence_transformers import SentenceTransformer


INDEX_PATH= "embeddings/faiss_index.bin"
VERSE_MAP_PATH = "embeddings/verse_map.json"

index = faiss.read_index(INDEX_PATH)
with open(VERSE_MAP_PATH,'r',encoding="utf-8") as f:
    verses= json.load(f)
model= SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_verses(query, top_k=3):
    query_vec=model.encode([query])
    distances,indices = index.search(np.array(query_vec,dtype=np.float32), top_k)
    valid_indices= [int(idx) for idx in indices[0] if 0<= idx < len(verses)]
    return [verses[i] for i in valid_indices]

def ask_krishna(query, verses):
    context = "\n\n".join([f"Chapter {v['chapter']} Verse {v['verse']}: {v['english']}" for v in verses])
    prompt = f"""
You are Lord Krishna speaking to Arjuna.
Use the Bhagavad Gita verses below to answer user's question
with calm wisdom and compassion.

Verses:
{context}

Question: {query}

Answer as Krishna would:
"""
    payload={"model":"phi3:3.8b","prompt": prompt}
    r= requests.post("http://localhost:11434/api/generate", json=payload,stream=True)
    full_reply= ""
    for line in r.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if 'response' in data:
                full_reply += data["response"]
    full_reply = full_reply.replace("<think>", "").replace ("</think>", "")
    return full_reply.strip() if full_reply else "No reply from model"

import gradio as gr

#if __name__=="__main__":
    #
    #print("Krishna AI - Ask your Question or type 'exit' to quit. \n")
    #while True:
    #    q= input("You: ")
    #    if q.lower()=="exit":
    #        break
    #    verses=retrieve_verses(q)
    #    reply = ask_krishna(q, verses)
    #    print(f"\nKrishna: {reply}\n")

def krishna_chat(query):
    verses = retrieve_verses(query)
    reply= ask_krishna(query, verses)
    return reply
with gr.Blocks(title="Krishna AI") as demo:
    gr.Markdown("## Krishna AI - Ask your Question")
    query = gr.Textbox(label="Your Question", placeholder="Ask somehing to lord Krishna...")
    output= gr.Textbox(label="Krishna's Response", lines=6)
    ask_btn= gr.Button("Ask Krishna")
    ask_btn.click(fn=krishna_chat, inputs=query,outputs=output)


demo.launch(share=False,server_port=7860)



