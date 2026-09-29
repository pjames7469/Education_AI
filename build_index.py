import json,os,faiss
from sentence_transformers import SentenceTransformer
docs=[json.loads(x) for x in open("data/rag_documents.jsonl",encoding="utf8") if x.strip()]
m=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
v=m.encode([d["title"]+". "+d["text"] for d in docs],normalize_embeddings=True).astype("float32")
idx=faiss.IndexFlatIP(v.shape[1]); idx.add(v); os.makedirs("outputs",exist_ok=True); faiss.write_index(idx,"outputs/faiss.index")
json.dump(docs,open("outputs/rag_metadata.json","w",encoding="utf8"),indent=2); print("Indexed",len(docs),"documents")
