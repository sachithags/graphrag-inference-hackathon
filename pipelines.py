
import pickle, time, json
import networkx as nx
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = "gsk_Qmlq6BWiw4"
groq_client = Groq(api_key=GROQ_API_KEY)

# Load the graph
with open("/content/graph.pkl", "rb") as f:
    G = pickle.load(f)

# Setup ChromaDB for Basic RAG (use the collection you already created, or create a new one)
chroma_client = chromadb.Client()
try:
    collection = chroma_client.get_collection("medical_rag")
except:
    collection = chroma_client.create_collection("medical_rag")
    # If collection is empty, you'll need to add documents before calling pipeline_basic_rag.
    # For now, we'll handle the case of empty collection.

def pipeline_llm_only(question):
    start = time.time()
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": question}
        ],
        max_tokens=400
    )
    latency = time.time() - start
    tokens = resp.usage.total_tokens
    cost = tokens * 0.00000059
    return {
        "answer": resp.choices[0].message.content,
        "tokens": tokens,
        "latency": round(latency, 2),
        "cost": round(cost, 6),
        "path": []
    }

def pipeline_basic_rag(question):
    start = time.time()
    # Try to query collection, if empty fallback to LLM-only
    try:
        results = collection.query(query_texts=[question], n_results=3)
        contexts = results["documents"][0]
        context = "\n".join(contexts)
    except:
        context = "No indexed documents found. Answering without context."
    prompt = f"Answer using only this context:\n{context}\n\nQuestion: {question}"
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400
    )
    latency = time.time() - start
    tokens = resp.usage.total_tokens
    cost = tokens * 0.00000059
    return {
        "answer": resp.choices[0].message.content,
        "tokens": tokens,
        "latency": round(latency, 2),
        "cost": round(cost, 6),
        "path": []
    }

def pipeline_graphrag(question):
    start = time.time()
    # Extract keywords and find seed nodes in graph
    keywords = [w.lower() for w in question.replace("?","").split() if len(w)>3]
    seeds = [n for n in G.nodes() if any(k in n.lower() for k in keywords)][:3]
    triples = []
    path = []
    for node in seeds:
        for nb in list(G.neighbors(node))[:4]:
            rel = G[node][nb].get("relation","related")
            triples.append(f"{node} {rel} {nb}")
            path.append(f"{node} →[{rel}]→ {nb}")
            # 2-hop
            for nb2 in list(G.neighbors(nb))[:2]:
                rel2 = G[nb][nb2].get("relation","related")
                triples.append(f"{nb} {rel2} {nb2}")
                path.append(f"{nb} →[{rel2}]→ {nb2}")
    context = "\n".join(triples[:10]) if triples else "No relevant graph connections found."
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer based only on this graph knowledge:\n" + context},
            {"role": "user", "content": question}
        ],
        max_tokens=400
    )
    latency = time.time() - start
    tokens = resp.usage.total_tokens
    cost = tokens * 0.00000059
    return {
        "answer": resp.choices[0].message.content,
        "tokens": tokens,
        "latency": round(latency, 2),
        "cost": round(cost, 6),
        "path": path[:5],
        "graph_context": context
    }
