"""Demo: find the most relevant document by MEANING, not keywords."""
from dotenv import load_dotenv
import numpy as np
import voyageai

load_dotenv()
vo = voyageai.Client()  # reads VOYAGE_API_KEY from environment

MODEL = "voyage-4-lite"

documents = [
    "Employees may work remotely up to three days per week with manager approval.",
    "Travel expenses over $500 require a receipt and VP sign-off.",
    "The company laptop must be returned within 5 business days after leaving.",
    "Health insurance enrollment opens every November for two weeks.",
]

queries = [
    "Can I work from home?",                 # no shared keywords with doc 1
    "Who approves a big flight cost?",       # 'flight' never appears in docs
    "What happens to my computer when I quit?",
]

# 1 request for ALL documents (free tier = 3 requests/minute)
doc_result = vo.embed(documents, model=MODEL, input_type="document")
doc_vecs = np.array(doc_result.embeddings)

# 1 request for ALL queries
query_result = vo.embed(queries, model=MODEL, input_type="query")
query_vecs = np.array(query_result.embeddings)

print(f"Vector size: {doc_vecs.shape[1]} numbers per text")
print(f"Tokens used: {doc_result.total_tokens + query_result.total_tokens}\n")

# Voyage vectors are normalized, so dot product = cosine similarity
scores = query_vecs @ doc_vecs.T   # shape: (num_queries, num_docs)

for i, q in enumerate(queries):
    best = int(np.argmax(scores[i]))
    print(f"Q: {q}")
    print(f"   -> {documents[best]}")
    print(f"   score={scores[i][best]:.3f}  (all: {np.round(scores[i], 3).tolist()})\n")