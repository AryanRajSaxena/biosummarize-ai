from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class QABot:
    def __init__(self, chunks):
        self.embedder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        self.chunks = chunks
        self.index = faiss.IndexFlatL2(768)
        self.embeddings = self.embedder.encode(chunks)
        self.index.add(np.array(self.embeddings))

    def retrieve_context(self, query, k=3):
        q_embed = self.embedder.encode([query])
        D, I = self.index.search(np.array(q_embed), k)
        return "\n".join([self.chunks[i] for i in I[0]])
