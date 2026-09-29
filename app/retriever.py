import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.chunks = []
        self.embeddings = None

    def index(self, transcript):
        self.chunks = transcript

        texts = [
            chunk["text"]
            for chunk in transcript
        ]

        print("Creating embeddings...")

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

    def search(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[index]

            results.append({
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "score": float(scores[index])
            })

        return results