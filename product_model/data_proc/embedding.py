from sentence_transformers import SentenceTransformer


class TextEmbedding:

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    def sent_embedding(self, raw_text):
        embeddings = self.model.encode(raw_text)
        return embeddings
