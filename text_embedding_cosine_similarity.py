from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

model = SentenceTransformer('all-MiniLM-L6-v2')

es = Elasticsearch()


index_name = "text-embeddings"
if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 384}  # Dimension of the embedding
                }
            }
        }
    )

def read_sentences_from_file(filename):
    with open(filename, 'r') as file:
        sentences = [line.strip() for line in file.readlines()]
    return sentences

sentences = read_sentences_from_file('sentences.txt')

embeddings = model.encode(sentences)

for i, sentence in enumerate(sentences):
    es.index(
        index=index_name,
        id=i,
        body={
            "text": sentence,
            "embedding": embeddings[i].tolist()  # Convert the embedding to a list
        }
    )


query = input("Enter a query to find similar sentences: ")

query_embedding = model.encode([query])[0]

response = es.search(
    index=index_name,
    body={
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding.tolist(),
                    "k": 1  # Number of nearest neighbors to return
                }
            }
        }
    }
)

most_similar = response['hits']['hits'][0]['_source']
stored_embedding = most_similar['embedding']

similarity_score = cosine_similarity(query_embedding, stored_embedding)

print(f"Most similar sentence: {most_similar['text']}")
print(f"Cosine similarity score: {similarity_score}")
