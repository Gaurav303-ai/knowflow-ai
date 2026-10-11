from sentence_transformers import SentenceTransformer

# Load a pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

text = "I love AI"

embedding = model.encode(text)

print(embedding)
print(type(embedding))
print(len(embedding))