from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")

text1 = "Car"
text2 = "Auto"

emb1 = model.encode(text1)
emb2 = model.encode(text2)

score = util.cos_sim(emb1, emb2)

print(score)