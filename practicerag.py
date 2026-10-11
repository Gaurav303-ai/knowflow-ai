from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client_gemini = genai.Client(api_key=api_key)


model = SentenceTransformer("all-MiniLM-L6-v2")

loader = PyPDFLoader(
           file_path = "Data/JS Notes.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

print("Number of chunks:", len(texts))
# print("Embedding shape:", embeddings.shape)
# print("First embedding:", embeddings[0])

client = QdrantClient(url="http://localhost:6333")

collection_name = "js_notes"

# 6. Create collection
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    )
)

# 7. Store chunks and embeddings in Qdrant
client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text": texts[i],
                "page": chunks[i].metadata.get("page", 0)
            }
        )
        for i in range(len(texts))
    ]
)

print("Successfully stored", len(texts), "chunks in Qdrant!")

# 8. Verify stored data
print("Points stored:", client.count(
    collection_name=collection_name,
    exact=True
).count)

# # 9. Semantic Search

# query = "What are JavaScript events?"

# # Convert question into an embedding
# query_vector = model.encode(query).tolist()

# # Search for the most relevant chunks in Qdrant
# results = client.query_points(
#     collection_name="js_notes",
#     query=query_vector,
#     limit=3
# ).points

# # Display search results
# for result in results:
#     print("\nScore:", result.score)
#     print("Page:", result.payload["page"])
#     print("Text:", result.payload["text"])

def search_documents(query, limit=3):
    # Step 1: Convert query into a vector
    query_vector = model.encode(query).tolist()

    # Step 2: Search similar vectors in Qdrant
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    ).points

    # Step 3: Display retrieved results
    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Score:", round(result.score, 3))
        print("Page:", result.payload["page"])
        print("Text:", result.payload["text"])



def ask_question(question):
    # Retrieve relevant information from PDF
    context = search_documents(question)

    if not context.strip():
        print("No relevant information found in the PDF.")
        return

    # Ask Gemini to answer using retrieved context
    prompt = f"""
You are a helpful assistant answering questions from a PDF.

Use the context below to answer the question.
If the answer is not present in the context, say:
"I could not find this information in the PDF."

Context:
{context}

Question:
{question}
"""

    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nQuestion:", question)
    print("\nAnswer:", response.text)


ask_question("What are JavaScript events?")

