from rag.retriever import search
from rag.generator import generate_answer

question = "How do I send a POST request with JSON?"
chunks = search(question, top_k=4)
answer = generate_answer(question, chunks)
print(answer)