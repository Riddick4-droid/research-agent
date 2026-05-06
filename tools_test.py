from tools.arxiv_tool import ArxivTool
from tools.text_processing import combine_papers, split_text, remove_duplicates
from tools.vector_store import create_vector_store, get_embedding_model, retrieve, retrieve_with_score
tool = ArxivTool()

papers = tool.run("machine learning",3)
print("Papers found:")
for paper in papers.split("\n\n"):
    print(paper['title'])
    print("\n---\n")

combined_text = combine_papers(papers)
print("Combined Text:")
print(combined_text)

chunks = split_text(combined_text)
print("Text Chunks:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n---\n")

unique_chunks = remove_duplicates(chunks)
print("Unique Chunks:")
for i, chunk in enumerate(unique_chunks):
    print(f"Unique Chunk {i+1}:\n{chunk}\n---\n")       

embedding_model = get_embedding_model()
vector_store = create_vector_store(unique_chunks, embedding_model)

docs = retrieve(vector_store, "What are the key findings in machine learning?")
print("Retrieved Documents:")
for doc in docs:
    print(doc[:200])  # Print the first 200 characters of each retrieved document