from pathlib import Path
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

TXT_FILE = "sample_data.txt"

def main():
    path = Path(TXT_FILE)
    if not path.exists():
        print(f"File not found: {TXT_FILE}")
        print("Create a text file named sample_data.txt in this folder and run again.")
        return

    loader = TextLoader(TXT_FILE, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Cat is my Friend. Cat can drink milk. Cat can sleep on the sofa. Cat is very cute. Cat likes to play with yarn. Cat is a good companion. Cat can purr when happy. Cat has sharp claws. Cat can climb trees. Cat is a popular pet.
# [Cat],[Cat is users frind],[Cat is Drinks milk],[Cat can sleep on the sofa],[Cat is very cute],[Cat likes to play with yarn],[Cat is a good companion],[Cat can purr when happy],[Cat has sharp claws],[Cat can climb trees],[Cat is a popular pet]
    
    llm = ChatOllama(model="phi3", temperature=0)

    question = input("Enter your question: ").strip()
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Use only the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\n=== Retrieved Chunks ===\n")
    for i, doc in enumerate(docs, start=1):
        print(f"[Chunk {i}]")
        print(doc.page_content)
        print("-" * 50)

    print("\n=== Final Answer ===\n")
    print(response.content)

if __name__ == "__main__":
    main()
