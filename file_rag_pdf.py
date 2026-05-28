from pathlib import Path
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FILE = "sample.pdf"

def main():
    path = Path(PDF_FILE)
    if not path.exists():
        print(f"File not found: {PDF_FILE}")
        print("Place your PDF in this folder and rename it to sample.pdf, or update PDF_FILE in the code.")
        return

    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="phi3", temperature=0)

    question = input("Enter your question: ").strip()
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer the question using only the PDF context below.

Context:  
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\n=== Retrieved PDF Chunks ===\n")
    for i, doc in enumerate(docs, start=1):
        print(f"[Chunk {i}]")
        print(doc.page_content[:1000])
        print("-" * 50)

    print("\n=== Final Answer ===\n")
    print(response.content)

if __name__ == "__main__":
    main()
