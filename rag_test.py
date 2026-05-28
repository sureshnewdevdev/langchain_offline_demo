from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def main():
    docs = [
        Document(page_content="LangChain is a framework for building LLM applications."),
        Document(page_content="Ollama helps run open-source models locally on your computer."),
        Document(page_content="RAG means Retrieval Augmented Generation. It retrieves relevant content before answering."),
        Document(page_content="Embeddings convert text into numeric vectors for similarity search."),
    ]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOllama(model="phi3", temperature=0)

    question = "What is RAG?"
    retrieved_docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
Answer only from the context below.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\n=== Retrieved Context ===\n")
    print(context)
    print("\n=== Final Answer ===\n")
    print(response.content)

if __name__ == "__main__":
    main()
