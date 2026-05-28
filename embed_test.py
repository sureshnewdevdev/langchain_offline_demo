from langchain_ollama import OllamaEmbeddings

def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector = embeddings.embed_query("What is LangChain?")
    print("\n=== Embedding Test ===\n")
    print("Vector length:", len(vector))
    print("First 10 values:", vector[:10])

if __name__ == "__main__":
    main()
