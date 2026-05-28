from langchain_ollama import ChatOllama

def main():
    llm = ChatOllama(model="phi3", temperature=0)
    response = llm.invoke("Explain LangChain in simple words for a beginner.")
    print("\n=== Chat Response ===\n")
    print(response.content)

if __name__ == "__main__":
    main()
