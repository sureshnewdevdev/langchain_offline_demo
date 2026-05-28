# LangChain Offline Demo with Ollama

This project shows how to use **LangChain with offline Ollama models**.

## Models already expected on your system

- `phi3:latest`
- `llama3.2:1b`
- `nomic-embed-text:latest`

## Python packages

Install inside your virtual environment:

```bash
pip install langchain langchain-ollama langchain-community chromadb pypdf
```

## Files in this project

- `chat_test.py` -> simple local chat test using `phi3`
- `embed_test.py` -> simple embedding test using `nomic-embed-text`
- `rag_test.py` -> simple in-memory RAG example
- `file_rag_txt.py` -> ask questions from a local TXT file
- `file_rag_pdf.py` -> ask questions from a local PDF file
- `requirements.txt` -> package list

## Step 1: test chat model

```bash
python chat_test.py
```

## Step 2: test embedding model

```bash
python embed_test.py
```

## Step 3: test simple RAG

```bash
python rag_test.py
```

## Step 4: ask questions from a TXT file

1. Put your content inside `sample_data.txt`
2. Run:

```bash
python file_rag_txt.py
```

## Step 5: ask questions from a PDF file

1. Place your PDF in the project folder
2. Update the PDF filename in `file_rag_pdf.py`
3. Run:

```bash
python file_rag_pdf.py
```

## Notes

- These examples are fully offline except for package installation.
- Ollama must be running on your machine.
- Start with `phi3` because it is a good lightweight model for your setup.
- `nomic-embed-text` is used for embeddings.

## Verify models

```bash
ollama list
```

Expected models:

- `phi3:latest`
- `llama3.2:1b`
- `nomic-embed-text:latest`

## Optional

You can change the model in the code:

```python
ChatOllama(model="llama3.2:1b", temperature=0)
```
