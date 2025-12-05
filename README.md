## Kerala Ayurveda RAG Demo 

This project is a lightweight Python-based **Retrieval-Augmented Generation (RAG)** prototype designed for a small Ayurveda knowledge corpus. It demonstrates how to:

* Load and chunk multiple document types (Markdown sections, FAQ Q&A pairs, and CSV product rows)
* Retrieve relevant chunks using a **hybrid BM25-like scoring system with metadata boosting**
* Generate short, grounded answers without using an external LLM
* Return citations, confidence estimates, and potential failure modes

### Key Features

* **Custom chunking** for structured content
* **Hybrid retrieval** using term frequency, coverage, phrase matching, and metadata scoring
* **Simple answer generator** that extracts meaningful sentences from retrieved chunks
* **Sample corpus** includes foundations, FAQs, product catalog, and stress support program

### Run the demo

```bash
python kerala_rag_demo.py
```

This script demonstrates a clear end-to-end RAG flow suitable for learning, teaching, and prototyping.
