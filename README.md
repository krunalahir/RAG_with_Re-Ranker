# RAG System (Retrieval-Augmented Generation) with Cross-Encoders

A comprehensive Retrieval-Augmented Generation (RAG) system designed to enhance language model responses with relevant context retrieved from documents. This modular system allows for flexible document processing, embedding, and retrieval workflows.

## 🚀 Overview

This RAG system implements a complete pipeline for processing documents and answering questions based on their content. The system follows a modular architecture that separates concerns into distinct components:

- **Document Loading**: Load various document formats (PDF, TXT)
- **Text Chunking**: Split documents into manageable chunks
- **Embedding**: Generate vector representations of text
- **Vector Storage**: Store embeddings in FAISS for efficient similarity search
- **Retrieval**: Find relevant document chunks based on queries
- **Re-Ranking**: Advanced technique using cross-encoders to re-rank retrieved results for improved relevance
- **Generation**: Generate answers using retrieved context

## 🏗️ Architecture

```
[Document] → [Chunker] → [Embedder] → [Vector Store] → [Retriever] → [Re-Ranker] → [LLM] → [Answer]
```

### Components

- **Document Loader**: Supports PDF and TXT formats with extensible base class for additional formats
- **Chunker**: Splits documents into smaller, manageable chunks with configurable sizes
- **Embedder**: Uses Sentence Transformers for high-quality text embeddings
- **Vector Store**: Leverages FAISS for efficient similarity search
- **Retriever**: Finds most relevant document chunks for a given query
- **Re-Ranker**: Advanced technique that re-ranks retrieved results using cross-encoders for improved relevance
- **Generator**: Combines retrieved context with LLM to generate answers

## 🛠️ Tech Stack

- **Python**: Core programming language
- **FAISS**: Efficient similarity search and clustering of dense vectors
- **Sentence Transformers**: State-of-the-art sentence, text, and image embeddings
- **Cross-Encoders**: Advanced re-ranking using cross-encoder models for improved relevance
- **PyPDF2/pdfplumber**: PDF document processing
- **NumPy**: Numerical computing
- **Abstract Base Classes**: Enforced modularity and extensibility

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-system

# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

Make sure to install the following packages:

```bash
pip install torch torchvision torchaudio
pip install sentence-transformers  # Includes cross-encoders
pip install faiss-cpu
pip install PyPDF2 pdfplumber
pip install transformers
```

## 🚀 Quick Start

### Example Usage

```python
from Document_Loader.pdf_loader import PdfLoader
from Chunking.simple_chunker import SimpleChunker
from Embedding.sentence_transformer import SentenceTransformerEmbedder
from vector_store.faiss_store import FaissVectorStore
from Retriever.retriever import Retriever
from Re_Ranker import ReRanker
from generator.llm import LLMGenerator

# Initialize components
loader = PdfLoader()
docs = loader.load("sample.pdf")

chunker = SimpleChunker()
chunks = chunker.chunk(docs)

embedder = SentenceTransformerEmbedder()
embeddings = embedder.embed_documents(chunks)

store = FaissVectorStore(len(embeddings[0]))
store.add(embeddings, chunks)

retriever = Retriever(embedder, store)
reranker = ReRanker()  # Initialize the re-ranker
llm = LLMGenerator()

# Ask a question
question = "What is RAG and what is retrieval process how faiss handle it?"
initial_contexts = retriever.retrieve(question)  # Initial retrieval from vector store

# Apply re-ranking to improve relevance
reranked_contexts = reranker.rerank(question, initial_contexts, top_k=3)
context_text = " ".join(reranked_contexts)
answer = llm.generate(context_text, question)

print(answer)
```

## 🧱 Modules

### Document Loader
- `PdfLoader`: Loads PDF documents and extracts text content
- `TxtLoader`: Loads plain text documents
- Extensible base class for adding new loaders

### Chunking
- `SimpleChunker`: Basic text chunking with configurable size and overlap
- Abstract base class for implementing custom chunking strategies

### Embedding
- `SentenceTransformerEmbedder`: Uses pre-trained transformer models for text embeddings
- Abstract base class for integrating different embedding models

### Vector Store
- `FaissVectorStore`: Efficient vector storage and similarity search using FAISS
- Abstract base class for supporting alternative vector databases

### Retriever
- `Retriever`: Performs semantic search to find relevant document chunks
- Configurable number of results and search parameters

### Generator
- `LLMGenerator`: Combines retrieved context with language model for answer generation
- Flexible interface for different LLM providers

### Re-Ranker
- `ReRanker`: Implements advanced re-ranking using cross-encoders to improve result relevance
- Uses pre-trained models like "cross-encoder/ms-marco-MiniLM-L-6-v2" for semantic matching
- Takes initial retrieval results and re-scores them based on query-chunk similarity
- Returns top-k most relevant chunks after re-ranking for enhanced answer quality

## 🔧 Configuration

The system is designed to be highly configurable:

- **Chunk Size**: Adjust the size of text chunks for optimal retrieval
- **Overlap**: Control overlap between chunks to preserve context
- **Top K**: Number of relevant chunks to retrieve for each query
- **Embedding Model**: Switch between different sentence transformer models
- **Vector Store Parameters**: Customize FAISS index for performance needs

## 🧪 Testing

Run the example with the provided sample PDF:

```bash
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow the abstract base classes when implementing new components
- Maintain consistent naming conventions
- Write docstrings for all public methods

## 🙏 Acknowledgments

- FAISS team for efficient similarity search
- Sentence Transformers for high-quality embeddings
- Open-source community for various libraries used in this project

## 🐛 Issues

If you encounter any issues or have suggestions for improvements, please open an issue in the repository.