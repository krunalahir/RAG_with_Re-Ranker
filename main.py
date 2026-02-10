from Document_Loader.pdf_loader import PdfLoader
from Chunking.simple_chunker import SimpleChunker
from Embedding.sentence_transformer import SentenceTransformerEmbedder
from vector_store.faiss_store import FaissVectorStore
from Retriever.retriever import Retriever
from generator.llm import LLMGenerator


loader = PdfLoader()
docs = loader.load("sample.pdf")

chunker = SimpleChunker()
chunks = chunker.chunk(docs)

embedder = SentenceTransformerEmbedder()
embeddings = embedder.embed_documents(chunks)

store = FaissVectorStore(len(embeddings[0]))
store.add(embeddings, chunks)

retriever = Retriever(embedder, store)

llm = LLMGenerator()
question = "What is RAG and what is retrieval process how faiss handle it?"
contexts = retriever.retrieve(question)

context_text = " ".join(contexts)

answer = llm.generate(context_text, question)

print(answer )