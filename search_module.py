"""Utility module for rewriting queries and retrieving documents with FAISS."""

from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


class SearchEngine:
    """Simple wrapper that rewrites a query and retrieves documents."""

    def __init__(self,
                 rewriter_model: str = "google/flan-t5-base",
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 summarizer_model: str | None = None):
        # Initialize the query rewriter pipeline
        self.rewriter = pipeline("text2text-generation", model=rewriter_model)

        # Embedding model used for generating document vectors
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

        # FAISS database instance built after documents are loaded
        self.db = None

        # Optional summarizer pipeline
        self.summarizer = None
        if summarizer_model:
            self.summarizer = pipeline("summarization", model=summarizer_model)

    def load_documents(self, file_path: str,
                       chunk_size: int = 500,
                       chunk_overlap: int = 50) -> None:
        """Load a text file and build the FAISS index."""
        # Load the raw text documents
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        # Split documents into manageable chunks
        splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                         chunk_overlap=chunk_overlap)
        docs = splitter.split_documents(documents)

        # Create the FAISS index using the embedding model
        self.db = FAISS.from_documents(docs, self.embeddings)

    def rewrite_query(self, query: str) -> str:
        """Use the T5 model to rewrite the query for search."""
        prompt = f"Rewrite the question for search: {query}"
        result = self.rewriter(prompt, max_new_tokens=50)[0]["generated_text"]
        return result

    def search(self, query: str, k: int = 2, rewrite: bool = True):
        """Return the top k relevant documents for the query."""
        if self.db is None:
            raise ValueError("Database not loaded. Call load_documents first.")
        # Optionally rewrite the query before retrieval
        query_to_use = self.rewrite_query(query) if rewrite else query

        # Retrieve top-k relevant documents from FAISS
        retriever = self.db.as_retriever()
        return retriever.get_relevant_documents(query_to_use, k=k)

    def summarize_documents(self, docs):
        """Summarize a list of documents using the summarizer pipeline."""
        if not self.summarizer:
            raise ValueError("Summarizer model not provided.")
        # Concatenate document text and generate a summary
        joined = "\n".join(doc.page_content for doc in docs)
        summary = self.summarizer(joined, max_length=120, min_length=40,
                                  do_sample=False)[0]["summary_text"]
        return summary


if __name__ == "__main__":
    # Example usage
    engine = SearchEngine(summarizer_model="sshleifer/distilbart-cnn-12-6")

    # Build the FAISS index from a local article
    engine.load_documents("sample_article.txt")

    # Execute a query and fetch the most relevant passages
    question = "王鴻薇有沒有被罷免？"
    results = engine.search(question, k=2)

    print("\n🔎 最相關段落:")
    for i, doc in enumerate(results, 1):
        print(f"段落 {i}: {doc.page_content[:100]}...")

    print("\n📄 摘要:")
    print(engine.summarize_documents(results))
