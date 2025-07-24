"""Utility module for rewriting queries and retrieving documents with FAISS."""
import os
from datetime import datetime
from transformers import pipeline


class QueryRewriter:
    """Class for handling query rewriting operations."""
    
    def __init__(self, model: str = "google/flan-t5-base"):
        """Initialize the query rewriter with a specific model."""
        self.model = model
        self.rewriter = pipeline("text2text-generation", model=model)
    
    def rewrite(self, query: str, max_tokens: int = 50) -> str:
        """Rewrite a query for better search results.
        
        Args:
            query: The original query string
            max_tokens: Maximum number of tokens in the rewritten query
            
        Returns:
            str: The rewritten query
        """
        prompt = f"Rewrite the question for search: {query}"
        result = self.rewriter(prompt, max_new_tokens=max_tokens)[0]["generated_text"]
        return result
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


class SearchEngine:
    """Simple wrapper that rewrites a query and retrieves documents."""
    
    # Class-level constants for chunking configuration
    DEFAULT_CHUNK_SIZE = 500
    # DEFAULT_CHUNK_SIZE = 800  # Uncomment if you want to use a larger default chunk size
    DEFAULT_CHUNK_OVERLAP = 100
    DEFAULT_CHUNK_METADATA = {
        "source": "text",
        "created_at": None,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP
    }

    def __init__(self,
                 rewriter_model: str = "google/flan-t5-base",
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 summarizer_model: str | None = None,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """Initialize the search engine with custom chunking parameters."""
        # Initialize the query rewriter
        self.rewriter = QueryRewriter(model=rewriter_model)

        # Embedding model used for generating document vectors
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

        # FAISS database instance built after documents are loaded
        self.db = None

        # Optional summarizer pipeline
        self.summarizer = None
        if summarizer_model:
            self.summarizer = pipeline("summarization", model=summarizer_model)
            
        # Store chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, file_path: str, metadata: dict | None = None) -> None:
        """Load a text file and build the FAISS index with metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
            
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File {file_path} is empty")
            
        try:
            # Prepare metadata
            base_metadata = self.DEFAULT_CHUNK_METADATA.copy()
            base_metadata.update({
                "source": os.path.basename(file_path),
                "created_at": datetime.now().isoformat(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            })
            if metadata:
                base_metadata.update(metadata)
            
            # Load and process text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if not text:
                    raise ValueError(f"File {file_path} is empty")
                
            # Split documents with metadata
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Create documents with metadata
            docs = splitter.create_documents(
                texts=[text],
                metadatas=[base_metadata]
            )
            
            if not docs:
                raise ValueError("No documents were created after splitting")
                
            # Create FAISS index
            self.db = FAISS.from_documents(docs, self.embeddings)
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def search(self, query: str, k: int = 2, rewrite: bool = True):
        """Return the top k relevant documents for the query."""
        if self.db is None:
            raise ValueError("Database not loaded. Call load_documents first.")
        # Optionally rewrite the query before retrieval
        query_to_use = self.rewriter.rewrite(query) if rewrite else query

        # Retrieve top-k relevant documents from FAISS
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query_to_use)

    def summarize_documents(self, docs):
        """Summarize documents with metadata awareness."""
        if not self.summarizer:
            raise ValueError("Summarizer model not provided.")
            
        try:
            # 處理每個文檔單獨進行摘要
            summaries = []
            for doc in docs:
                text = doc.page_content
                if len(text) > 500:  # 如果單一文檔太長，進行截斷
                    text = text[:500] + "..."
                
                # 添加 metadata
                if hasattr(doc, 'metadata'):
                    source = doc.metadata.get('source', 'unknown')
                    text = f"[Source: {source}] {text}"
                
                try:
                    summary = self.summarizer(
                        text,
                        max_length=120,
                        min_length=30,
                        do_sample=False
                    )[0]["summary_text"]
                    summaries.append(summary)
                except Exception as e:
                    print(f"Warning: Failed to summarize a document: {str(e)}")
                    summaries.append(text[:100] + "...")  # 使用文本開頭作為備用
            
            # 合併所有摘要
            if not summaries:
                return "無法生成摘要：沒有可用的文本"
                
            # 如果有多個摘要，進行最終合併
            final_text = " ".join(summaries)
            if len(final_text) > 500:
                final_text = final_text[:500]
            
            try:
                final_summary = self.summarizer(
                    final_text,
                    max_length=120,
                    min_length=30,
                    do_sample=False
                )[0]["summary_text"]
                return final_summary
            except Exception as e:
                return final_text  # 如果最終摘要失敗，返回合併的文本
                
        except Exception as e:
            return f"無法生成摘要：{str(e)}"


if __name__ == "__main__":
    # 示範獨立使用 QueryRewriter
    rewriter = QueryRewriter()
    original_query = "這篇文章在講甚麼？"
    rewritten_query = rewriter.rewrite(original_query)
    print(f"原始查詢: {original_query}")
    print(f"重寫查詢: {rewritten_query}\n")

    # 初始化搜尋引擎
    engine = SearchEngine(
        summarizer_model="sshleifer/distilbart-cnn-12-6",
        chunk_size=300,
        chunk_overlap=50
    )

    # 添加自定義 metadata
    custom_metadata = {
        "category": "article",
        "language": "zh-tw",
        "tags": ["example", "test"]
    }

    # 載入文件
    engine.load_documents("example.txt", metadata=custom_metadata)

    # 執行搜尋
    results = engine.search(original_query, k=2)
    
    print("\n🔎 最相關段落 (含metadata):")
    for i, doc in enumerate(results, 1):
        print(f"段落 {i}: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}\n")

    print("\n📄 摘要:")
    print(engine.summarize_documents(results))
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("這篇文章在講甚麼？")

