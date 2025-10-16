from typing import List, Union 
import numpy as np 
from sentence_transformers import SentenceTransformer 
import time 

class Embedder: 
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"): 
        print(f"Initializing Embedder with model: {model_name}") 
        self.model = SentenceTransformer(model_name) 
        self.dimension = self.model.get_sentence_embedding_dimension() 
        print(f"Model loaded successfully. Dimension: {self.dimension}") 

    def embed_text(self, text:str) -> np.ndarray: 
        return self.model.encode(text, convert_to_numpy = True) 
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray: 

        return self.model.encode(
            texts,
            batch_size = batch_size,
            show_progress_bar = show_progress,
            convert_to_numpy = True,
        )
    
    def embed_chunks(self, chunks: List)-> List[np.ndarray]: 
        texts = [chunk.text for chunk in chunks] 

        print(f"\n Generating embeddings for {len(texts)} chunks...") 
        start_time = time.time() 

        embeddings = self.embed_batch(texts) 

        end_time = time.time() 
        duration = end_time - start_time 

        print(f"\n Embedding completed in {duration:.2f} seconds") 
        return embeddings 
    
if __name__ == "__main__": 
    from document_loader import DocumentLoader 
    from text_chunker import TextChunker 

    print("Step 1: Loading documents...") 
    loader = DocumentLoader() 
    docs = loader.load_directory() 

    print(f"Step 2: Loaded {len(docs)} documents") 

    chunker = TextChunker(chunk_size=512, chunk_overlap=50) 
    chunks = chunker.chunk_documents(docs) 

    embedder = Embedder() 
    embeddings = embedder.embed_chunks(chunks) 

    print(f"Step 3: Generated {len(embeddings)} embeddings") 

    print(f"\n{'='*60}")
    print("Embedding Summary")
    print(f"{'='*60}")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings[0].shape}")
    print(f"Total size in memory: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\n{'='*60}")
    print("Sample Embedding")
    print(f"{'='*60}")
    print(f"First 10 values: {embeddings[0][:10]}")
    print(f"Embedding range: [{embeddings[0].min():.4f}, {embeddings[0].max():.4f}]")
    
    
