import chromadb 
from chromadb.config import Settings 
from typing import List, Dict, Optional
import numpy as np 
from pathlib import Path 

class VectorStore: 
    def __init__(
        self, 
        collection_name: str = "engineering_docs",
        persist_directory: str = "../../data/vector_store"
    ):
        self.collection_name = collection_name 
        self.persist_directory = Path(persist_directory)

        self.persist_directory.mkdir(parents=True, exist_ok=True) 

        print(f"Initializing ChromaDB at {self.persist_directory}")

        self.client = chromadb.PersistentClient(path=str(self.persist_directory)) 

        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            metadata = {"description": "Engineering documentation embeddings"} 
        ) 

        print(f"Collection '{collection_name}' ready with {self.collection.count()} created successfully")

    def add_chunks(
        self, 
        chunks: List,
        embeddings: np.ndarray
    ) -> None: 

        if len(chunks) != len(embeddings): 
            raise ValueError("Number of chunks and embeddings must match") 
        
        print(f"\nAdding {len(chunks)} chunks to the vector store...") 

        ids = [] 
        documents = [] 
        metadatas = [] 
        embeddings_list = [] 

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)): 

            chunk_id = f"{chunk.metadata.get('title', 'Unknown')}_{chunk.chunk_index}_{i}" 
            chunk_id = chunk_id.replace(" ", "_").replace('/', '_')

            ids.append(chunk_id) 
            documents.append(chunk.text)

            metadata = {
                'title': str(chunk.metadata.get('title', 'Unknown')),
                'source': str(chunk.metadata.get('source', 'Unknown')),
                'chunk_index': chunk.chunk_index,
                'chunk_size': len(chunk.text)
            }

            metadatas.append(metadata) 
            embeddings_list.append(embedding.tolist()) 

        batch_size = 100 

        for i in range(0, len(ids), batch_size): 
            batch_end = min(i+batch_size, len(ids)) 

            self.collection.add(
                ids = ids[i:batch_end],
                documents = documents[i:batch_end],
                metadatas = metadatas[i:batch_end],
                embeddings = embeddings_list[i:batch_end]
            )

            print(f"Added {batch_end-i} chunks to the vector store") 

        print(f"\n{'='*60}") 
        print(f"Successfully added {len(ids)} chunks to the vector store") 
        print(f"Total chunks added: {len(ids)}") 


    def search(
        self, 
        query_embedding: np.ndarray,
        top_k: int = 5, 
        filter_metadata: Optional[Dict] = None
    ) -> Dict: 

        query_embedding_list = query_embedding.tolist() 
        results = self.collection.query(
            query_embeddings = [query_embedding_list], 
            n_results = top_k,
            where = filter_metadata
        )

        return { 
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0],
            'ids': results['ids'][0]
        }
    
    def search_by_text(
        self, 
        query_text: str,
        embedder, 
        top_k: int = 5
    ) -> Dict: 
        query_embedding = embedder.embed_text(query_text) 

        return self.search(query_embedding, top_k) 
    
    def get_stats(self) -> Dict: 
        count = self.collection.count() 

        if count == 0: 
            return {'total_documents': 0}
        
        sample = self.collection.peek(limit=1) 

        return { 
            'collection_name': self.collection_name, 
            'total_documents': count, 
            'persist_directory': str(self.persist_directory),
            'sample_metadata': sample['metadatas'][0] if sample['metadatas'] else {}
        }
    
    def clear(self) -> None: 
        self.client.delete_collection(self.collection_name) 
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            metadata = {"description": "Engineering documentation embeddings"}
        )

        print(f"Collection '{self.collection_name}' cleared successfully") 

    def delete_by_filter(self, filter_metadata: Dict) -> None: 
        results = self.collection.get(where = filter_metadata) 
        if results['ids']: 
            self.collection.delete(ids = results['ids']) 
            print(f"Deleted {len(results['ids'])} documents") 

if __name__ == "__main__": 
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'ingestion'))
    from document_loader import DocumentLoader 
    from text_chunker import TextChunker 
    from embedder import Embedder 

    print(f"{'='*60}") 
    print("BUILDING VECTOR STORE") 
    print(f"{'='*60}") 

    print("\nStep 1: Loading documents...") 
    loader = DocumentLoader() 
    documents = loader.load_directory() 

    print(f"\nLoaded {len(documents)} documents") 

    print("\nStep 2: Chunking documents...") 
    chunker = TextChunker(chunk_size = 512, chunk_overlap = 50)
    chunks = chunker.chunk_documents(documents) 

    print(f"\nChunked into {len(chunks)} chunks") 

    print("\nStep 3: Embedding chunks...") 
    embedder = Embedder() 
    embeddings = embedder.embed_chunks(chunks) 

    print(f"\nGenerated {len(embeddings)} embeddings") 

    print("\nStep 4: Building vector store...") 
    vector_store = VectorStore(collection_name = "engineering_docs")

    if vector_store.collection.count()>0: 
        print("\nClearing existing vector store...") 
        vector_store.clear() 

    print("\nAdding chunks to vector store...") 
    vector_store.add_chunks(chunks, embeddings) 

    print("\nStep 5: Testing search...") 
    test_queries = [
        "How do I optimize Redis caching?",
        "What are the best practices for Go error handling?",
        "How to implement microservices communication?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        results = vector_store.search_by_text(query, embedder, top_k=3)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        )):
            print(f"\nResult {i+1} (distance: {distance:.4f}):")
            print(f"Source: {metadata['title']}")
            print(f"Text preview: {doc[:150]}...")
    
    print("\n" + "="*60) 
    print("VECTOR STORE STATS") 
    print("="*60) 

    stats = vector_store.get_stats() 
    for k,v in stats.items(): 
        print(f"{k}: {v}")
    
    print(f"\nCollection Name: {stats['collection_name']}") 
    print("\nVector Store Setup Complete") 


    
    
