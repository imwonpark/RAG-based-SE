from typing import List, Dict, Optional 
import time 
from dataclasses import dataclass

@dataclass 
class RAGResult: 
    query: str 
    answer: str 
    sources: List[Dict]
    retrieval_time_ms: float 
    generation_time_ms: float 
    total_time_ms: float 

    def __repr__(self): 
        return f"RAGResult(query = '{self.query[:50]}...', sources = {len(self.sources)})"

    
class RAGPipeline: 
    def __init__(
        self, 
        vector_store, 
        embedder, 
        use_llm: bool = False, 
        llm_client = None 
    ): 
        self.vector_store = vector_store 
        self.embedder = embedder 
        self.use_llm = use_llm 
        self.llm_client = llm_client 

        if use_llm and llm_client is None: 
            raise ValueError("llm_client is required when use_llm is True") 

    def query(
        self, 
        query_text: str, 
        top_k: int = 5, 
        similarity_threshold: float = 0.7, 
    ): 
        total_start = time.time() 

        retrieval_start = time.time() 
        results = self.vector_store.search_by_text(
            query_text, 
            self.embedder, 
            top_k = top_k
        ) 

        retrieval_time = (time.time() - retrieval_start) * 1000 

        sources = [] 
        context_chunks = [] 

        for doc, metadata, distance in zip(
            results['documents'], 
            results['metadatas'], 
            results['distances']
        ): 

            #Rough normalization since ChromaDB uses L2 Distance
            similarity = 1 - (distance /2) 

            if similarity >= similarity_threshold or len(sources) == 0: 
                sources.append({
                    'text': doc, 
                    'title': metadata.get('title', 'Unknown'), 
                    'source': metadata.get('source', 'Unknown'), 
                    'chunk_index': metadata.get('chunk_index', 0), 
                    'distance': distance, 
                    'similarity': similarity 
                })

                context_chunks.append(doc) 
        generation_start = time.time() 

        if self.use_llm: 
            answer = self._generate_answer_with_llm(query_text, context_chunks) 
        else: 
            answer = self._generate_answer_without_llm(query_text, context_chunks) 

        generation_time = (time.time() - generation_start) * 1000 
        total_time = (time.time() - total_start) * 1000 

        return RAGResult(
            query = query_text,
            answer = answer, 
            sources = sources, 
            retrieval_time_ms = retrieval_time, 
            generation_time_ms = generation_time, 
            total_time_ms = total_time 
        )
    
    def _generate_answer_without_llm(
        self, 
        query: str, 
        context_chunks: List[str]
    ) -> str: 

        if not context_chunks: 
            return "No relevant information found in the documents." 
        
        answer = f"Based on the documentation, here's what I found: \n\n" 
        answer += context_chunks[0] 

        if len(context_chunks) > 1: 
            answer += f"\n\n---\n\nAdditional relevant information:\n\n" 
            answer += context_chunks[1][:300] + "..."
        
        return answer 
    
    def _generate_answer_with_llm(
        self, 
        query: str,
        context_chunks: List[str]  
    ) -> str: 
        if not context_chunks: 
            return "No relevant info available, retrieving answer from training data" 
        
        context = "\n\n---\n\n".join(context_chunks) 

        sys_prompt =  """
        You are a helpful assistant that answers questions about engineering documentation.
        Use the provided context to answer the question accurately and concisely.
        If the context doesn't contain enough information, say so.
        Always cite which parts of the context you used.
        """

        user_prompt = f"""Context: 
        {context}
        Question: {query}

        Answer:
        """ 


        try: 
            response = self.llm_client.chat.completions.create(
                model = "gpt-3.5-turbo", 
                messages = [
                    {"role": "system", "content": sys_prompt}, 
                    {"role": "user", "content": user_prompt}
                ], 
                temperature = 0.7, 
                max_tokens = 500
            )

            return response.choices[0].message.content.strip() 
        except Exception as e: 
            print(f"Error generating answer with LLM: {e}") 
            return self._generate_answer_without_llm(query, context_chunks) 
        
    def batch_query(
        self, 
        queries: List[str],
        top_k: int = 5
    ) -> List[RAGResult]: 
        results = [] 

        for query in queries: 
            result = self.query(query, top_k = top_k) 
            results.append(result) 
        
        return results 
    
    def performance(
        self, 
        results: List[RAGResult]
    ) -> Dict: 
        if not results: 
            return {} 
        
        retrieval_times = [r.retrieval_time_ms for r in results] 
        generation_times = [r.generation_time_ms for r in results] 
        total_times = [r.total_time_ms for r in results] 

        return { 
            'num_queries': len(results),
            'avg_retrieval_time_ms': sum(retrieval_times)/len(retrieval_times),
            'avg_generation_time_ms': sum(generation_times)/len(generation_times),
            'avg_total_time_ms': sum(total_times)/len(total_times),
            'max_total_time_ms': max(total_times),
            'min_total_time_ms': min(total_times)
        }
    
if __name__ == "__main__": 
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / 'ingestion'))
    from vector_store import VectorStore 
    from embedder import Embedder 
    