from typing import List, Dict 
from dataclasses import dataclass 
import re 
from document_loader import Document 


@dataclass 
class Chunk: 
    text: str 
    metadata: Dict 
    chunk_index: int 

    def __repr__(self): 
        return f"Chunk(index = {self.chunk_index}, length = {len(self.text)})"


class TextChunker: 

    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 50, 
        separator: str = "\n\n" 
    ): 
        """
        Args:
            chunk_size: Target size in tokens (roughly 4 chars = 1 token)
            chunk_overlap: Number of tokens to overlap between chunks
            separator: Primary separator for splitting (paragraphs by default)
        """
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap 
        self.separator = separator 

        self.chunk_size_chars = chunk_size * 4 
        self.chunk_overlap_chars = chunk_overlap * 4 

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Chunk]: 
        """
        Split text into chunks of specified size with overlap. 
        """
        if metadata is None: 
            metadata = {} 

        sections = text.split(self.separator) 

        chunks = [] 
        current_chunk = "" 
        chunk_index = 0 

        for section in sections: 
            section = section.strip() 
            if not section: 
                continue 

            if len(current_chunk) + len(section) > self.chunk_size_chars and current_chunk:
                chunks.append(self._create_chunk(current_chunk, metadata, chunk_index))
                chunk_index += 1 

                current_chunk = self._get_overlap(section, current_chunk) + section 
            else: 
                if current_chunk: 
                    current_chunk += self.separator + section 
                else: 
                    current_chunk = section 

        if current_chunk: 
            chunks.append(self._create_chunk(current_chunk, metadata, chunk_index)) 
        
        return chunks 
    
    def _create_chunk(self, text: str, metadata: Dict, index: int) -> Chunk: 
        """ Create a chunk object w metadata""" 
        chunk_metadata = {
            **metadata,
            'chunk_index': index, 
            'chunk_size': len(text)
        }

        return Chunk(
            text=text.strip(),
            metadata = chunk_metadata,
            chunk_index = index
        )

    def _get_overlap(self, text: str) -> str: 
        """ Get overlap between current chunk and next section""" 
        if len(text) <= self.chunk_overlap_chars: 
            return text 
        
        overlap = text[-self.chunk_overlap_chars:] 

        sentences = re.split(r'[.!?]\s+', overlap) 
        if len(sentences) > 1: 
            return sentences[-1]
        
        return overlap 
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]: 
        """Chunk a list of documents""" 
        chunks = [] 

        for doc in documents: 
            chunks.extend(self.chunk_text(doc.content, doc.metadata)) 

        return chunks 
    
    def get_stats(self, chunks: List[Chunk]) -> Dict[str, any]: 
        """Get stats about the chunks""" 
        if not chunks: 
            return {} 
        
        chunk_sizes = [len(chunk.text) for chunk in chunks] 

        return {
            'total_chunks': len(chunks), 
            'avg_chunk_size': sum(chunk_sizes) // len(chunks), 
            'min_chunk_size': min(chunk_sizes), 
            'max_chunk_size': max(chunk_sizes), 
            'total_characters': sum(chunk_sizes)
        }

if __name__ == "__main__": 

    from document_loader import DocumentLoader 

    print("Loading documents...") 
    loader = DocumentLoader() 
    docs = loader.load_directory() 

    print(f"Loaded {len(docs)} documents") 

    print("\nChunking documents...") 
    chunker = TextChunker(chunk_size=512, chunk_overlap=50) 
    chunks = chunker.chunk_documents(docs) 

    print(f"\n{'='*60}") 
    print("Chunking Summary:") 
    print(f"\n{'='*60}") 

    stats = chunker.get_stats(chunks) 
    for k,v in stats.items(): 
        print(f"{k}: {v}") 
        
    print(f"\n{'='*60}") 
    print("Chunk Preview:") 
    print(f"\n{'='*60}") 

    if chunks: 
        for i in range(min(3, len(chunks))): 
            chunk = chunks[i] 
            print(f"\nChunk {i+1}:") 
            print(f"From: {chunk.metadata.get('title', 'Unknown')}") 
            print(f"Size: {len(chunk.text)} characters") 
            print(f"Preview: {chunk.text[:200]}...") 

    print("\nDone!") 