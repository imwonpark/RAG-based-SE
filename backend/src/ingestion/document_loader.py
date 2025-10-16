import os 
from pathlib import Path 
from typing import List, Dict, Optional 
from dataclasses import dataclass 
from datetime import datetime 


@dataclass
class Document: 
    content: str 
    metadata: Dict[str, any] 
    source: str 

    def __repr__(self): 
        return f"Document(source={self.source}, length={len(self.content)})"


class DocumentLoader: 
    SUPPORTED_EXTENSIONS = {'.md', '.txt', '.pdf', '.docx', '.doc'}

    def __init__(self, data_dir: Optional[str] = None): 
        if data_dir is None:
            script_dir = Path(__file__).parent
            self.data_dir = script_dir.parent.parent / "data" / "raw"
        else:
            self.data_dir = Path(data_dir) 

    def load_file(self, filepath:str) -> Optional[Document]: 
        path = Path(filepath) 

        if not path.exists(): 
            print(f"File not found: {path}") 
            return None 
        
        extension = path.suffix.lower() 

        if extension not in self.SUPPORTED_EXTENSIONS: 
            print(f"Unsupported file extension: {extension}") 
            return None 
        
        try: 
            if extension == ".md" or extension == '.txt': 
                return self._load_text_file(path) 
            elif extension == '.pdf': 
                return self._load_pdf_file(path) 
        except Exception as e: 
            print(f"Error loading{filepath}: {e}") 
            return None 
    
    def _load_text_file(self, path: Path) -> Document: 
            with open(path, 'r', encoding='utf-8') as f: 
                content = f.read() 
            
            title = path.stem 
            lines = content.split('\n') 
            if lines and lines[0].startswith('# '):
                title = lines[0].replace('#', '').strip() 

            metadata = {
                'title': title, 
                'file_type': path.suffix,
                'file_size': path.stat().st_size, 
                'created_at': datetime.fromtimestamp(path.stat().st_ctime),
                'modified_at': datetime.fromtimestamp(path.stat().st_mtime)
            } 

            return Document(content=content, metadata=metadata, source=str(path)) 
        
    def _load_pdf_file(self, path: Path) -> Document: 
        try:
            from pypdf import PdfReader 
        except ImportError: 
            print("pypdf not installed") 
            return None 
        
        reader = PdfReader(path) 
        content = "" 

        for page in reader.pages:
            content += page.extract_text() + "\n" 
        
        metadata = {
            'title': path.stem, 
            'file_type': 'pdf',
            'page_count': len(reader.pages), 
            'file_size': path.stat().st_size, 
            'created_at': datetime.fromtimestamp(path.stat().st_ctime),
        }

        return Document(content=content, metadata=metadata, source=str(path)) 
    
    def load_directory(self, directory: Optional[str] = None) -> List[Document]: 
        if directory is None: 
            directory = self.data_dir 
        else: 
            directory = Path(directory) 
        
        if not directory.exists(): 
            print(f"Directory not found: {directory}") 
            return [] 
        
        documents = [] 

        for file in directory.rglob('*'): 
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_EXTENSIONS: 
                doc = self.load_file(file) 
                if doc: 
                    documents.append(doc) 
                    print(f"Loaded: {doc.metadata['title']}") 
        
        return documents 
    
    def get_stats(self, documents: List[Document]) -> Dict[str, any]: 
        if not documents: 
            return {} 

        total_chars = sum(len(doc.content) for doc in documents) 

        return { 
            'total_documents': len(documents), 
            'total_characters': total_chars, 
            'average_length': total_chars // len(documents), 
            'file_types': list(set(doc.metadata['file_type'] for doc in documents))
        }

if __name__ == "__main__": 
    loader = DocumentLoader() 

    print("Loading documents from data/raw...") 
    docs = loader.load_directory() 

    print(f"\n{'='*60}")
    print("Document Loading Summary:") 
    print(f"\n{'='*60}")

    stats = loader.get_stats(docs) 
    for k,v in stats.items():
        print(f"{k}: {v}") 

    print(f"\n{'='*60}") 
    print("Document Preview")
    print(f"\n{'='*60}") 

    if docs: 
        doc = docs[0] 
        print(f"\nTitle: {doc.metadata['title']}") 
        print(f"Source: {doc.source}") 
        print(f"Length: {len(doc.content)} characters") 
        print(f"\nFirst 300 characters:") 
        print(doc.content[:300]) 


    