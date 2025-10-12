import os 
import pathlib import Path 
from typing import list, Dict, Optional 
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

    def __init__(self, data_dir: str = "../../data/raw"): 
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
            print(f"")