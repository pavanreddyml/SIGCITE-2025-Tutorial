import fitz

from .base import Fetcher


class DocumentFetcher(Fetcher):
    def fetch(self, file_path: str) -> str:
        text = f"""
        -----------------------------------------------------------------------
        # {file_path} \n\n
        # """
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return text + file.read()
            
        if file_path.endswith('.pdf'):
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text() + '\n'
                return text
            
        