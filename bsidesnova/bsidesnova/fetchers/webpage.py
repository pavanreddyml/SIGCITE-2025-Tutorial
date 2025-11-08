import requests

from .base import Fetcher

class WebpageFetcher():
    def __init__(self, server_url="http://localhost:2500/fetch-webpage"):
        self.server_url = server_url

    def fetch(self, url: str) -> str:
        try:
            response = requests.get(self.server_url, params={"url": url}, timeout=5)
            response.raise_for_status()
            content = response.json().get("content", "")
        except Exception as e:
            return f"Error fetching webpage: {str(e)}"
        return content