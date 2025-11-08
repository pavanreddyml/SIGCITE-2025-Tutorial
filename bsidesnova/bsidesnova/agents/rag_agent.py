import markdown
import textwrap

from ..llm.ollama_client import OllamaClient

class RAGAgent:
    def __init__(self, ollama_client: OllamaClient, context: str):
        self.ollama_client = ollama_client
        self.context = context
        self.system_prompt = textwrap.dedent(f"""
        You are a helpful assistant that uses the provided document context to answer user questions.

        Instructions:
        - Use the document context to provide accurate and relevant answers to user questions.
        - If the answer is not found in the document context, respond with "I don't know" or "The document does not provide this information."
        - Provide clear and concise answers without unnecessary details. 

        Generate the response in markdown
        """)

        self.formatting_prompt = """
        You will be given a question by the user and the context from the uploaded document.
        """

    def get_response(self, user_prompt: str) -> str:
        prompt = f"Context: {self.context}\n\nUser Question: {user_prompt}\n\n"
        response = self.ollama_client.generate(prompt=prompt, system=self.system_prompt)
        response = markdown.markdown(response)
        return response