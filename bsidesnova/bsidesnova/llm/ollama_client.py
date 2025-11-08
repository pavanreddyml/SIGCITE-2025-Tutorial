from typing import Dict, Iterable, Optional
import ollama


class OllamaClient:
    def __init__(
        self,
        model: str = "gemma:2b",
        host: str = "http://localhost",
        port: int = 11434,
    ) -> None:
        self.model = model
        self.base_url = f"{host}:{port}" if host.startswith("http") else f"http://{host}:{port}"
        self.client = ollama.Client(host=self.base_url)

    def set_model(self, model: str) -> None:
        self.model = model

    def get_models(self):
        return self.client.list()

    def _enforce_options(self, options: Optional[Dict]) -> Dict:
        # Always set temperature to 1.0 and attempt to disable caching where supported.
        enforced = dict(options or {})
        enforced["temperature"] = 1.0
        # Some Ollama builds may ignore unknown options; safe no-ops if unsupported:
        enforced["cache"] = False          # best-effort: disable prompt/result cache if available
        enforced["cache_prompt"] = False   # best-effort: llama.cpp-style prompt cache flag
        return enforced

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        options: Optional[Dict] = None,
    ) -> str:
        resp = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            template=template,
            options=self._enforce_options(options),
            keep_alive=0,  # do not keep the model/warm cache in memory
        )
        return resp.get("response", "")

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        options: Optional[Dict] = None,
    ) -> Iterable[str]:
        stream = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            template=template,
            options=self._enforce_options(options),
            stream=True,
            keep_alive=0,  # do not keep the model/warm cache in memory
        )
        for part in stream:
            chunk = part.get("response")
            if chunk:
                yield chunk
