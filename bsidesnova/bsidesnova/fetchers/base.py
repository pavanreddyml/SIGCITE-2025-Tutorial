from abc import ABC, abstractmethod


class Fetcher(ABC):
    @abstractmethod
    def fetch(self):
        pass