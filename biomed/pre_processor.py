from abc import ABC, abstractmethod


class PreProcessor(ABC):
    @abstractmethod
    def preprocess_text_corpus(self, text: str) -> str:
        pass
