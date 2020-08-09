from abc import ABC, abstractmethod

class Normalizer( ABC ):
    @abstractmethod
    def apply( self, StackOfDocuments: list, Flags: list ) -> list:
        pass

    def _reassemble( self, Text: list ) -> str:
        return " ".join( Text )

class NormalizerFactory( ABC ):
    @abstractmethod
    def getApplicableFlags() -> list:
        pass
    @abstractmethod
    def getInstance() -> Normalizer:
        pass
