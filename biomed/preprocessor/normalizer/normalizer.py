from abc import ABC, abstractmethod

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Normalizer( ABC ):
    @abstractmethod
    def apply( self, StackOfDocuments: list, Flags: list ) -> list:
        pass

    def _reassemble( self, Text: list ) -> str:
        return " ".join( Text )

class NormalizerFactory( ABC ):
    @abstractstatic
    def getInstance() -> Normalizer:
        pass
