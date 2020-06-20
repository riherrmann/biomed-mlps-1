from abc import ABC, abstractmethod

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Normalizer( ABC ):
    @abstractmethod
    def apply( self, Token: str, Flags: list ) -> list:
        pass

class NormalizerFactory( ABC ):
    @abstractstatic
    def getInstance() -> Normalizer:
        pass
