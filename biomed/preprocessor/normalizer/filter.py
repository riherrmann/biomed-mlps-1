from abc import ABC, abstractmethod

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Filter( ABC ):
    @abstractmethod
    def apply( self, Text: str ) -> str:
        pass

class FilterFactory( ABC ):
    @abstractstatic
    def getInstance() -> Filter:
        pass
