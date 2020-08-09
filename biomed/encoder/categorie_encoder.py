from abc import ABC, abstractmethod
from pandas import Series
from numpy import array as Array

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True

class CategoriesEncoder( ABC ):
    @abstractmethod
    def setCategories( self, Categories: Series ):
        pass

    @abstractmethod
    def getCategories( self ) -> list:
        pass

    @abstractmethod
    def amountOfCategories( self ):
        pass

    @abstractmethod
    def encode( self, ToEncode: Array ) -> Array:
        pass

    @abstractstatic
    def hotEncode( self, ToEncode: Array ) -> Array:
        pass

    @abstractmethod
    def decode( self, ToDecode: Array ) -> Array:
        pass

class CategoriesEncoderFactory:
    @abstractstatic
    def getInstance() -> CategoriesEncoder:
        pass
