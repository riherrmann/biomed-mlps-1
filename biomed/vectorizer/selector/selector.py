from abc import ABC, abstractmethod
from numpy import array as Array
from pandas import Series
from biomed.services_getter import ServiceGetter

class abstractstatic( staticmethod ):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True

class Selector( ABC ):
    @abstractmethod
    def build( self, X: Array, Labels: Series ):
        pass

    @abstractmethod
    def select( self, X: Array ) -> Array:
        pass

    @abstractmethod
    def getSupportedFeatures( self, Labels: list ) -> list:
        pass


class SelectorFactory( ABC ):
    def __init__( self, getService: ServiceGetter ):
        self._getService = getService

    @abstractmethod
    def getInstance( self ) -> Selector:
        pass
