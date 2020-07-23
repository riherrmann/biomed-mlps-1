from abc import ABC, abstractmethod
from numpy import array as Array
from pandas import Series

class abstractstatic( staticmethod ):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True

class Selector( ABC ):
    @abstractmethod
    def build( X: Array, Labels: Series ):
        pass

    @abstractmethod
    def select( X: Array ) -> Array:
        pass

class SelectorFactory( ABC ):
    @abstractmethod
    def getInstance() -> Selector:
        pass
