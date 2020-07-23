from abc import ABC, abstractmethod
from numpy import array as Array
from pandas import Series

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class Vectorizer( ABC ):
    @abstractmethod
    def featureizeTrain( Train: Series, Labels: Series ) -> Array:
        pass

    @abstractmethod
    def featureizeTest( Test: Series ) -> Array:
        pass

class VectorizerFactory( ABC ):
    @abstractstatic
    def getInstance() -> Vectorizer:
        pass
