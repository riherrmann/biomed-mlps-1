from abc import ABC, abstractmethod
from numpy import array as Array
from pandas import Series
from typing import Union
from biomed.services_getter import ServiceGetter

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class Vectorizer( ABC ):
    @abstractmethod
    def featureizeTrain(
        self,
        Train: Series,
        Labels: Series,
        Weights: Union[ None, dict ]
    ) -> Array:
        pass

    @abstractmethod
    def featureizeTest( self, Test: Series ) -> Array:
        pass

    @abstractmethod
    def getSupportedFeatures( self ) -> list:
        pass

class VectorizerFactory( ABC ):
    @abstractstatic
    def getInstance( getService: ServiceGetter ) -> Vectorizer:
        pass
