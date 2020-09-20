from abc import ABC, abstractmethod
from biomed.services_getter import ServiceGetter
from numpy import array as Array

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True

class Measurer( ABC ):
    @abstractmethod
    def measureClassWeights( self, Classes: Array, Actual: Array ) -> Array:
        pass

class MeasurerFactory:
    @abstractstatic
    def getInstance( getService: ServiceGetter ) -> Measurer:
        pass
