from biomed.measurer.measurer import Measurer
from biomed.measurer.measurer import MeasurerFactory
from biomed.services_getter import ServiceGetter
from biomed.properties_manager import PropertiesManager
from sklearn.utils.class_weight import compute_class_weight as weightClasses
from numpy import array as Array

class StdMeasurer( Measurer ):
    def __init__( self, Properties: PropertiesManager ):
        self.__Active = Properties.weights[ 'use_class_weights' ]

    def measureClassWeights( self, Classes: Array, Actual: Array ) -> Array:
        if self.__Active:
            return weightClasses( 'balanced', Classes, Actual )
        else:
            return None

    class Factory( MeasurerFactory ):
        @staticmethod
        def getInstance( getService: ServiceGetter ) -> Measurer:
            return StdMeasurer( getService( 'properties', PropertiesManager ) )
