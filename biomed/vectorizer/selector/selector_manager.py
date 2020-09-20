from biomed.vectorizer.selector.selector import Selector, SelectorFactory
from biomed.vectorizer.selector.dependency_selector import DependencySelector
from biomed.vectorizer.selector.factor_selector import FactorSelector
from biomed.vectorizer.selector.regression_selector import RegressionSelector
from biomed.properties_manager import PropertiesManager
from biomed.services_getter import ServiceGetter
from pandas import Series
from typing import Union
from numpy import array as Array

class SelectorManager( Selector ):
    def __init__( self, Properties: PropertiesManager ):
        self.__Properties = Properties
        self.__Selectors = {
            "dependency": DependencySelector,
            "factor": FactorSelector,
            "regression": RegressionSelector,
        }

    def __buildSelectorModel( self, X: Array, Labels: Series, Weights: Union[ None, dict ] ):
        self.__Selector = self.__Selectors[ self.__Properties.selection[ 'type' ] ]( self.__Properties )
        self.__Selector.build( X, Labels, Weights )

    def build( self, X: Array, Labels: Series, Weights: Union[ None, dict ] ):
        if not self.__Properties.selection[ 'type' ]:
            self.__Selector = None
        else:
            self.__buildSelectorModel( X, Labels, Weights )

    def select( self, X: Array ) -> Array:
        if not self.__Selector:
            return X.toarray()
        else:
            return self.__Selector.select( X )

    def getSupportedFeatures( self, FeatureNames: list ) -> list:
        if not self.__Selector:
            return FeatureNames
        else:
            return self.__Selector.getSupportedFeatures( FeatureNames )

    class Factory( SelectorFactory ):
        @staticmethod
        def getInstance( getService: ServiceGetter ) -> Selector:
            return SelectorManager( getService( "properties", PropertiesManager ) )
