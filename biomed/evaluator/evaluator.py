from abc import ABC, abstractmethod
from typing import Union
from numpy import array as Array
from pandas import Series
from biomed.services_getter import ServiceGetter

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class Evaluator( ABC ):
    @abstractmethod
    def start( self, ShortName: str, Description: str ):
        pass

    @abstractmethod
    def setFold( self, Fold ):
        pass

    @abstractmethod
    def finalize( self ) -> dict:
        pass

    @abstractmethod
    def captureStartTime( self ):
        pass

    @abstractmethod
    def capturePreprocessingTime( self ):
        pass

    @abstractmethod
    def captureVectorizingTime( self ):
        pass

    @abstractmethod
    def captureTrainingTime( self ):
        pass

    @abstractmethod
    def caputrePredictingTime( self ):
        pass

    @abstractmethod
    def captureData( self, Train: Series, Test: Series ):
        pass

    @abstractmethod
    def captureClassWeights( self, Weights: Union[ None, Series ] ):
        pass

    @abstractmethod
    def capturePreprocessedData( self, Processed: Series, Original: Series ):
        pass

    @abstractmethod
    def captureFeatures(
        self,
        TrainFeatures: tuple,
        TestFeatures: tuple,
        BagOfWords: list
    ):
        pass

    @abstractmethod
    def captureModel( self, Model: str ):
        pass

    @abstractmethod
    def captureTrainingHistory( self, History: dict ):
        pass

    @abstractmethod
    def captureEvaluationScore( self, Score: dict ):
        pass

    @abstractmethod
    def capturePredictions(
        self,
        Predictions: Array,
        PMIds: Series,
        Actual: list = None
    ):
        pass

    @abstractmethod
    def score(
        self,
        Predictions: Array,
        Actual: Series,
        Labels: Series
    ):
        pass

class EvaluatorFactory:
    @abstractstatic
    def getInstance( getService: ServiceGetter ) -> Evaluator:
        pass
