from abc import ABC, abstractmethod
from numpy import array as Array
from pandas import Series

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class Evaluator( ABC ):
    @abstractmethod
    def start( self, ShortName: str, Description ):
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
    def captureData( self, Train: list, Test: list ):
        pass

    @abstractmethod
    def capturePreprocessedData( self, TrainDocs: Series, TestDocs: Series ):
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
    def captureTrainingHistory( self, History: dict ):
        pass

    @abstractmethod
    def captureEvaluationScore( self, Score: dict ):
        pass

    @abstractmethod
    def capturePredictions(
        self,
        Predictions: Array,
        PMIds: list,
        Actual: list = None
    ):
        pass

    @abstractmethod
    def score(
        self,
        Predictions: Array,
        Actual: list,
        Labels: list
    ):
        pass

class EvaluatorFactory:
    @abstractstatic
    def getInstance() -> Evaluator:
        pass
