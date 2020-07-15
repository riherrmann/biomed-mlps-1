from abc import ABC, abstractmethod
from pandas import DataFrame
from numpy import array as Array
from biomed.properties_manager import PropertiesManager

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        ID: str,
        Configuration: PropertiesManager,
        ExpectedPreditions: DataFrame,
        Predicted: Array
    ):
        pass

class EvaluatorFactory( ABC ):
    @abstractstatic
    def getInstance() -> Evaluator:
        pass
