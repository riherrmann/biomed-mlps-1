from keras.losses import Loss
import tensorflow as TF
import tensorflow.keras.backend as KerasAdapter
from tensorflow.math import multiply
from keras.losses import BinaryCrossentropy as Binary
from keras.losses import CategoricalCrossentropy as Categorical
from keras.losses import SparseCategoricalCrossentropy as Sparse
import numpy as Numpy
from typing import Union
from tensorflow.python.keras.utils import losses_utils as Utils

"""
Thankfully build on top of https://github.com/keras-team/keras/issues/2115#issuecomment-530762739
"""
class WeightedCrossentropy( Loss ):
    def __init__(
        self,
        Name: str,
        ClassWeights: Union[ dict, list ],
        EntropyKey: str,
        KeywordedEntropyArgs: dict = {},
        Reduction: None = None, #should be ignored
    ):
        self.name = Name
        self.__EntropyKey = EntropyKey
        self.__OrgWeights = ClassWeights
        self.__KeywordedEntropyArgs = KeywordedEntropyArgs
        self.reduction = Utils.ReductionV2.AUTO

        self.__LossFunction = self.__initLossFunction( EntropyKey, KeywordedEntropyArgs )
        self.setCostMatrix( self.__remapOrConvert( ClassWeights ) )

    def __initLossFunction( self, Key: str, Arguments ) -> Loss:
        Entropies = {
            'bin': Binary,
            'categorical': Categorical,
            'sparse': Sparse,
        }

        return Entropies[ self.__EntropyKey ]( **Arguments )

    def __remapOrConvert( self, Weights: Union[ dict, list ] ) -> Numpy.array:
        if isinstance( Weights, list ):
            return Numpy.array( Weights )
        else:
            return self.__remapCostMatrix( Weights )

    def __remapCostMatrix( self, CostMatrix: dict ) -> Numpy.array:
        Transformed = Numpy.ones( ( len( CostMatrix ), len( CostMatrix ) ) )
        for Index, Weight in CostMatrix.items():
            Transformed[ 0 ][ Index ] = Weight
            Transformed[ Index ][ 0 ] = Weight

        return Transformed

    def setCostMatrix( self, Matrix: Numpy.array ) -> None:
        self.__AmountOfClasses = len( Matrix )
        self.__NormalizedCosts = KerasAdapter.cast(
            KerasAdapter.expand_dims( Matrix, 0 ),
            KerasAdapter.floatx()
        )

        self.__OrgWeights = Matrix.tolist()

    def __validateRank( self, Labels: TF, Predictions: TF ) -> None:
        try:
            # This should be done in a more abstract way
            Labels.shape.assert_has_rank(2)
        except:
            raise NotImplementedError( 'Dealing with other ranks then 2 is not implemented yet' )

        try:
            Labels.shape.assert_is_compatible_with( Predictions.shape )
        except:
            raise TypeError( "Labels and predictions have different ranks" )

    def __validateCompatibility( self, Predictions: TF ) -> None:
        try:
            Predictions.shape[1:].assert_is_compatible_with( self.__AmountOfClasses )
        except:
            raise TypeError( "Predictions and given weights are incompatible" )

    def __hotEncode( self, Predictions: TF ) -> TF:
        return KerasAdapter.one_hot(
            KerasAdapter.argmax( Predictions ),
            self.__AmountOfClasses
        )

    def __normalizeAxises( self, Labels: TF, Predictions: TF ) -> tuple:
        return (
            KerasAdapter.cast( KerasAdapter.expand_dims( Labels, 2 ), KerasAdapter.floatx() ),
            KerasAdapter.cast( KerasAdapter.expand_dims( Predictions, 1 ), KerasAdapter.floatx() )
        )

    def __buildProducts( self, Labels: TF, Predictions: TF ) -> TF:
        return multiply(
            self.__NormalizedCosts,
            multiply(
                Labels,
                Predictions,
            )
        )

    def __computeWeights( self, Labels: TF, Predictions: TF ) -> TF:
        Labels, Predictions = self.__normalizeAxises( Labels, Predictions )
        return KerasAdapter.sum(
            self.__buildProducts( Labels, Predictions ),
            axis = [ 1, 2 ]
        )

    def __call__( self, y_true: TF, y_pred: TF, sample_weight: None = None) -> TF:
        return self.call( y_true, y_pred )

    def call( self, y_true: TF, y_pred: TF ):
        self.__validateRank( y_true, y_pred )
        self.__validateCompatibility( y_pred )
        return self.__LossFunction(
            y_true,
            y_pred,
            self.__computeWeights(
                y_true,
                self.__hotEncode( y_pred )
            )
        )

    def get_config( self ) -> dict:
        return {
            'Name': self.name,
            'ClassWeights': self.__OrgWeights,
            'EntropyKey': self.__EntropyKey,
            'Reduction': None,
            'KeywordedEntropyArgs': self.__KeywordedEntropyArgs,
        }
