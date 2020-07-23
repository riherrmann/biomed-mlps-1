import unittest
from unittest.mock import MagicMock
from biomed.mlp.model_base import ModelBase
from biomed.properties_manager import PropertiesManager
from biomed.mlp.input_data import InputData
from keras.models import Sequential
import numpy as NP
from numpy.testing import assert_array_equal as arrayEqual

class ModelBaseSpec( unittest.TestCase ):
    class StubbedFFN( ModelBase ):
        def __init__( self, Properties: PropertiesManager, Model ):
            super( ModelBaseSpec.StubbedFFN, self ).__init__( Properties )
            self._Model = Model

        def buildModel( self ):
            pass

    def test_it_trains_a_model( self ):
        Model = MagicMock( spec = Sequential )
        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( MagicMock(), MagicMock(), MagicMock() )

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 1

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )

        Model.fit.assert_called_once_with(
            x = X.Train,
            y = Y.Train,
            shuffle = True,
            epochs = PM[ "training" ][ "epochs" ],
            batch_size = PM[ "training" ][ "batch_size" ],
            validation_data = ( X.Val, Y.Val ),
            workers = PM[ "training" ][ "workers" ],
            use_multiprocessing = False
        )

    def test_it_trains_a_model_multiprocessing_if_more_then_one_worker_is_available( self ):
        Model = MagicMock( spec = Sequential )
        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( MagicMock(), MagicMock(), MagicMock() )

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 2

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )

        Model.fit.assert_called_once_with(
            x = X.Train,
            y = Y.Train,
            shuffle = True,
            epochs = PM[ "training" ][ "epochs" ],
            batch_size = PM[ "training" ][ "batch_size" ],
            validation_data = ( X.Val, Y.Val ),
            workers = PM[ "training" ][ "workers" ],
            use_multiprocessing = True
        )

    def test_it_returns_the_training_history( self ):
        Hist = MagicMock()
        Model = MagicMock( spec = Sequential )
        Model.fit.return_value = Hist

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 2

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        self.assertEqual(
            FFN.train( MagicMock(), MagicMock() ),
            Hist
        )

    def test_it_fails_if_the_model_was_not_trained_while_evaluating( self ):
        Model = MagicMock( spec = Sequential )

        FFN = ModelBaseSpec.StubbedFFN( PropertiesManager(), Model )
        with self.assertRaises( RuntimeError, msg = "The model has not be trained" ):
            FFN.getTrainingScore( MagicMock(), MagicMock() )

    def test_it_gets_the_training_evaluation( self ):
        Model = MagicMock( spec = Sequential )
        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 3 ) ), MagicMock(), MagicMock() )

        FFN = ModelBaseSpec.StubbedFFN( PropertiesManager(), Model )
        FFN.train( X, Y )
        FFN.getTrainingScore( X, Y )

        Model.evaluate.assert_called_once_with(
            X.Test,
            Y.Test,
            verbose = 0
        )

    def test_it_retruns_the_evaluation_score( self ):
        Eval = MagicMock()
        Model = MagicMock( spec = Sequential )
        Model.evaluate.return_value = Eval

        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 3 ) ), MagicMock(), MagicMock() )

        FFN = ModelBaseSpec.StubbedFFN( PropertiesManager(), Model )
        FFN.train( X, Y )
        self.assertEqual(
            FFN.getTrainingScore( X, Y ),
            Eval
        )

    def test_it_fails_if_the_model_was_not_trained_while_predicting( self ):
        Model = MagicMock( spec = Sequential )
        ToPredict = MagicMock()

        FFN = ModelBaseSpec.StubbedFFN( PropertiesManager(), Model )
        with self.assertRaises( RuntimeError, msg = "The model has not be trained" ):
            FFN.predict( ToPredict )

    def test_it_predicts( self ):
        Model = MagicMock( spec = Sequential )
        ToPredict = MagicMock()

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 1


        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 3 ) ), MagicMock(), MagicMock() )

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )
        FFN.predict( ToPredict )

        Model.predict.assert_called_once_with(
            ToPredict,
            batch_size = PM.training['batch_size'],
            workers = PM.training[ "workers" ],
            use_multiprocessing = False
        )

    def test_it_predicts_with_mulitprocessing( self ):
        Model = MagicMock( spec = Sequential )
        ToPredict = MagicMock()

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 2

        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 3 ) ), MagicMock(), MagicMock() )

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )
        FFN.predict( ToPredict )

        Model.predict.assert_called_once_with(
            ToPredict,
            batch_size = PM.training['batch_size'],
            workers = PM.training[ "workers" ],
            use_multiprocessing = True
        )

    def test_it_returns_normalized_binary_classified_data( self ):
        Model = MagicMock( spec = Sequential )
        ToPredict = MagicMock()

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 1

        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 2 ) ), MagicMock(), MagicMock() )

        Model.predict.return_value = NP.array( [ [0.00716622], [0.98867947], [0.01186692] ] )

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )
        arrayEqual(
            FFN.predict( ToPredict ),
            NP.array( [ [0], [1], [0] ] )
        )

    def test_it_returns_normalized_multi_classified_data( self ):
        Model = MagicMock( spec = Sequential )
        ToPredict = MagicMock()

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 1

        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 4, 4 ) ), MagicMock(), MagicMock() )

        Model.predict.return_value = NP.array( [ [ -0.00716622, 23 ], [ -23, -98.98867947 ], [ -42, 12 ] ] )

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )
        arrayEqual(
            FFN.predict( ToPredict ),
            NP.array( [ 1, 0, 1 ] )
        )
