import unittest
from unittest.mock import MagicMock, patch
from biomed.mlp.model_base import ModelBase
from biomed.properties_manager import PropertiesManager
from biomed.mlp.input_data import InputData
from keras.models import Sequential
import numpy as NP
from numpy.testing import assert_array_equal as arrayEqual

class ModelBaseSpec( unittest.TestCase ):
    class StubbedFFN( ModelBase ):
        def __init__(
                self,
                Properties: PropertiesManager,
                Model,
                Weights = None,
                CustomObj = None,
        ):
            super( ModelBaseSpec.StubbedFFN, self ).__init__( Properties )
            self._Model = Model
            self._Weights = Weights
            self._CustomObjects = CustomObj

        def buildModel( self, Weights ):
            pass

    def setUp( self ):
        self.__StopperP = patch( 'biomed.mlp.model_base.Stopper' )
        self.__CheckpointP = patch( 'biomed.mlp.model_base.Checkpoint' )
        self.__LoaderP = patch( 'biomed.mlp.model_base.loadModel' )
        self.__uuidP = patch( 'biomed.mlp.model_base.uuid' )
        self.__PathP = patch( 'biomed.mlp.model_base.Path' )
        self.__RemoverP = patch( 'biomed.mlp.model_base.removeFile' )

        self.__Stopper = self.__StopperP.start()
        self.__Stopper.return_value = self.__Stopper
        self.__Checkpoint = self.__CheckpointP.start()
        self.__Checkpoint.return_value = self.__Checkpoint
        self.__Loader = self.__LoaderP.start()
        self.__Loader.return_value = MagicMock()
        self.__uuid = self.__uuidP.start()
        self.__uuid.return_value = '123'
        self.__Path = self.__PathP.start()
        self.__Path.join.return_value = 'abc'
        self.__Remover = self.__RemoverP.start()

    def tearDown( self ):
        self.__StopperP.stop()
        self.__CheckpointP.stop()
        self.__LoaderP.stop()
        self.__uuidP.stop()
        self.__PathP.stop()
        self.__RemoverP.stop()

    def test_it_builds_a_path_for_the_saved_model( self ):
        DirName = 'mydir'
        self.__Path.dirname.return_value = DirName

        UUID = '12-af-23-42'
        self.__uuid.return_value = UUID

        ModelBaseSpec.StubbedFFN( MagicMock(), MagicMock() )

        self.__Path.join.assert_called_once_with(
            DirName,
            '..',
            '..',
            '.cache',
            '12_af_23_42.h5',
        )

    def test_it_initlializes_early_stopping_callback( self ):
        PM = PropertiesManager()
        PM.training[ 'patience' ] = 50

        FFN = ModelBaseSpec.StubbedFFN( PM, MagicMock( spec = Sequential ) )
        FFN.train( MagicMock(), MagicMock() )

        self.__Stopper.assert_called_once_with(
            monitor = 'val_loss',
            mode = 'min',
            verbose = 1,
            patience = PM.training[ 'patience' ]
        )

    def test_it_initlializes_the_checkpoint_callback( self ):
        FileName = 'mbc'
        self.__Path.join.return_value = FileName

        FFN = ModelBaseSpec.StubbedFFN( PropertiesManager(), MagicMock( spec = Sequential ) )
        FFN.train( MagicMock(), MagicMock() )

        self.__Checkpoint.assert_called_once_with(
            FileName,
            monitor = 'val_accuracy',
            mode = 'max',
            verbose = 1,
            save_best_only = True
        )

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
            x = X.Training,
            y = Y.Training,
            class_weight = None,
            shuffle = True,
            epochs = PM[ "training" ][ "epochs" ],
            batch_size = PM[ "training" ][ "batch_size" ],
            validation_data = ( X.Validation, Y.Validation ),
            workers = PM[ "training" ][ "workers" ],
            use_multiprocessing = False,
            callbacks = [ self.__Stopper, self.__Checkpoint ]
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
            x = X.Training,
            y = Y.Training,
            class_weight = None,
            shuffle = True,
            epochs = PM[ "training" ][ "epochs" ],
            batch_size = PM[ "training" ][ "batch_size" ],
            validation_data = ( X.Validation, Y.Validation ),
            workers = PM[ "training" ][ "workers" ],
            use_multiprocessing = True,
            callbacks = [ self.__Stopper, self.__Checkpoint ]
        )

    def test_it_trains_a_model_with_class_weights_if_weights_are_given( self ):
        Model = MagicMock( spec = Sequential )
        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( MagicMock(), MagicMock(), MagicMock() )
        Weights = MagicMock()

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 2

        FFN = ModelBaseSpec.StubbedFFN( PM, Model, Weights )
        FFN.train( X, Y )

        Model.fit.assert_called_once_with(
            x = X.Training,
            y = Y.Training,
            class_weight = Weights,
            shuffle = True,
            epochs = PM[ "training" ][ "epochs" ],
            batch_size = PM[ "training" ][ "batch_size" ],
            validation_data = ( X.Validation, Y.Validation ),
            workers = PM[ "training" ][ "workers" ],
            use_multiprocessing = True,
            callbacks = [ self.__Stopper, self.__Checkpoint ]
        )

    def test_it_returns_the_training_history( self ):
        Hist = MagicMock()
        Hist.history = Hist
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

    def test_it_loads_the_best_model( self ):
        PM = PropertiesManager()
        PM.training[ 'patience' ] = 50

        FileName = 'kjf'
        self.__Path.join.return_value = FileName

        Best = MagicMock()
        self.__Loader.return_value = Best

        FFN = ModelBaseSpec.StubbedFFN( PM, MagicMock( spec = Sequential ) )
        FFN.train( MagicMock(), MagicMock() )

        self.__Loader.assert_called_once_with( FileName, custom_objects = None )

        self.assertEqual(
            Best,
            FFN._Model
        )

    def test_it_loads_the_best_model_with_custom_objects( self ):
        PM = PropertiesManager()
        Objs = MagicMock()
        PM.training[ 'patience' ] = 50

        FileName = 'kjf'
        self.__Path.join.return_value = FileName

        Best = MagicMock()
        self.__Loader.return_value = Best

        FFN = ModelBaseSpec.StubbedFFN( PM, MagicMock( spec = Sequential ), CustomObj = Objs )
        FFN.train( MagicMock(), MagicMock() )

        self.__Loader.assert_called_once_with( FileName, custom_objects = Objs )

        self.assertEqual(
            Best,
            FFN._Model
        )

    def test_it_removes_the_saved_model_if_it_is_there( self ):
        FileName = 'op'
        self.__Path.exists.return_value = True
        self.__Path.join.return_value = FileName

        FFN = ModelBaseSpec.StubbedFFN( PropertiesManager(), MagicMock( spec = Sequential ) )
        FFN.train( MagicMock(), MagicMock() )

        self.__Remover.assert_called_once_with( FileName )

    def test_it_does_nothing_the_saved_model_if_it_is_not_there( self ):
        self.__Path.exists.return_value = False

        FFN = ModelBaseSpec.StubbedFFN( PropertiesManager(), MagicMock( spec = Sequential ) )
        FFN.train( MagicMock(), MagicMock() )

        self.__Remover.assert_not_called()

    def test_it_fails_if_the_model_was_not_trained_while_evaluating( self ):
        Model = MagicMock( spec = Sequential )

        FFN = ModelBaseSpec.StubbedFFN( PropertiesManager(), Model )
        with self.assertRaises( RuntimeError, msg = "The model has not be trained" ):
            FFN.getTrainingScore( MagicMock(), MagicMock() )

    def test_it_gets_the_training_evaluation( self ):
        Model = MagicMock( spec = Sequential )
        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 3 ) ), MagicMock(), MagicMock() )

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 1

        self.__Loader.return_value = Model

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )
        FFN.getTrainingScore( X, Y )

        Model.evaluate.assert_called_once_with(
            X.Test,
            Y.Test,
            batch_size = PM[ "training" ][ "batch_size" ],
            workers = PM[ "training" ][ "workers" ],
            use_multiprocessing = False,
            return_dict = True,
            verbose = 0
        )

    def test_it_gets_the_training_evaluation_for_multi_processing( self ):
        Model = MagicMock( spec = Sequential )
        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 3 ) ), MagicMock(), MagicMock() )

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 3

        self.__Loader.return_value = Model

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )
        FFN.getTrainingScore( X, Y )

        Model.evaluate.assert_called_once_with(
            X.Test,
            Y.Test,
            batch_size = PM[ "training" ][ "batch_size" ],
            workers = PM[ "training" ][ "workers" ],
            use_multiprocessing = True,
            return_dict = True,
            verbose = 0
        )

    def test_it_retruns_the_evaluation_score( self ):
        Eval = MagicMock()
        Model = MagicMock( spec = Sequential )
        Model.evaluate.return_value = Eval
        self.__Loader.return_value = Model

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
        Model.predict.return_value = NP.array( [ [ 0.0, 0.0 ] ] )

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 1

        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 3 ) ), MagicMock(), MagicMock() )

        self.__Loader.return_value = Model

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
        Model.predict.return_value = NP.array( [ [ 0., 0. ] ] )

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 2

        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 2, 3 ) ), MagicMock(), MagicMock() )

        self.__Loader.return_value = Model

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

        Model.predict.return_value = NP.array(
            [
                [ 0.0, 0.00 ],
                [ 0.0, 0.98867947 ],
                [ 0.0, 0.00 ]
             ]
        )

        self.__Loader.return_value = Model


        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )
        arrayEqual(
            FFN.predict( ToPredict ),
            NP.array( [ 0, 1, 0 ] )
        )

    def test_it_returns_normalized_multi_classified_data( self ):
        Model = MagicMock( spec = Sequential )
        ToPredict = MagicMock()

        PM = PropertiesManager()
        PM[ "training" ][ "epochs" ] = 1
        PM[ "training" ][ "batch_size" ] = 2
        PM[ "training" ][ "workers" ] = 1
        PM.classifier = 'doid'

        X = InputData( MagicMock(), MagicMock(), MagicMock() )
        Y = InputData( NP.zeros( ( 4, 4 ) ), MagicMock(), MagicMock() )

        Model.predict.return_value = NP.array( [ [ -0.00716622, 23 ], [ -23, -98.98867947 ], [ -42, 12 ] ] )

        self.__Loader.return_value = Model

        FFN = ModelBaseSpec.StubbedFFN( PM, Model )
        FFN.train( X, Y )
        arrayEqual(
            FFN.predict( ToPredict ),
            NP.array( [ 1, 0, 1 ] )
        )
