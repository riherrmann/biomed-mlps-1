import unittest
from unittest.mock import MagicMock, patch, ANY
from biomed.mlp.mlp_manager import MLPManager
from biomed.mlp.mlp import MLP
from biomed.properties_manager import PropertiesManager

class MLPManagerSpec( unittest.TestCase ):
    def setUp( self ):
        self.__B2P = patch( 'biomed.mlp.mlp_manager.Bin2Layered', spec = MLP )
        self.__WB2P = patch( 'biomed.mlp.mlp_manager.WeightedBin2Layered', spec = MLP )

        self.__B2 = self.__B2P.start()
        self.__WB2 = self.__WB2P.start()

        self.__ReferenceModel = MagicMock( spec = MLP )
        self.__B2.return_value = self.__ReferenceModel

    def tearDown( self ):
        self.__B2P.stop()
        self.__WB2P.stop()

    def __fakeLocator( self, _, __ ):
        PM = PropertiesManager()
        PM.model = "b2"
        return PM

    def test_it_is_a_mlp_instance( self ):
        self.assertTrue( isinstance( MLPManager.Factory.getInstance( self.__fakeLocator ), MLP ) )

    def test_it_initializes_a_models( self  ):
        Models = {
            "b2": self.__B2,
            "wb2": self.__WB2,
        }

        for ModelKey in Models:
            pm = PropertiesManager()
            pm.model = ModelKey

            def fakeLocator( _, __ ):
                return pm

            ServiceGetter = MagicMock()
            ServiceGetter.side_effect = fakeLocator

            MyManager = MLPManager.Factory.getInstance( ServiceGetter )
            MyManager.buildModel( 2 )

            Models[ ModelKey ].assert_called_once_with( pm )
            ServiceGetter.assert_called_once()

    def test_it_deligates_the_dimensionality_to_the_model( self ):
        InputShape = ( 2, 3 )

        MyManager = MLPManager.Factory.getInstance( self.__fakeLocator )
        MyManager.buildModel( InputShape )

        self.__ReferenceModel.buildModel.assert_called_once_with( InputShape, ANY )

    def test_it_deligates_the_training_arguments_without_weights_to_the_model_by_default( self ):
        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3 ) )

        self.__ReferenceModel.buildModel.assert_called_once_with( ANY, None )

    def test_it_deligates_given_weights_to_the_model( self ):
        Weights = MagicMock()

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3 ), Weights )

        self.__ReferenceModel.buildModel.assert_called_once_with( ANY, Weights )

    def test_it_returns_the_summary_of_the_builded_model( self ):
        Expected = "summary"

        self.__ReferenceModel.buildModel.return_value = Expected

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )

        self.assertEqual(
            Expected,
            Model.buildModel( MagicMock() )
        )

    def test_it_returns_the_history_of_the_training( self ):
        Expected = "this should be not a string in real"

        self.__ReferenceModel.train.return_value = Expected

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3 ) )

        self.assertEqual(
            Expected,
            Model.train( MagicMock(), MagicMock() )
        )

    def test_it_returns_the_score_of_the_training( self ):
        Expected = "this should be not a string in real"

        self.__ReferenceModel.getTrainingScore.return_value = Expected

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3 ) )

        self.assertEqual(
            Expected,
            Model.getTrainingScore( MagicMock(), MagicMock() )
        )

    def test_it_returns_the_predictions( self ):
        Expected = "this should be not a string in real"

        self.__ReferenceModel.predict.return_value = Expected

        Model = MLPManager.Factory.getInstance( self.__fakeLocator )
        Model.buildModel( ( 2, 3) )

        self.assertEqual(
            Expected,
            Model.predict( MagicMock() )
        )
