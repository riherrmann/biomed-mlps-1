import unittest
from unittest.mock import MagicMock, patch
from biomed.mlp.mlp_manager import MLPManager
from biomed.mlp.mlp import MLP
from biomed.properties_manager import PropertiesManager

class MLPManagerSpec( unittest.TestCase ):
    def setUp( self ):
        self.__SP = patch( 'biomed.mlp.mlp_manager.SimpleFFN', spec = MLP )
        self.__SXP = patch( 'biomed.mlp.mlp_manager.SimpleExtendedFFN', spec = MLP )
        self.__SBP = patch( 'biomed.mlp.mlp_manager.SimpleBFFN', spec = MLP )
        self.__SBXP = patch( 'biomed.mlp.mlp_manager.SimpleBExtendedFFN', spec = MLP )
        self.__SCP = patch( 'biomed.mlp.mlp_manager.SimpleCFFN', spec = MLP )
        self.__SCXP = patch( 'biomed.mlp.mlp_manager.SimpleCExtendedFFN', spec = MLP )
        self.__MSP = patch( 'biomed.mlp.mlp_manager.MultiSimpleFFN', spec = MLP )
        self.__MSBP = patch( 'biomed.mlp.mlp_manager.MultiSimpleBFFN', spec = MLP )
        self.__CP = patch( 'biomed.mlp.mlp_manager.ComplexFFN', spec = MLP )

        self.__S = self.__SP.start()
        self.__SX = self.__SXP.start()
        self.__SB = self.__SBP.start()
        self.__SBX = self.__SBXP.start()
        self.__SC = self.__SCP.start()
        self.__SCX = self.__SCXP.start()
        self.__MS = self.__MSP.start()
        self.__MSB = self.__MSBP.start()
        self.__C = self.__CP.start()

        self.__ReferenceModel = MagicMock( spec = MLP )
        self.__S.return_value = self.__ReferenceModel

    def tearDown( self ):
        self.__SP.stop()
        self.__SXP.stop()
        self.__SBP.stop()
        self.__SBXP.stop()
        self.__SCP.stop()
        self.__SCXP.stop()
        self.__MSP.stop()
        self.__MSBP.stop()
        self.__CP.stop()

    def fakeLocator( self, _, __ ):
        PM = PropertiesManager()
        PM.model = "s"
        return PM

    @patch( 'biomed.mlp.mlp_manager.Services.getService' )
    def test_it_is_a_mlp_instance( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.fakeLocator
        self.assertTrue( isinstance( MLPManager.Factory.getInstance(), MLP ) )

    @patch( 'biomed.mlp.mlp_manager.Services.getService' )
    def test_it_initializes_a_models( self, ServiceGetter: MagicMock  ):
        Models = {
            "s": self.__S,
            "sx": self.__SX,
            "sb": self.__SB,
            "sbx": self.__SBX,
            "sc": self.__SC,
            "scx": self.__SCX,
            "ms": self.__MS,
            "msb": self.__MSB,
            "c": self.__C,
        }

        for ModelKey in Models:
            pm = PropertiesManager()
            pm.model = ModelKey

            def fakeLocator( _, __ ):
                return pm

            ServiceGetter.side_effect = fakeLocator

            MLPManager.Factory.getInstance()
            Models[ ModelKey ].assert_called_once_with( pm )

    @patch( 'biomed.mlp.mlp_manager.Services.getService' )
    def test_it_returns_the_summary_of_the_builded_model( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.fakeLocator
        Expected = "summary"

        self.__ReferenceModel.buildModel.return_value = Expected

        Model = MLPManager.Factory.getInstance()

        self.assertEqual(
            Expected,
            Model.buildModel( MagicMock(), MagicMock() )
        )

    @patch( 'biomed.mlp.mlp_manager.Services.getService' )
    def test_it_returns_the_history_of_the_training( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.fakeLocator
        Expected = "this should be not a string in real"

        self.__ReferenceModel.train.return_value = Expected

        Model = MLPManager.Factory.getInstance()

        self.assertEqual(
            Expected,
            Model.train( MagicMock(), MagicMock() )
        )

    @patch( 'biomed.mlp.mlp_manager.Services.getService' )
    def test_it_returns_the_score_of_the_training( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.fakeLocator
        Expected = "this should be not a string in real"

        self.__ReferenceModel.getTrainingScore.return_value = Expected

        Model = MLPManager.Factory.getInstance()

        self.assertEqual(
            Expected,
            Model.getTrainingScore( MagicMock(), MagicMock() )
        )

    @patch( 'biomed.mlp.mlp_manager.Services.getService' )
    def test_it_returns_the_predictions( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.fakeLocator
        Expected = "this should be not a string in real"

        self.__ReferenceModel.predict.return_value = Expected

        Model = MLPManager.Factory.getInstance()

        self.assertEqual(
            Expected,
            Model.predict( MagicMock() )
        )
