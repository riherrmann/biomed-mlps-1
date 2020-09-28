import unittest
from unittest.mock import MagicMock, patch
from biomed.pipeline import Pipeline
from biomed.properties_manager import PropertiesManager
from biomed.text_mining.text_mining_controller import TextminingController

class PipelineSpec( unittest.TestCase ):
    def test_it_is_a_pipeline( self ):
        self.assertTrue( isinstance( Pipeline.Factory.getInstance(), Pipeline ) )

    @patch( 'biomed.pipeline.Services' )
    def test_it_starts_the_service_locator( self, Services: MagicMock ):
        Pipe = Pipeline.Factory.getInstance()
        Pipe.pipe( MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock() )

        Services.startServices.assert_called_once()

    @patch( 'biomed.pipeline.Services' )
    def test_it_reassings_properties( self, Services: MagicMock ):
        PM = PropertiesManager()
        PM.classifier = 'is_cancer'

        Services.getService.side_effect = lambda Key, __ : PM if Key == 'properties' else MagicMock()

        Pipe = Pipeline.Factory.getInstance()
        Pipe.pipe( MagicMock(), MagicMock(), MagicMock(), MagicMock(), { 'classifier': 'doid' } )

        self.assertEqual(
            'doid',
            PM.classifier
        )

    @patch( 'biomed.pipeline.Services' )
    def test_it_kicks_off_the_test_run( self, Services: MagicMock ):
        TMC = MagicMock( spec = TextminingController )
        Short = MagicMock()
        Description = MagicMock()
        Data = MagicMock()

        Services.getService.side_effect = lambda Key, __ : TMC if Key == 'test.textminer' else MagicMock()

        Pipe = Pipeline.Factory.getInstance()
        Pipe.pipe( Data, None, Short, Description, None  )

        TMC.process.assert_called_once_with( Data, None, Short, Description )

    @patch( 'biomed.pipeline.Services' )
    def test_it_kicks_off_the_test_run_and_delegates_test_data( self, Services: MagicMock ):
        TMC = MagicMock( spec = TextminingController )
        Short = MagicMock()
        Description = MagicMock()
        Data = MagicMock()
        Test = MagicMock()

        Services.getService.side_effect = lambda Key, __ : TMC if Key == 'test.textminer' else MagicMock()

        Pipe = Pipeline.Factory.getInstance()
        Pipe.pipe( Data, Test, Short, Description, None  )

        TMC.process.assert_called_once_with( Data, Test, Short, Description )
