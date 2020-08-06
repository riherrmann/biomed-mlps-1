import unittest
from unittest.mock import MagicMock, patch
from biomed.pipeline_runner import PipelineRunner
from biomed.pipeline import Pipeline

class PipelineRunnerSpec( unittest.TestCase ):
    def test_it_is_a_Pipeline_Runner( self ):
        Runner = PipelineRunner.Factory.getInstance()
        self.assertTrue( isinstance( Runner, PipelineRunner ) )

    @patch( 'biomed.pipeline_runner.Pipeline.Factory.getInstance' )
    def test_it_initalizes_a_pipeline( self, PLF: MagicMock ):
        Runner = PipelineRunner.Factory.getInstance()
        Runner.run( MagicMock() )

        PLF.assert_called_once()

    @patch( 'biomed.pipeline_runner.Pipeline.Factory.getInstance' )
    def test_it_runs_a_singel_configuration( self, PLF: MagicMock ):
        Data = MagicMock()
        TestData = MagicMock()
        ShortName = MagicMock()
        Description = MagicMock()
        Config = {
            "trainings_data": Data,
            "test_data": TestData,
            "shortname": ShortName,
            "description": Description
        }

        Pipe = MagicMock( spec = Pipeline )
        PLF.return_value = Pipe

        Runner = PipelineRunner.Factory.getInstance()
        Runner.run( [ Config ] )

        Pipe.pipe.assert_called_once_with(
            Data = Data,
            TestData = TestData,
            ShortName = ShortName,
            Description = Description,
            Properties = Config
        )

    @patch( 'biomed.pipeline_runner.Pipeline.Factory.getInstance' )
    def test_it_runs_multiple_configurations( self, PLF: MagicMock ):
        Data1 = MagicMock()
        TestData1 = MagicMock()
        ShortName1 = MagicMock()
        Description1 = MagicMock()
        Config1 = {
            "trainings_data": Data1,
            "test_data": TestData1,
            "shortname": ShortName1,
            "description": Description1
        }

        Data2 = MagicMock()
        TestData2 = MagicMock()
        ShortName2 = MagicMock()
        Description2 = MagicMock()
        Config2 = {
            "trainings_data": Data2,
            "test_data": TestData2,
            "shortname": ShortName2,
            "description": Description2
        }

        Pipe = MagicMock( spec = Pipeline )
        PLF.return_value = Pipe

        Runner = PipelineRunner.Factory.getInstance()
        Runner.run( [ Config1, Config2 ] )

        self.assertEqual(
            2,
            Pipe.pipe.call_count
        )

        Pipe.pipe.assert_any_call(
            Data = Data1,
            TestData = TestData1,
            ShortName = ShortName1,
            Description = Description1,
            Properties = Config1
        )

        Pipe.pipe.assert_any_call(
            Data = Data2,
            TestData = TestData2,
            ShortName = ShortName2,
            Description = Description2,
            Properties = Config2
        )
