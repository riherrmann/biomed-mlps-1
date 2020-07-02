import unittest
from unittest.mock import MagicMock, patch, ANY

from pandas import DataFrame
from biomed.pipeline import Pipeline
from biomed.text_mining_manager import TextMiningManager
from biomed.properties_manager import PropertiesManager

class PipelineSpec( unittest.TestCase ):

    @patch( 'biomed.pipeline.PolymorphPreprocessor.Factory.getInstance' )
    def test_it_is_a_Pipeline( self, _ ):
        Pipe = Pipeline.Factory.getInstance( "is_cancer" )
        self.assertTrue( isinstance( Pipe, Pipeline ) )

    @patch( 'biomed.pipeline.PolymorphPreprocessor.Factory.getInstance' )
    @patch( 'biomed.pipeline.PropertiesManager' )
    def test_it_initializes_the_properties_manager( self, PM: MagicMock, _ ):
        Pipeline.Factory.getInstance( "is_cancer" )
        PM.assert_called_once_with()

    @patch( 'biomed.pipeline.PolymorphPreprocessor.Factory.getInstance' )
    @patch( 'biomed.pipeline.PropertiesManager' )
    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_initializes_the_text_mining(
        self,
        TM: MagicMock,
        PMF: MagicMock,
        PPF: MagicMock
    ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        Given = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )

        PM = MagicMock( spec = PropertiesManager )
        PMF.return_value = PM

        Pipe = Pipeline.Factory.getInstance( "is_cancer" )
        Pipe.pipe( Given )

        PPF.assert_called_once_with( PM )

        TM.assert_called_once_with(
            PM,
            ANY
        )

    @patch( 'biomed.pipeline.PolymorphPreprocessor.Factory.getInstance' )
    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_runs_the_text_miner_with_the_given_data( self, TM: MagicMock, _ ):
        TestData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        Given = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )

        TMM = MagicMock( spec = TextMiningManager )
        TM.return_value = TMM

        Pipe = Pipeline.Factory.getInstance( "is_cancer" )
        Pipe.pipe( Given )

        TMM.setup_for_input_data.assert_called_once_with( Given )

    @patch( 'biomed.pipeline.PolymorphPreprocessor.Factory.getInstance' )
    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_runs_the_text_miner_with_the_given_target_dimension( self, TM: MagicMock, _ ):
        Given = "is_cancer"
        TMM = MagicMock( spec = TextMiningManager )
        TM.return_value = TMM

        Pipe = Pipeline.Factory.getInstance( Given )
        Pipe.pipe( MagicMock() )

        TMM.setup_for_target_dimension.assert_called_once_with( Given )

    @patch( 'biomed.pipeline.PolymorphPreprocessor.Factory.getInstance' )
    @patch( 'biomed.pipeline.TextMiningManager' )
    def test_it_returns_the_computed_predictions( self, TM: MagicMock, _ ):
        Expected = 42
        TMM = MagicMock( spec = TextMiningManager )
        TMM.get_binary_mlp_predictions.return_value = Expected
        TM.return_value = TMM

        Pipe = Pipeline.Factory.getInstance( "is_cancer" )
        self.assertEqual(
            Expected,
            Pipe.pipe( MagicMock() )
        )

    @patch( 'biomed.pipeline.PolymorphPreprocessor.Factory.getInstance' )
    @patch( 'biomed.pipeline.TextMiningManager' )
    @patch( 'biomed.pipeline.PropertiesManager' )
    def test_it_assigns_new_properties( self, PMF: MagicMock, _, __ ):
        PM = dict()
        PMF.return_value = PM
        Expected = { "workers": 23 }

        Pipe = Pipeline.Factory.getInstance( "is_cancer" )
        Pipe.pipe( MagicMock(), Expected )

        self.assertDictEqual(
            Expected,
            PM
        )
