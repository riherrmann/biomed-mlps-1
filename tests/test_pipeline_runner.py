import unittest
from unittest.mock import MagicMock, patch, ANY
from pandas import DataFrame
from biomed.pipeline_runner import PipelineRunner
from biomed.pipeline import Pipeline

def reflectId( Train: DataFrame, Test: DataFrame, Config: dict ):
    return Config[ "id" ]

class PipelineRunnerSpec( unittest.TestCase ):
    def test_it_is_a_Pipeline_Runner( self ):
        Runner = PipelineRunner.Factory.getInstance()
        self.assertTrue( isinstance( Runner, PipelineRunner ) )

    @patch( 'biomed.pipeline_runner.Pipeline.Factory.getInstance' )
    def test_it_runs_the_pipeline_with_given_permutations( self, PMF: MagicMock ):
        TrainingsData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        TestData = {
            'pmid': [ 23 ],
            'cancer_type': [ -1 ],
            'doid': [ 42 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        Train = DataFrame( TrainingsData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        Test = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        Permutations = [ {
            "id": 1,
            "training": Train,
            "test": Test,
            "workers": 23
        }, {
            "id": 2,
            "training": Train,
            "test": Test,
            "workers": 42
        } ]

        P = MagicMock( spec = Pipeline )
        P.pipe.side_effect = reflectId
        PMF.return_value = P

        Runner = PipelineRunner.Factory.getInstance()
        Runner.run( Permutations )

        P.pipe.assert_any_call( Train, Test, Permutations[ 0 ] )
        P.pipe.assert_any_call( Train, Test, Permutations[ 1 ] )

    @patch( 'biomed.pipeline_runner.Pipeline.Factory.getInstance' )
    def test_it_gathers_and_returns_the_output_of_the_pipeline( self, PMF: MagicMock ):
        TrainingsData = {
            'pmid': [ 42 ],
            'cancer_type': [ -1 ],
            'doid': [ 23 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        TestData = {
            'pmid': [ 23 ],
            'cancer_type': [ -1 ],
            'doid': [ 42 ],
            'is_cancer': [ False ],
            'text': [ "My little cute poney is a poney" ]
        }

        Train = DataFrame( TrainingsData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )
        Test = DataFrame( TestData, columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ] )

        Permutations = [ {
            "id": 1,
            "training": Train,
            "test": Test,
            "workers": 23
        }, {
            "id": 2,
            "training": Train,
            "test": Test,
            "workers": 42
        } ]

        P = MagicMock( spec = Pipeline )
        P.pipe.side_effect = reflectId
        PMF.return_value = P

        ExpectedOutput = { 1: 1 , 2: 2 }
        Runner = PipelineRunner.Factory.getInstance()

        self.assertDictEqual(
            ExpectedOutput,
            Runner.run( Permutations )
        )
