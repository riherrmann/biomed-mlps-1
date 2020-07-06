import unittest
from unittest.mock import MagicMock, patch, ANY
from pandas import DataFrame
from biomed.pipeline_runner import PipelineRunner
from biomed.pipeline import Pipeline
from multiprocessing import Process

def reflectId( Train: DataFrame, Test: DataFrame, Config: dict ):
    return Config[ "id" ]

class PipelineRunnerSpec( unittest.TestCase ):
    def test_it_is_a_Pipeline_Runner( self ):
        Runner = PipelineRunner.Factory.getInstance()
        self.assertTrue( isinstance( Runner, PipelineRunner ) )

    @patch( 'biomed.pipeline_runner.Process' )
    def test_it_instanciates_worker_processes( self, MProcess: MagicMock ):
        Workers = 2
        Permutations = [
            { "id": 1, "workers": 32 },
            { "id": 1, "workers": 2 },
            { "id": 3, "workers": 42 }
        ]

        Runner = PipelineRunner.Factory.getInstance()
        Runner.run( Permutations, Workers )
        self.assertEqual(
            Workers,
            MProcess.call_count
        )

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

    @patch( 'biomed.pipeline_runner.Pipeline.Factory' )
    @patch( 'biomed.pipeline_runner.Process' )
    def test_it_spilts_the_permutation_in_chunks( self, MProcess: MagicMock, _ ):
        Workers = 2
        Permutations = [
            { "id": 1, "workers": 32 },
            { "id": 2, "workers": 2 },
            { "id": 3, "workers": 42 }
        ]

        Runner = PipelineRunner.Factory.getInstance()
        Runner.run( Permutations, Workers )

        MProcess.assert_any_call(
            target = ANY,
            args = (
                Runner, [
                    { "id": 1, "workers": 32 },
                    { "id": 2, "workers": 2 }
                ] )
        )

        MProcess.assert_any_call(
            target = ANY,
            args = (
                Runner, [
                    { "id": 3, "workers": 42 },
                ] )
        )

    @patch( 'biomed.pipeline_runner.Pipeline.Factory' )
    @patch( 'biomed.pipeline_runner.Process' )
    def test_it_dispatches_the_subprocess( self, MProcess: MagicMock, _ ):

        SubProcess = MagicMock( spec = Process )
        MProcess.return_value = SubProcess

        Permutations = [
            { "id": 1, "workers": 32 },
            { "id": 2, "workers": 2 },
            { "id": 3, "workers": 42 }
        ]
        Workers = 2

        Runner = PipelineRunner.Factory.getInstance()
        Runner.run( Permutations, Workers )
        self.assertEqual(
            Workers,
            SubProcess.start.call_count
        )

    @patch( 'biomed.pipeline_runner.Pipeline.Factory' )
    @patch( 'biomed.pipeline_runner.Process' )
    def test_it_waits_of_all_subprocess( self, MProcess: MagicMock, _ ):

        SubProcess = MagicMock( Process )
        MProcess.return_value = SubProcess

        Permutations = [
            { "id": 1, "workers": 32 },
            { "id": 2, "workers": 2 },
            { "id": 3, "workers": 42 }
        ]
        Workers = 2

        Runner = PipelineRunner.Factory.getInstance()
        Runner.run( Permutations, Workers )
        self.assertEqual(
            Workers,
            SubProcess.join.call_count
        )

    @patch( 'biomed.pipeline_runner.Pipeline.Factory.getInstance' )
    def test_it_gathers_and_returns_the_output_of_the_pipeline_in_for_all_subprocesses( self, PMF: MagicMock ):
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

        Permutations = [
            { "id": 1, "workers": 32, "training": Train, "test": Test },
            { "id": 2, "workers": 2, "training": Train, "test": Test },
            { "id": 3, "workers": 42, "training": Train, "test": Test }
        ]

        Workers = 2

        P = MagicMock( spec = Pipeline )
        P.pipe.side_effect = reflectId
        PMF.return_value = P

        ExpectedOutput = { 1: 1 , 2: 2, 3: 3 }
        Runner = PipelineRunner.Factory.getInstance()

        self.assertDictEqual(
            ExpectedOutput,
            Runner.run( Permutations, Workers )
        )
