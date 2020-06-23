import unittest
from unittest.mock import MagicMock, patch, ANY
from biomed.pipeline_runner import PipelineRunner

class StubbedSubprocess:
    def start( self ):
        pass
    def join( self ):
        pass

class PipelineRunnerSpec( unittest.TestCase ):

    @patch( 'biomed.pipeline_runner.Process' )
    def test_it_instanciates_worker_processes( self, MProcess: MagicMock ):
        Workers = 2
        Permutations = [
            { "id": 1, "workers": 32 },
            { "id": 1, "workers": 2 },
            { "id": 3, "workers": 42 }
        ]

        Runner = PipelineRunner()
        Runner.run( Permutations, Workers )
        self.assertEqual(
            Workers,
            MProcess.call_count
        )

    @patch( 'biomed.pipeline_runner.Process' )
    def test_it_spilts_the_permutation_in_chunks( self, MProcess: MagicMock ):
        Workers = 2
        Permutations = [
            { "id": 1, "workers": 32 },
            { "id": 2, "workers": 2 },
            { "id": 3, "workers": 42 }
        ]

        Runner = PipelineRunner()
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

    @patch( 'biomed.pipeline_runner.Process' )
    def test_it_dispatches_the_subprocess( self, MProcess: MagicMock ):

        SubProcess = StubbedSubprocess()
        SubProcess.start = MagicMock()
        MProcess.return_value = SubProcess

        Permutations = [
            { "id": 1, "workers": 32 },
            { "id": 2, "workers": 2 },
            { "id": 3, "workers": 42 }
        ]
        Workers = 2

        Runner = PipelineRunner()
        Runner.run( Permutations, Workers )
        self.assertEqual(
            Workers,
            SubProcess.start.call_count
        )

    @patch( 'biomed.pipeline_runner.Process' )
    def test_it_waits_of_all_subprocess( self, MProcess: MagicMock ):

        SubProcess = StubbedSubprocess()
        SubProcess.join = MagicMock()
        MProcess.return_value = SubProcess

        Permutations = [
            { "id": 1, "workers": 32 },
            { "id": 2, "workers": 2 },
            { "id": 3, "workers": 42 }
        ]
        Workers = 2

        Runner = PipelineRunner()
        Runner.run( Permutations, Workers )
        self.assertEqual(
            Workers,
            SubProcess.join.call_count
        )
