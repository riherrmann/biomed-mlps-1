import os as OS
import sys as Sys
import argparse as Args
import pandas

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

from biomed.pipeline_runner import PipelineRunner

if __name__ == '__main__':
    Parser = Args.ArgumentParser( description = 'FeedForword NN' )
    Parser.add_argument(
        "-e",
        "--train_data",
        type=str,
        required=True,
        help='Path to the trainings data'
    )
    Parser.add_argument(
        "-t",
        "--test_data",
        type=str,
        required=False,
        help='Path to the test data',
        const = None
    )

    Parsed = Parser.parse_args()

    Data = pandas.read_csv( Parsed.train_data, delimiter="\t" )
    if Parsed.test_data:
        TestData = pandas.read_csv( Parsed.test_data, delimiter = "\t" )
    else:
        TestData = None

    Runner = PipelineRunner.Factory.getInstance()
    Runner.run( [ {
        "shortname": "test1",
        "description": "test run",
        "trainings_data": Data,
        "test_data": TestData,
    } ] )
