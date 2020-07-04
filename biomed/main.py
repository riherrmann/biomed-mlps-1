import collections
import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

from biomed.file_handler import FileHandler
from biomed.pipeline_runner import PipelineRunner
from biomed.plotter import Plotter

def printResults( Predictions ):
    def outputResults( prediction: list ):
        found = list()
        for index in range( 0, len( prediction[ 0 ] ) ):
            if prediction[ 0 ][ index ] != 0:
                found.append( prediction[ 1 ][ index ] )

        print("scores:")
        print( prediction[ 2 ] )

        print('number of cancer predictions found:', len(found))
        counter = collections.Counter(found)
        print('(doid, count):', counter.most_common())

    for key in Predictions:
        print( "Configuration ID ", key )
        outputResults( Predictions[ key ] )

if __name__ == '__main__':
    training_data_location = training_data_location = OS.path.abspath(
        OS.path.join(
            OS.path.dirname( __file__ ), "..", "training_data", "train.tsv"
        )
    )

    fh = FileHandler()
    training_data = fh.read_tsv_pandas_data_structure(training_data_location)

    plotter = Plotter()
    plotter.plot_target_distribution(training_data)
    # exit(0)

    Runner = PipelineRunner.Factory.getInstance()
    Results = Runner.run( [ { "id": "1", "data": training_data } ] )
    printResults( Results )
