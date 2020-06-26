import collections
import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

from biomed.file_handler import FileHandler
from biomed.pipeline_runner import PipelineRunner

if __name__ == '__main__':
    training_data_location = training_data_location = OS.path.abspath(
        OS.path.join(
            OS.path.dirname( __file__ ), "..", "training_data", "train.tsv"
        )
    )

    fh = FileHandler()
    training_data = fh.read_tsv_pandas_data_structure(training_data_location)

    Runner = PipelineRunner.Factory.getInstance( "is_cancer" )
    preds = Runner.run( [ { "id": "1", "data": training_data } ] )

    print(preds[ "1" ][ 0 ])

    found = list()
    for index in range( 0, len( preds[ "1" ][ 0 ] ) ):
        if preds[ "1" ][ 0 ][ index ] != 0:
            found.append( preds[ "1" ][ 1 ][ index ] )

    print('number of cancer predictions found:', len(found))
    counter = collections.Counter(found)
    print('(doid, count):', counter.most_common())
