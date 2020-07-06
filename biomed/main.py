import collections
import os as OS
import sys as Sys
from datetime import datetime

from biomed.properties_manager import PropertiesManager

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

from biomed.file_handler import FileHandler
from biomed.pipeline_runner import PipelineRunner
from biomed.plotter import Plotter

def printResults( Predictions ):
    def outputResults( prediction: list ):
        output_predictions = f"pmid,{ PropertiesManager().classifier }\n"
        found_targets = list()
        found_pmids = list()
        for index in range( 0, len( prediction[ 0 ] ) ):
            output_predictions += f"{prediction[ 3 ][ 'pmid' ].iloc[ index ]},{prediction[ 1 ][ index ]}\n"
            if prediction[ 0 ][ index ] != 0:
                found_targets.append( prediction[ 1 ][ index ] )
                found_pmids.append( prediction[ 3 ][ 'pmid' ].iloc[ index ] )

        print("scores:")
        print( prediction[ 2 ] )

        print('number of cancer predictions found_targets:', len(found_targets))
        counter = collections.Counter(found_targets)
        print('(doid, count):', counter.most_common())
        print('cancer found in articles with PMID:', found_pmids)
        return output_predictions

    for key in Predictions:
        print( "Configuration ID ", key )
        output_predictions = outputResults( Predictions[ key ] )
        path = OS.path.abspath(
            OS.path.join(
                OS.path.dirname( __file__ ), "..", "results",
                f"{ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') }_{ key }.csv "
            )
        )
        with open(path, "w") as file:
            file.write(output_predictions)

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
