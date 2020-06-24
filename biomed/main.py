from biomed.file_handler import FileHandler
from biomed.pipeline_runner import PipelineRunner



if __name__ == '__main__':
    training_data_location = "training_data/train.tsv"
    fh = FileHandler()
    training_data = fh.read_tsv_pandas_data_structure(training_data_location)

    Runner = PipelineRunner.Factory.getInstance( "is_cancer" )
    preds = Runner.run( [ { "id": 1, "data": training_data } ] )

    print(preds)
    #cancer_types_found = [x for x in preds if x != 0]
    #cancer_types_found = tmm.map_doid_values_to_nonsequential(cancer_types_found)
    #print('number of cancer predictions found:', len(cancer_types_found))
    #print('(doid, count):', counter.most_common())
