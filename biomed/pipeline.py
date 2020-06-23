import collections

from biomed.text_mining_manager import TextMiningManager

def pipeline(
    training_data,
    tmm: TextMiningManager,
) -> int:
    print('Setup for input data')
    tmm.setup_for_input_data(training_data)
    target_dimension = 'doid'
    # target_dimension = 'is_cancer'
    print('Setup for target dimension', target_dimension)
    tmm.setup_for_target_dimension(target_dimension)
    print('Build MLP and get predictions')
    preds = tmm.get_binary_mlp_predictions()
    print(preds)
    cancer_types_found = [x for x in preds if x != 0]
    cancer_types_found = tmm.map_doid_values_to_nonsequential(cancer_types_found)
    print('number of cancer predictions found:', len(cancer_types_found))

    return collections.Counter(cancer_types_found)
