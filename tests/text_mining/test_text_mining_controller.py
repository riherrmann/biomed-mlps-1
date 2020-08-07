import unittest
from unittest.mock import patch, MagicMock
from biomed.text_mining.controller import Controller
from biomed.text_mining.text_mining_controller import TextminingController
from biomed.properties_manager import PropertiesManager
from biomed.facilitymanager.facility_manager import FacilityManager
from biomed.splitter.splitter import Splitter
from biomed.preprocessor.preprocessor import Preprocessor
from biomed.vectorizer.vectorizer import Vectorizer
from biomed.mlp.mlp import MLP
from biomed.mlp.input_data import InputData
from biomed.evaluator.evaluator import Evaluator
from pandas import DataFrame, Series
from numpy import array as Array

class TextminingControllerSpec( unittest.TestCase ):
    def setUp( self ):
        self.__Data = DataFrame(
            {
                'pmid': [ '1a', '2a', '3a', '4a' ],
                'text': [
                    "My little cute Poney is a Poney",
                    "My little farm is cute.",
                    "My little programm is a application and runs and runs and runs.",
                    "My little keyboard is to small"
                ],
                'is_cancer': [ 0, 1, 1, 0 ],
                'doid': [ -1, 1, 2, -1 ],
                'cancer_type': [ 'no cancer', 'cancer', 'cancer', 'no cancer' ],
            },
            columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ]
        )


        self.__PM = PropertiesManager()
        self.__FacilityManager = MagicMock( spec = FacilityManager )
        self.__Splitter = MagicMock( spec = Splitter )
        self.__Preprocessor = MagicMock( spec = Preprocessor )
        self.__Vectorizer = MagicMock( spec = Vectorizer )
        self.__MLP = MagicMock( spec = MLP )
        self.__Evaluator = MagicMock( spec = Evaluator )

        self.__FacilityManager.clean.return_value = self.__Data

        self.__Splitter.trainingSplit.return_value = [ ( MagicMock(), MagicMock() ) ]
        self.__Splitter.validationSplit.return_value = ( MagicMock(), MagicMock() )

        self.__Preprocessor.preprocessCorpus.return_value = MagicMock()

        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = []
        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__Vectorizer.featureizeTest.return_value = MagicMock()
        self.__Vectorizer.getSupportedFeatures.return_value = MagicMock()

    def __fakeLocator( self, ServiceKey: str, _ ):
        Dependencies = {
            'properties': self.__PM,
            'facilitymanager': self.__FacilityManager,
            'splitter': self.__Splitter,
            'preprocessor': self.__Preprocessor,
            'vectorizer': self.__Vectorizer,
            'mlp': self.__MLP,
            'evaluator': self.__Evaluator
        }

        return Dependencies[ ServiceKey ]

    def test_it_is_a_Controller( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MyController, Controller ) )

    def test_it_depends_on_many_things( self ):
        def fakeLocator( ServiceKey: str, Type ):
            Dependencies = {
                'properties': PropertiesManager,
                'splitter': Splitter,
                'facilitymanager': FacilityManager,
                'preprocessor': Preprocessor,
                'vectorizer': Vectorizer,
                'mlp': MLP,
                'evaluator': Evaluator
            }

            if not ServiceKey in Dependencies.keys():
                raise RuntimeError( 'Unknown ServiceKey {}'.format( ServiceKey ) )

            if not Type == Dependencies[ ServiceKey ]:
                raise RuntimeError( 'Unknown Type for {}'.format( ServiceKey ) )

            return MagicMock( spec = Dependencies[ ServiceKey ] )

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeLocator

        TextminingController.Factory.getInstance( ServiceGetter )
        self.assertEqual(
            7,
            ServiceGetter.call_count
        )

    def test_it_starts_the_evaluator( self ):
        Name = MagicMock()
        Description = MagicMock()

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = Name,
            Description = Description
        )

        self.__Evaluator.start.assert_called_once_with(
            Name,
            Description
        )

    def test_it_cleans_up_the_Data( self ):
        Data = MagicMock()

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__FacilityManager.clean.assert_called_once_with( Data )

    def test_it_splits_the_test_data_for_binary( self ):
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Splitter.trainingSplit.assert_called_once_with(
            self.__Data[ 'pmid' ],
            self.__Data[ 'is_cancer' ]
        )

    def test_it_splits_the_test_data_for_mulitclass( self ):
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Splitter.trainingSplit.assert_called_once_with(
            self.__Data[ 'pmid' ],
            self.__Data[ 'doid' ]
        )

    def test_it_caputure_the_ids_of_training_including_the_validation_ids_and_test_sets( self ):
        TrainingsIds = MagicMock()
        TestIds = MagicMock()

        self.__Splitter.trainingSplit.return_value = [ ( TrainingsIds, TestIds ) ]

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureData.assert_called_once_with(
            TrainingsIds,
            TestIds
        )

    def test_it_caputure_the_ids_of_training_including_the_validation_ids_and_test_sets_for_k_folds( self ):
        TrainingsIds1 = MagicMock()
        TestIds1 = MagicMock()

        TrainingsIds2 = MagicMock()
        TestIds2 = MagicMock()

        self.__Splitter.trainingSplit.return_value = [
            ( TrainingsIds1, TestIds1 ),
            ( TrainingsIds2, TestIds2 )
        ]

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureData.assert_any_call(
            TrainingsIds1,
            TestIds1
        )

        self.__Evaluator.captureData.assert_any_call(
            TrainingsIds2,
            TestIds2
        )

    def test_it_caputure_the_start_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureStartTime.assert_called_once()

    def test_it_preprocesses_the_trainings_data_including_validation_data_and_test_data( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Preprocessor.preprocessCorpus.assert_called_once_with(
            self.__Data[ 'pmid' ],
            self.__Data[ 'text' ]
        )

    def test_it_captures_the_preprocessing_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.capturePreprocessingTime.assert_called_once()

    def test_it_captures_the_preprocessed_coprus_in_compare_to_org_corpus( self ):
        ProCorpus = MagicMock()

        self.__Preprocessor.preprocessCorpus.return_value = ProCorpus

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.capturePreprocessedData(
            Processed = ProCorpus,
            Original = self.__Data[ 'text' ]
        )

    def test_it_vectorizes_the_trainings_coprus_for_binary( self ):
        ProCorpus = Series(
            [
                "Poney Poney",
                "farm",
                "programm application",
                "keyboard"
            ],
            index = [ '1a', '2a', '3a', '4a' ]
        )

        TrainingsIds = Series( [ '1a', '3a' ], index = [ 0, 1 ] )
        Labels = self.__Data[ 'is_cancer' ]
        Labels.index = [ '1a', '2a', '3a', '4a' ]
        Labels = Labels.filter( [ '1a', '3a' ] )
        TestIds = Series( [ '2a', '4a' ] )

        self.__Splitter.trainingSplit.return_value = [ ( TrainingsIds, TestIds ) ]
        self.__Preprocessor.preprocessCorpus.return_value = ProCorpus
        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = [[1],[2]]
        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Vectorizer.featureizeTrain.call_args_list[ 0 ]

        self.assertListEqual(
            list( ProCorpus.filter( list( TrainingsIds ) ) ),
            list( Arguments[ 0 ] )
        )

        self.assertListEqual(
            list( Labels ),
            list( Arguments[ 1 ] )
        )

        self.__Vectorizer.featureizeTrain.assert_called_once()

    def test_it_vectorizes_the_trainings_coprus_for_mulitclass( self ):
        ProCorpus = Series(
            [
                "Poney Poney",
                "farm",
                "programm application",
                "keyboard"
            ],
            index = [ '1a', '2a', '3a', '4a' ]
        )

        TrainingsIds = Series( [ '1a', '3a' ], index = [ 0, 1 ] )
        Labels = self.__Data[ 'doid' ]
        Labels.index = [ '1a', '2a', '3a', '4a' ]
        Labels = Labels.filter( [ '1a', '3a' ] )
        TestIds = Series( [ '2a', '4a' ] )

        self.__Splitter.trainingSplit.return_value = [ ( TrainingsIds, TestIds ) ]
        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = [[1],[2]]
        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__Preprocessor.preprocessCorpus.return_value = ProCorpus
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Vectorizer.featureizeTrain.call_args_list[ 0 ]

        self.assertListEqual(
            list( ProCorpus.filter( list( TrainingsIds ) ) ),
            list( Arguments[ 0 ] )
        )

        self.assertListEqual(
            list( Labels ),
            list( Arguments[ 1 ] )
        )

        self.__Vectorizer.featureizeTrain.assert_called_once()

    def test_it_vectorizes_the_test_coprus( self ):
        ProCorpus = Series(
            [
                "Poney Poney",
                "farm",
                "programm application",
                "keyboard"
            ],
            index = [ '1a', '2a', '3a', '4a' ]
        )

        TrainingsIds = Series( [ '1a', '3a' ], index = [ 0, 1 ] )
        TestIds = Series( [ '2a', '4a' ], index = [ 0, 1 ] )

        self.__Splitter.trainingSplit.return_value = [ ( TrainingsIds, TestIds ) ]
        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = [ [1],[2] ]
        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__Preprocessor.preprocessCorpus.return_value = ProCorpus

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Vectorizer.featureizeTest.call_args_list[ 0 ]

        self.assertListEqual(
            list( ProCorpus.filter( list( TestIds ) ) ),
            list( Arguments[ 0 ] )
        )

        self.__Vectorizer.featureizeTest.assert_called_once()

    def test_it_captures_the_vectorizing_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureVectorizingTime.assert_called_once()

    def test_it_captures_the_resulting_features_and_their_Labels( self ):
        TrainingIds = MagicMock()
        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = []
        TestIds = MagicMock()
        TestFeatures = MagicMock()
        BagOfWords = MagicMock()

        self.__Splitter.trainingSplit.return_value = [ ( TrainingIds, TestIds ) ]
        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__Vectorizer.getSupportedFeatures.return_value = BagOfWords

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureFeatures.assert_called_once_with(
            ( TrainingIds, TrainFeatures ),
            ( TestIds, TestFeatures ),
            BagOfWords
        )

    def test_it_splits_the_validation_data_of_the_trainings_data_for_binary( self ):
        TrainingIds = Series( [ '1a', '3a' ], index = [ 0, 1 ] )
        TestIds = MagicMock()
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ] ] )
        Expected = self.__Data[ 'is_cancer' ]
        Expected.index = list( self.__Data[ 'pmid' ] )
        Expected = Expected.filter( list( TrainingIds ) )

        self.__Splitter.trainingSplit.return_value = [ ( TrainingIds, TestIds ) ]
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Splitter.validationSplit.call_args_list[ 0 ]

        self.assertListEqual(
            list( TrainingIds ),
            list( Arguments[ 0 ] )
        )

        self.assertListEqual(
            list( Expected ),
            list( Arguments[ 1 ] )
        )

        self.__Splitter.validationSplit.assert_called_once()

    def test_it_splits_the_validation_data_of_the_trainings_data_for_multiclass( self ):
        TrainingIds = Series( [ '1a', '3a' ], index = [ 0, 1 ] )
        TestIds = MagicMock()
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ] ] )
        Expected = self.__Data[ 'doid' ]
        Expected.index = list( self.__Data[ 'pmid' ] )
        Expected = Expected.filter( list( TrainingIds ) )

        self.__PM.classifier = 'doid'
        self.__Splitter.trainingSplit.return_value = [ ( TrainingIds, TestIds ) ]
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Splitter.validationSplit.call_args_list[ 0 ]

        self.assertListEqual(
            list( TrainingIds ),
            list( Arguments[ 0 ] )
        )

        self.assertListEqual(
            list( Expected ),
            list( Arguments[ 1 ] )
        )

        self.__Splitter.validationSplit.assert_called_once()

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_brings_the_features_into_model_input_format(
        self,
        DataBinding: MagicMock
    ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ] ] )
        TrainingIds = Series( [ '1a' ] )
        ValidationIds = Series( [ '2a' ] )
        TestFeatures = Array( [ [ 0.1, 0. ], [ 0.15, 0.5 ] ] )
        TestIds = Series( [ '3a', '4a' ] )

        self.__Splitter.trainingSplit.return_value = [
            ( Series( [ '1a', '2a' ] ), TestIds )
        ]
        self.__Splitter.validationSplit.return_value = ( TrainingIds, ValidationIds )
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        ArgumentsFeatures, _ = DataBinding.call_args_list[ 0 ]
        self.assertEqual(
            [ TrainingFeatures.tolist()[ 0 ] ],
            ArgumentsFeatures[ 0 ].tolist()
        )
        self.assertEqual(
            [ TrainingFeatures.tolist()[ 1 ] ],
            ArgumentsFeatures[ 1 ].tolist()
        )
        self.assertEqual(
            TestFeatures.tolist(),
            ArgumentsFeatures[ 2 ].tolist()
        )

    @patch( 'biomed.text_mining.text_mining_controller.hotEncode' )
    def test_it_hot_encodes_the_labels_for_binary(
        self,
        HotEncoder: MagicMock
    ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ] ] )
        TrainingIds = Series( [ '1a' ] )
        ValidationIds = Series( [ '2a' ] )
        TestFeatures = Array( [ [ 0.1, 0. ], [ 0.15, 0.5 ] ] )
        TestIds = Series( [ '3a', '4a' ] )

        self.__Splitter.trainingSplit.return_value = [
            ( Series( [ '1a', '2a' ] ), TestIds )
        ]
        self.__Splitter.validationSplit.return_value = ( TrainingIds, ValidationIds )
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        ArgumentsLabels, _ = HotEncoder.call_args_list[ 0 ]

        self.assertEqual(
            list( self.__Data[ 'is_cancer' ].filter( list( TrainingIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )
        self.assertEqual(
            2,
            ArgumentsLabels[ 1 ]
        )

        ArgumentsLabels, _ = HotEncoder.call_args_list[ 1 ]
        self.assertEqual(
            list( self.__Data[ 'is_cancer' ].filter( list( ValidationIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )
        self.assertEqual(
            2,
            ArgumentsLabels[ 1 ]
        )

        ArgumentsLabels, _ = HotEncoder.call_args_list[ 2 ]
        self.assertEqual(
            list( self.__Data[ 'is_cancer' ].filter( list( TestIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )
        self.assertEqual(
            2,
            ArgumentsLabels[ 1 ]
        )

    @patch( 'biomed.text_mining.text_mining_controller.hotEncode' )
    def test_it_hot_encodes_the_labels_for_multiclass(
        self,
        HotEncoder: MagicMock
    ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ] ] )
        TrainingIds = Series( [ '1a' ] )
        ValidationIds = Series( [ '2a' ] )
        TestFeatures = Array( [ [ 0.1, 0. ], [ 0.15, 0.5 ] ] )
        TestIds = Series( [ '3a', '4a' ] )

        self.__Splitter.trainingSplit.return_value = [
            ( Series( [ '1a', '2a' ] ), TestIds )
        ]
        self.__Splitter.validationSplit.return_value = ( TrainingIds, ValidationIds )
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        ArgumentsLabels, _ = HotEncoder.call_args_list[ 0 ]

        self.assertEqual(
            list( self.__Data[ 'doid' ].filter( list( TrainingIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )
        self.assertEqual(
            len( self.__Data[ 'doid' ].unique() ),
            ArgumentsLabels[ 1 ]
        )

        ArgumentsLabels, _ = HotEncoder.call_args_list[ 1 ]
        self.assertEqual(
            list( self.__Data[ 'doid' ].filter( list( ValidationIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )
        self.assertEqual(
            len( self.__Data[ 'doid' ].unique() ),
            ArgumentsLabels[ 1 ]
        )

        ArgumentsLabels, _ = HotEncoder.call_args_list[ 2 ]
        self.assertEqual(
            list( self.__Data[ 'doid' ].filter( list( TestIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )
        self.assertEqual(
            len( self.__Data[ 'doid' ].unique() ),
            ArgumentsLabels[ 1 ]
        )


    @patch( 'biomed.text_mining.text_mining_controller.hotEncode' )
    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_collects_the_input_data_for_labels(
        self,
        DataBinding: MagicMock,
        HotEncoder: MagicMock
    ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ] ] )
        TrainingIds = Series( [ '1a' ] )
        ValidationIds = Series( [ '2a' ] )
        TestFeatures = Array( [ [ 0.1, 0. ], [ 0.15, 0.5 ] ] )
        TestIds = Series( [ '3a', '4a' ] )

        EncodedTrainingLabels = MagicMock()
        EncodedValidationLabels = MagicMock()
        EncodedTestLabels = MagicMock()
        Encoded = [ EncodedTrainingLabels, EncodedValidationLabels, EncodedTestLabels ]


        self.__Splitter.trainingSplit.return_value = [
            ( Series( [ '1a', '2a' ] ), TestIds )
        ]
        self.__Splitter.validationSplit.return_value = ( TrainingIds, ValidationIds )
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        HotEncoder.side_effect = lambda _, __ : Encoded.pop( 0 )

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        ArgumentsLabels, _ = DataBinding.call_args_list[ 1 ]

        self.assertEqual(
            EncodedTrainingLabels,
            ArgumentsLabels[ 0 ]
        )
        self.assertEqual(
            EncodedValidationLabels,
            ArgumentsLabels[ 1 ]
        )
        self.assertEqual(
            EncodedTestLabels,
            ArgumentsLabels[ 2 ]
        )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_builds_a_model( self, DataBinding: MagicMock ):
        Dimension = 2
        Payload = MagicMock( spec = InputData )
        Payload.Training = MagicMock()
        Payload.Training.shape = ( Dimension, MagicMock() )
        Payload.Test = MagicMock()

        DataBinding.return_value = Payload

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.buildModel.assert_called_once_with( Dimension )

    def test_it_saves_the_model_structure( self ):
        Expected = "structure"

        self.__MLP.buildModel.return_value = Expected

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureModel.assert_called_once_with( Expected )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_trains_the_model( self, DataBinding: MagicMock ):
        Features = MagicMock()
        Labels = MagicMock()
        Bindings = [ Features, Labels ]

        DataBinding.side_effect = lambda _, __, ___ : Bindings.pop( 0 )

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.train.assert_called_once_with( Features, Labels )

    def test_it_captures_the_training_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureTrainingTime.assert_called_once()

    def test_it_saves_the_trainings_history( self ):
        History = MagicMock()

        self.__MLP.train.return_value = History

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureTrainingHistory.assert_called_once_with( History )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_evaluates_the_training( self, DataBinding: MagicMock ):
        Features = MagicMock()
        Labels = MagicMock()
        Bindings = [ Features, Labels ]

        DataBinding.side_effect = lambda _, __, ___ : Bindings.pop( 0 )

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.getTrainingScore.assert_called_once_with( Features, Labels )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_saves_the_training_evaluation( self, DataBinding: MagicMock ):
        Evaluation = MagicMock()

        self.__MLP.getTrainingScore.return_value = Evaluation

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureEvaluationScore.assert_called_once_with( Evaluation )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_predicts_on_the_given_test_data( self, DataBinding: MagicMock ):
        Features = MagicMock( spec = InputData )
        Test = MagicMock()
        Features.Training = MagicMock()
        Features.Validation = MagicMock()
        Features.Test = Test

        Labels = MagicMock()
        Bindings = [ Features, Labels ]

        DataBinding.side_effect = lambda _, __, ___ : Bindings.pop( 0 )

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.predict.assert_called_once_with( Test )

    def test_it_captures_the_prediction_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.caputrePredictingTime.assert_called_once()

    def test_it_captures_the_predictions_for_binary( self ):
        TestIds = Series( [ '2a', '4a' ] )

        self.__Splitter.trainingSplit.return_value = [ ( MagicMock(), TestIds ) ]

        Predictions = MagicMock()


        self.__MLP.predict.return_value = Predictions
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Evaluator.capturePredictions.call_args_list[ 0 ]

        self.assertEqual(
            Arguments[ 0 ],
            Predictions,
        )

        self.assertEqual(
            list( Arguments[ 1 ] ),
            list( TestIds )
        )

        self.assertEqual(
            list( Arguments[ 2 ] ),
            list( self.__Data[ 'is_cancer' ].filter( list( TestIds ) ) )
        )

        self.__Evaluator.capturePredictions.assert_called_once()

    def test_it_captures_the_predictions_for_mulitclass( self ):
        TestIds = Series( [ '2a', '4a' ] )

        self.__Splitter.trainingSplit.return_value = [ ( MagicMock(), TestIds ) ]

        Predictions = MagicMock()


        self.__MLP.predict.return_value = Predictions
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Evaluator.capturePredictions.call_args_list[ 0 ]

        self.assertEqual(
            Arguments[ 0 ],
            Predictions,
        )

        self.assertEqual(
            list( Arguments[ 1 ] ),
            list( TestIds )
        )

        self.assertEqual(
            list( Arguments[ 2 ] ),
            list( self.__Data[ 'doid' ].filter( list( TestIds ) ) )
        )

        self.__Evaluator.capturePredictions.assert_called_once()

    def test_it_captures_the_score_of_predictions_for_binary( self ):
        TestIds = Series( [ '2a', '4a' ] )
        ClassLabels = [ 0, 1 ]

        self.__Splitter.trainingSplit.return_value = [ ( MagicMock(), TestIds ) ]

        Predictions = MagicMock()

        self.__MLP.predict.return_value = Predictions
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Evaluator.score.call_args_list[ 0 ]

        self.assertEqual(
            Arguments[ 0 ],
            Predictions,
        )

        self.assertEqual(
            list( Arguments[ 1 ] ),
            list( self.__Data[ 'is_cancer' ].filter( list( TestIds ) ) )
        )

        self.assertEqual(
            list( Arguments[ 2 ] ),
            ClassLabels
        )

        self.__Evaluator.score.assert_called_once()

    def test_it_captures_the_score_of_predictions_for_multiclass( self ):
        TestIds = Series( [ '2a', '4a' ] )
        ClassLabels = [ -1, 1, 2 ]

        self.__Splitter.trainingSplit.return_value = [ ( MagicMock(), TestIds ) ]

        Predictions = MagicMock()

        self.__MLP.predict.return_value = Predictions
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Evaluator.score.call_args_list[ 0 ]

        self.assertEqual(
            Arguments[ 0 ],
            Predictions,
        )

        self.assertEqual(
            list( Arguments[ 1 ] ),
            list( self.__Data[ 'doid' ].filter( list( TestIds ) ) )
        )

        self.assertEqual(
            list( Arguments[ 2 ] ),
            ClassLabels
        )

        self.__Evaluator.score.assert_called_once()


    def test_it_finalizes_the_evaluation_for_each_fold( self ):
        self.__Splitter.trainingSplit.return_value = [
            ( MagicMock(), MagicMock() ),
            ( MagicMock(), MagicMock() ),
        ]

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = None,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.assertEqual(
            2,
            self.__Evaluator.finalize.call_count
        )
